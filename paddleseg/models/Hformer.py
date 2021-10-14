import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from einops import rearrange

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

def to_2tuple(x):
    return tuple([x] * 2)

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

def partition(x, patch_size):
    """
    Args:
        x: (B, H, W, C)
        patch_size (int): patch size

    Returns:
        patches: (num_patches*B, patch_size, patch_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // patch_size, patch_size, W // patch_size, patch_size, C))
    patches = x.transpose((0, 1, 3, 2, 4, 5)).reshape((-1, patch_size, patch_size, C))
    return patches


def reverse(patches, patch_size, H, W):
    """
    Args:
        patches: (num_patches*B, patch_size, patch_size, C)
        patch_size (int): Patch size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(patches.shape[0] / (H * W / patch_size / patch_size))
    x = patches.reshape((B, H // patch_size, W // patch_size, patch_size, patch_size, -1))
    x = x.transpose((0, 1, 3, 2, 4, 5)).reshape((B, H, W, -1))
    return x


class Mlp(nn.Layer):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchProjection(nn.Layer):
    """ Patch Projection Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape((B, H, W, C))

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = x.reshape((B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape((B, -1, 4 * C))  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class PatchExpand(nn.Layer):
    
    '''    
    The PatchExpand implementation based on PaddlePaddle.

    The original article refers to "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"
    (https://arxiv.org/abs/2105.05537)
    
    '''

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape([B, H, W, C])
        x = paddle.to_tensor(rearrange(x.numpy(), 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4))
        x = x.reshape([B,-1,C//4])
        x= self.norm(x)

        return x

class Attention(nn.Layer):
    """ Basic attention of IPSA and CPSA.

    Args:
        dim (int): Number of input channels.
        patch_size (tuple[int]): Patch size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        attn_drop (float, optional): Dropout ratio of attention weight.
        proj_drop (float, optional): Dropout ratio of output.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe

        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = self.create_parameter(
                shape=((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1),
                    num_heads),
                default_initializer=zeros_)# 2*Ph-1 * 2*Pw-1, nH
            self.add_parameter("relative_position_bias_table",
                            self.relative_position_bias_table)
        
            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(self.patch_size[0])
            coords_w = paddle.arange(self.patch_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h,
                                                coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
            coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
            relative_coords = coords_flatten_1 - coords_flatten_2
            relative_coords = relative_coords.transpose((1, 2, 0))

            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            
            trunc_normal_(self.relative_position_bias_table)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)
        
    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_patches*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape((B_, N, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose((0, 1, 3, 2)))

        if self.rpe:
            index = self.relative_position_index.reshape([-1])
            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index)

            relative_position_bias = relative_position_bias.reshape([
                self.patch_size[0] * self.patch_size[1],
                self.patch_size[0] * self.patch_size[1], -1
            ])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0, 2, 1, 3)).reshape((B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CATBlock(nn.Layer):

    '''    
    The CAT Block implementation based on PaddlePaddle.
    The original article refers to "CAT: Cross Attention in Vision Transformer"
    (https://arxiv.org/abs/2106.05786)
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        attn_drop (float, optional): Attention dropout rate.
        drop_path (float, optional): Stochastic depth rate.
        act_layer (nn.Layer, optional): Activation layer.
        norm_layer (nn.Layer, optional): Normalization layer.
        rpe (bool): Use relative position encoding or not.
    '''

    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type="ipsa", rpe=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim if attn_type == "ipsa" else self.patch_size ** 2, patch_size=to_2tuple(self.patch_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # need to be changed in different stage during forward phase
        self.H = None
        self.W = None
        
    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape((B, H, W, C))
        
        # padding to multiple of patch size in each layer
        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size

        x = x.transpose([0, 3, 1, 2])
        x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
        x = x.transpose([0, 2, 3, 1])

        _, Hp, Wp, _ = x.shape

        # partition
        patches = partition(x, self.patch_size)  # nP*B, patch_size, patch_size, C
        patches = patches.reshape((-1, self.patch_size * self.patch_size, C))  # nP*B, patch_size*patch_size, C

        # IPSA or CPSA
        if self.attn_type == "ipsa":
            attn = self.attn(patches)  # nP*B, patch_size*patch_size, C
        elif self.attn_type == "cpsa":
            patches = patches.reshape((B, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2, C)).transpose((0, 3, 1, 2))
            patches = patches.reshape((-1, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2)) # nP*B*C, nP*nP, patch_size*patch_size
            attn = self.attn(patches).reshape((B, C, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2))
            attn = attn.transpose((0, 2, 3, 1)).reshape((-1, self.patch_size ** 2, C)) # nP*B, patch_size*patch_size, C
        else :
            raise NotImplementedError(f"Unkown Attention type: {self.attn_type}")

        # reverse opration of partition
        attn = attn.reshape((-1, self.patch_size, self.patch_size, C))
        x = reverse(attn, self.patch_size, Hp, Wp)  # B H' W' C
        
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape((B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        
class CATLayer(nn.Layer):
    """ Basic CAT layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size of IPSA or CPSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        ipsa_attn_drop (float): Attention dropout rate of InnerPatchSelfAttention.
        cpsa_attn_drop (float): Attention dropout rate of CrossPatchSelfAttention.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
        norm_layer (nn.Layer, optional): Normalization layer.
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
    """

    def __init__(self, dim, depth, num_heads, patch_size, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., ipsa_attn_drop=0., cpsa_attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth


        # build blocks
        self.pre_ipsa_blocks = nn.LayerList()
        self.cpsa_blocks = nn.LayerList()
        self.post_ipsa_blocks = nn.LayerList()
        for i in range(depth):
            self.pre_ipsa_blocks.append(CATBlock(dim=dim, num_heads=num_heads, patch_size=patch_size,
                                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                 qk_scale=qk_scale, drop=drop,
                                                 attn_drop=ipsa_attn_drop, drop_path=drop_path[i],
                                                 norm_layer=norm_layer, attn_type="ipsa", rpe=True))

            self.cpsa_blocks.append(CATBlock(dim=dim, num_heads=1, patch_size=patch_size,
                                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                             qk_scale=qk_scale, drop=drop,
                                             attn_drop=cpsa_attn_drop, drop_path=drop_path[i],
                                             norm_layer=norm_layer, attn_type="cpsa", rpe=False))

            self.post_ipsa_blocks.append(CATBlock(dim=dim, num_heads=num_heads, patch_size=patch_size,
                                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                  qk_scale=qk_scale, drop=drop,
                                                  attn_drop=ipsa_attn_drop, drop_path=drop_path[i],
                                                  norm_layer=norm_layer, attn_type="ipsa", rpe=True))

        # patch projection layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        num_blocks = len(self.cpsa_blocks)
        for i in range(num_blocks):
            self.pre_ipsa_blocks[i].H, self.pre_ipsa_blocks[i].W = H, W
            self.cpsa_blocks[i].H, self.cpsa_blocks[i].W = H, W
            self.post_ipsa_blocks[i].H, self.post_ipsa_blocks[i].W = H, W


            x = self.pre_ipsa_blocks[i](x)
            x = self.cpsa_blocks[i](x)
            x = self.post_ipsa_blocks[i](x)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
        return x


class PatchEmbedding(nn.Layer):
    """ Image to Patch Embedding

    Args:
        patch_emb_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Layer, optional): Normalization layer.
    """

    def __init__(self,patch_emb_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_emb_size = to_2tuple(patch_emb_size)
        self.patch_emb_size = patch_emb_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_emb_size, stride=1,padding = 1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.shape
        if W % self.patch_emb_size[1] != 0:
            x = F.pad(x, (0, self.patch_emb_size[1] - W % self.patch_emb_size[1]))
        if H % self.patch_emb_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_emb_size[0] - H % self.patch_emb_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            _,_,Wh, Ww = x.shape
            x = x.flatten(2).transpose((0,2,1))
            x = self.norm(x)
            x = x.transpose((0,2,1)).reshape((-1, self.embed_dim, Wh, Ww))

        return x

@manager.BACKBONES.add_component
class Hformer(nn.Layer):

    """
    The Hformer implementation based on PaddlePaddle.
    """

    def __init__(self,  
                pre_trained_img_size=384,
                patch_emb_size=4,
                patch_size = 7,
                in_channels = 1,
                num_classes = 34,
                embed_dims = [64, 128, 256, 512],
                bone_dims = [18,36,72,144],
                drop_path_rate = 0.3,
                ipsa_attn_drop=0., cpsa_attn_drop=0.,
                depths = [2, 2, 2, 2],
                num_heads=[1, 2, 4, 8],
                mlp_ratio=4,
                qkv_bias = True,
                qk_scale = None,
                drop_rate = 0.0,
                attn_drop_rate = 0.0,
                out_indices=(0, 1, 2, 3, 4, 5, 6, 7),
                norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                frozen_stages=-1,
                backbone = None,
                pretrained = None
    ):
        super(Hformer, self).__init__()

        self.num_classes = num_classes
        self.depths = depths

        self.pre_trained_img_size = pre_trained_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.use_ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.backbone = backbone
        emd = embed_dims[0] + embed_dims[1] + embed_dims[2] + embed_dims[3]
        self.cls = nn.Sequential(nn.Conv2D(emd,emd//2,kernel_size = 3,stride = 1,padding = 1),
                                 nn.ReLU(),
                                 nn.Conv2D(emd//2,num_classes,kernel_size = 1,stride = 1,padding = 0))

        # split image into non-overlapping patches
        self.patch_embed1 = PatchEmbedding(in_chans=bone_dims[0], embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed2 = PatchEmbedding(in_chans=bone_dims[1], embed_dim=embed_dims[1],
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed3 = PatchEmbedding(in_chans=bone_dims[2], embed_dim=embed_dims[2],
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed4 = PatchEmbedding(in_chans=bone_dims[3], embed_dim=embed_dims[3],
            norm_layer=norm_layer if self.patch_norm else None)


        # absolute position embedding
        if self.use_ape:
            pre_trained_img_size = to_2tuple(pre_trained_img_size)
            patch_emb_size = to_2tuple(patch_emb_size)
            patches_resolution = [pre_trained_img_size[0] // patch_emb_size[0], pre_trained_img_size[1] // patch_emb_size[1]]

            self.ape1 = self.create_parameter(
                shape=(1, embed_dims[0], patches_resolution[0],
                       patches_resolution[1]),
                default_initializer=zeros_)
            self.add_parameter("ape1", self.ape1)
            trunc_normal_(self.ape1)

            self.ape2 = self.create_parameter(
                shape=(1, embed_dims[1], patches_resolution[0]//2,
                       patches_resolution[1]//2),
                default_initializer=zeros_)
            self.add_parameter("ape2", self.ape2)
            trunc_normal_(self.ape2)

            self.ape3 = self.create_parameter(
                shape=(1, embed_dims[2], patches_resolution[0]//4,
                       patches_resolution[1]//4),
                default_initializer=zeros_)
            self.add_parameter("ape3", self.ape3)
            trunc_normal_(self.ape3)

            self.ape4 = self.create_parameter(
                shape=(1, embed_dims[3], patches_resolution[0]//8,
                       patches_resolution[1]//8),
                default_initializer=zeros_)
            self.add_parameter("ape4", self.ape4)
            trunc_normal_(self.ape4)

        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_drop4 = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()

        # build encoders
        self.encoders = nn.LayerList()
        for i_layer in range(self.num_layers):
            self.encoders.append(CATLayer(dim=int(self.embed_dim * 2 ** i_layer),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        patch_size=patch_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop_rate,
                                        ipsa_attn_drop=ipsa_attn_drop,
                                        cpsa_attn_drop=cpsa_attn_drop,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchProjection if (i_layer < self.num_layers - 1) else None))

        # build encoders
        self.decoders = nn.LayerList()
        for i_layer in range(self.num_layers):
            self.decoders.append(CATLayer(dim=int(self.embed_dim * 2 ** i_layer),
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        patch_size=patch_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop_rate,
                                        ipsa_attn_drop=ipsa_attn_drop,
                                        cpsa_attn_drop=cpsa_attn_drop,
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=None))

        self.expand1 = PatchExpand(self.embed_dim* 2 ** 3)
        self.expand2 = PatchExpand(self.embed_dim* 2 ** 2)
        self.expand3 = PatchExpand(self.embed_dim* 2 ** 1)
        self.expand4 = PatchExpand(self.embed_dim* 2 ** 0)                           

        self.concat_linear1 = nn.Sequential(nn.Linear(2*embed_dims[1],embed_dims[1]))
        self.concat_linear2 = nn.Sequential(nn.Linear(2*embed_dims[2],embed_dims[2]))
        self.concat_linear3 = nn.Sequential(nn.Linear(2*embed_dims[3],embed_dims[3]))

        self.concat_linear4 = nn.Sequential(nn.Linear(2*embed_dims[2],embed_dims[2]))
        self.concat_linear5 = nn.Sequential(nn.Linear(2*embed_dims[1],embed_dims[1]))
        self.concat_linear6 = nn.Sequential(nn.Linear(2*embed_dims[0],embed_dims[0]))

        num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
        num_features.append(int(self.embed_dim* 2 ** 3))
        num_features.append(int(self.embed_dim* 2 ** 2))
        num_features.append(int(self.embed_dim* 2 ** 1))
        num_features.append(int(self.embed_dim* 2 ** 0))

        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_sublayer(layer_name, layer)
            
        self._freeze_stages()

        self.pretrained = pretrained
        self.init_weights(self.pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.use_ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                layer = self.encoders[i]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
            for i in range(0, self.frozen_stages - 1):
                layer = self.decoders[i]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Linear):
                    trunc_normal_(sublayer.weight)
                    if isinstance(sublayer,
                                  nn.Linear) and sublayer.bias is not None:
                        zeros_(sublayer.bias)
                elif isinstance(sublayer, nn.LayerNorm):
                    zeros_(sublayer.bias)
                    ones_(sublayer.weight)

    def forward(self, x):

        y = x
        B = x.shape[0]
        
        b4,b3,b2,b1 = self.backbone(x)

        _,_,Wh, Ww = b1.shape
        if self.use_ape:
            # interpolate the absolute position encoding to the corresponding size
            ape = F.interpolate(self.ape1, size=(Wh, Ww), mode='bicubic')
            b1 = (b1 + ape).flatten(2).transpose((0,2,1))  # B Wh*Ww C
        else:
            b1 = b1.flatten(2).transpose((0,2,1))
        b1 = self.pos_drop1(b1)


        _,_,Wh, Ww = b2.shape
        if self.use_ape:
            # interpolate the absolute position encoding to the corresponding size
            ape = F.interpolate(self.ape2, size=(Wh, Ww), mode='bicubic')
            b2 = (b2 + ape).flatten(2).transpose((0,2,1))  # B Wh*Ww C
        else:
            b2 = b2.flatten(2).transpose((0,2,1))
        b2 = self.pos_drop2(b2)

        _,_,Wh, Ww = b3.shape
        if self.use_ape:
            # interpolate the absolute position encoding to the corresponding size
            ape = F.interpolate(self.ape3, size=(Wh, Ww), mode='bicubic')
            b3 = (b3 + ape).flatten(2).transpose((0,2,1))  # B Wh*Ww C
        else:
            b3 = b3.flatten(2).transpose((0,2,1))
        b3 = self.pos_drop3(b3)

        _,_,Wh, Ww = b4.shape
        if self.use_ape:
            # interpolate the absolute position encoding to the corresponding size
            ape = F.interpolate(self.ape4, size=(Wh, Ww), mode='bicubic')
            b4 = (b4 + ape).flatten(2).transpose((0,2,1))  # B Wh*Ww C
        else:
            b4 = b4.flatten(2).transpose((0,2,1))
        b4 = self.pos_drop4(b4)

        #encoder 1
        z = b1
        layer = self.encoders[0]
        x_out, H, W, z, Wh, Ww = layer(z, Wh*8, Ww*8)
        e1 = x_out
        #encoder 2
        layer = self.encoders[1]
        z = paddle.concat([z,b2],-1)
        z = self.concat_linear1(z)
        x_out, H, W, z, Wh, Ww = layer(z, Wh, Ww)
        e2 = x_out        
        #encoder 3
        layer = self.encoders[2]
        z = paddle.concat([z,b3],-1)
        z = self.concat_linear2(z)
        x_out, H, W, z, Wh, Ww = layer(z, Wh, Ww)
        e3 = x_out   
        #encoder 4
        layer = self.encoders[3]
        z = paddle.concat([z,b4],-1)
        z = self.concat_linear3(z)
        x_out, H, W, z, Wh, Ww = layer(z, Wh, Ww)
        e4 = x_out   

        #decoder 1
        z = e4
        layer = self.decoders[3]
        x_out, H, W, z, Wh, Ww = layer(z, Wh, Ww)
        norm_layer = getattr(self, f'norm{4}')
        x_out = norm_layer(x_out)
        d4 = x_out.reshape((B, Wh, Ww,-1)).transpose((0, 3, 1, 2))

        #decoder 2
        z = paddle.concat([self.expand1(z,Wh,Wh),e3],-1)
        z = self.concat_linear4(z)
        layer = self.decoders[2]
        x_out, H, W, z, Wh, Ww = layer(z, 2*Wh, 2*Ww)
        norm_layer = getattr(self, f'norm{5}')
        x_out = norm_layer(x_out)
        d3 = x_out.reshape((B, Wh, Ww,-1)).transpose((0, 3, 1, 2))      
  
        #decoder 3
        z = paddle.concat([self.expand2(z,Wh,Wh),e2],-1)
        z = self.concat_linear5(z)
        layer = self.decoders[1]
        x_out, H, W, z, Wh, Ww = layer(z, 2*Wh, 2*Ww)   
        norm_layer = getattr(self, f'norm{6}')
        x_out = norm_layer(x_out)
        d2 = x_out.reshape((B, Wh, Ww,-1)).transpose((0, 3, 1, 2))         

        #decoder 4
        z = paddle.concat([self.expand3(z,Wh,Wh),e1],-1)
        z = self.concat_linear6(z)
        layer = self.decoders[0]
        x_out, H, W, z, Wh, Ww = layer(z, 2*Wh, 2*Ww)  
        norm_layer = getattr(self, f'norm{7}')
        x_out = norm_layer(x_out)
        d1 = x_out.reshape((B, Wh, Ww,-1)).transpose((0, 3, 1, 2))

        x0_h, x0_w = d1.shape[2:]
        d2 = F.interpolate(
            d2, (x0_h, x0_w),
            mode='bilinear',
            align_corners=False)
        d3 = F.interpolate(
            d3, (x0_h, x0_w),
            mode='bilinear',
            align_corners=False)
        d4 = F.interpolate(
            d4, (x0_h, x0_w),
            mode='bilinear',
            align_corners=False)

        d = paddle.concat([d1, d2, d3, d4], 1)

        d = self.cls(d)
        
        logit_list = [d]

        logit_list = [
            F.interpolate(
                logit, paddle.shape(y)[2:], mode='bilinear', align_corners=False)
            for logit in logit_list
        ]        
        return logit_list

