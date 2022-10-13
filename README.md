
<!-- ## Main Pipeline
![mainpipline](https://user-images.githubusercontent.com/84077203/137656145-ee630b3a-e9cd-4faf-9302-b3534bd9952f.png) -->

<!-- ## Representative Visual Results
![wrap139](https://user-images.githubusercontent.com/84077203/137653421-9d4baef7-0bc9-4c4a-affe-726cfe87a15c.png)
![unwrap139](https://user-images.githubusercontent.com/84077203/137655883-fd9c4e43-50fc-4b31-b394-ab166af21a70.png) -->


# Xinjun Zhu, Zhiqiang Han, Mengkai Yuan, Qinghua Guo, Hongyi Wang, Limei Song, "Hformer: hybrid convolutional neural network transformer network for fringe order prediction in phase unwrapping of fringe projection," Opt. Eng. 61(9), 093107 (2022), doi: 10.1117/1.OE.61.9.093107.

<!-- ## We show the fringe pattern, the wrapped phase, the fringe order and the unwrapped phase in Fig. \ref{dataset}. The wrapped phase can be obtained with the phase shifting method as follows:
![1](https://user-images.githubusercontent.com/84077203/167300823-ed646543-5712-441c-b062-727841b5e9ea.png)
![2](https://user-images.githubusercontent.com/84077203/167300824-3d545c7a-3c61-4583-9a15-bd00d0b15332.png)
-->

# Note
* The code is tested with python=3.7 and paddlepaddle=2.0.2


# Begin to train
* python PaddleSeg/train.py --config PaddleSeg/configs/Hformer/Hformer.yml --do_eval --use_vdl

# Begin to test
* python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path dataset path --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96

# Begin to test with horizontal flip and vertical flip
* python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path dataset path --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96  --flip_horizontal --flip_vertical

# Requirements
* pip install -r requirements.txt

# References of datasets
* [1]  Qian, J., Feng, S., Tao, T., Hu, Y., Li, Y., Chen, Q., Zuo, C.: Deep learning-enabled geometric constraints and phase unwrapping for single shot absolute 3d shape measurement. Apl Photonics 5(4), 046105 (2020). https://doi.org/10.1063/5.0003217
* [2] Zuo, C., Qian, J., Feng, S. et al. Deep learning in optical metrology: a review. Light Sci Appl 11, 39 (2022). https://doi.org/10.1038/s41377-022-00714-x
# Acknowledgement
* codebase of CAT block from https://github.com/linhezheng19/CAT
* codebase of Patch Expanding from https://github.com/HuCaoFighting/Swin-Unet

# Citation
Xinjun Zhu, Zhiqiang Han, Mengkai Yuan, Qinghua Guo, Hongyi Wang, Limei Song, "Hformer: hybrid convolutional neural network transformer network for fringe order prediction in phase unwrapping of fringe projection," Opt. Eng. 61(9), 093107 (2022), doi: 10.1117/1.OE.61.9.093107.
