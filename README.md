
# Hformer: Hybrid CNN-Transformer for Single Wrapped Phase to Fringe Order Prediction Fringe Projection 3D Measurement

Code used for Hformer: Hybrid CNN-Transformer for Single Wrapped Phase to Fringe Order Prediction Fringe Projection 3D Measurement.

The whole codes will be released after the paper has been completely reviewed or accepted.

## Implementation
Base on Python 3.7 and PaddlePaddle 2.0
<!-- ## Main Pipeline
![mainpipline](https://user-images.githubusercontent.com/84077203/137656145-ee630b3a-e9cd-4faf-9302-b3534bd9952f.png) -->

<!-- ## Representative Visual Results
![wrap139](https://user-images.githubusercontent.com/84077203/137653421-9d4baef7-0bc9-4c4a-affe-726cfe87a15c.png)
![unwrap139](https://user-images.githubusercontent.com/84077203/137655883-fd9c4e43-50fc-4b31-b394-ab166af21a70.png) -->
## Acknowledgement
codebase of CAT block from https://github.com/linhezheng19/CAT

codebase of Patch Expanding from https://github.com/HuCaoFighting/Swin-Unet
=======
# Hformer: Hybrid CNN-Transformer network for fringe order prediction in phase unwrapping of fringe projection

* The dataset used in this work is from reference [1], which includes 1000 wrapped phase and fringe order images with a resolution of 640×480 pixels. Both single object and multiple objects are considered in the dataset. The wrapped phase can be obtained with the phase shifting method as follows:

  ![image](https://user-images.githubusercontent.com/84077203/164176953-5828b832-9e1e-4af9-be7e-e3c5abb9ccfb.png)

  where  is wrapped phase, and   is the (n+1)-th captured fringe pattern image, n=0, 1,…, N-1.

# Note
* The code is tested with python=3.7 and paddlepaddle=2.0.2


# Begin to train
* python PaddleSeg/train.py --config PaddleSeg/configs/Hformer/Hformer.yml --do_eval --use_vdl

# Begin to test
* python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path WrapDataset_test_72/x_data --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96

# Begin to test with horizontal flip and vertical flip
* python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path WrapDataset_test_72/x_data --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96  --flip_horizontal --flip_vertical

# Requirements
* pip install -r requirements.txt

# References
* [1] W. Yin, Q. Chen, S. Feng, T. Tao, L. Huang, M. Trusiak, A. Asundi, and C. Zuo, “Temporal phase unwrapping using deep learning,” Sci Rep. 9(1),20175 (2019).
>>>>>>> 8346d995d0cb22f3890827adb1811e722c051846
