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
