# Hformer
Hformer: Hybrid CNN-Transformer network for fringe order prediction in phase unwrapping of fringe projection

Dataset used for training, validation and testing in this paper are available in W. Yin, Q. Chen, S. Feng, T. Tao, L. Huang, M. Trusiak, A. Asundi, and C. Zuo, “Temporal phase unwrapping using deep learning,” Sci Rep. 9(1),20175 (2019)..

# Begin to train
!python PaddleSeg/train.py --config PaddleSeg/configs/Hformer/Hformer.yml --do_eval --use_vdl

# Begin to test
!python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path WrapDataset_test_72/x_data --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96

# Begin to test with horizontal flip and vertical flip
!python PaddleSeg/predict.py --config PaddleSeg/configs/Hformer/Hformer.yml --model_path output/best_model/model.pdparams --image_path WrapDataset_test_72/x_data --save_dir result --aug_pred  --is_slide --crop_size 384 384 --stride 256 96  --flip_horizontal --flip_vertical
