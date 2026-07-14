# CUDA_VISIBLE_DEVICES=0 python -u preprocess.py --data_path data/truck 
# CUDA_VISIBLE_DEVICES=0 python train_features.py -s data/truck -m output/truck
# CUDA_VISIBLE_DEVICES=0 python render_features.py -m output/truck --save_ply --style_dir styles
CUDA_VISIBLE_DEVICES=0 python preprocess.py --data_path data/truck 
CUDA_VISIBLE_DEVICES=0 python train_features.py -s data/truck -m output/truck
CUDA_VISIBLE_DEVICES=0 python render_features.py -m output/truck --save_ply --style_dir styles_table