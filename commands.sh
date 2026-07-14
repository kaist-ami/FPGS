CUDA_VISIBLE_DEVICES=0 python -u preprocess.py --data_path data/garden 
CUDA_VISIBLE_DEVICES=0 python train_features.py -s data/garden -m output/garden
CUDA_VISIBLE_DEVICES=0 python render_features.py -m output/garden --save_ply --style_dir styles
