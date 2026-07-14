# FPGS (IJCV 2026)
### [Project Page](https://kim-geonu.github.io/FPGS/) | [arXiv](https://arxiv.org/abs/2503.09635)
This repository contains a pytorch implementation for the IJCV 2026 paper, [FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting](https://arxiv.org/abs/2503.09635).



https://github.com/user-attachments/assets/37dd9979-4f39-410b-89da-5d420b95e3d4



## Getting Started
This code was developed on Ubuntu 24.04 with Python 3.9, CUDA 11.8 and PyTorch 2.0.0. Later versions should work, but have not been tested.

### Environment Setup
Create and activate a virtual environment, then install `pytorch` and `tiny-cuda-nn`:
```
conda create -n FPGS python=3.9
conda activate FPGS
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

### Download calibrated datasets
To run FPGS, please download the [calibrated scenes](https://drive.google.com/drive/folders/1l45X5sgjf134KRJkyiXLnPvL4RBIz45l) provided by [StyleGaussian](https://github.com/Kunhao-Liu/StyleGaussian) and put it in ./data. 

## Preprocess
Run below commands to extract semantic features from the scene and trian autoencoder for compressing semantic features. 
```bash
python preprocess.py --data_path data/truck 
```

## Training Scene
Run below commands to train a stylizable 3D scene. 
```bash
python train_features.py -s data/truck -m output/truck
```

## Style transfer 
Run below commands to transfer the style of a 3D scene to the refernece images in ./styles.

```bash
python render_features.py -m output/truck --save_ply --style_dir styles
```
<details><summary>Here are the controllable hyperparameters.</summary>
      
```bash
python render_features.py -m output/truck --semantic_model dino --save_ply --style_dir styles --temperature 100 --blending_ratio 0.3 --stylize_iterations 2 
```

* temperature - Temperature of softmax operation for semantic correspondence matching. 

* local_global_blending_ratio - Ratio of global style feature for style transfer. 1 refers using only global style features and 0 refers using only local style features.

* stylize_iterations - Number of iterative style transfer

</details>

You can also use a 3D Gaussian viewer to render the transferred scene, which can be saved by `--save_ply` option.

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@article{kim2026fpgs,
  title={FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting},
  author={Kim, GeonU and Youwang, Kim and Hyoseok, Lee and Oh, Tae-Hyun},
  journal={International Journal of Computer Vision},
  volume={134},
  number={5},
  pages={214},
  year={2026},
  publisher={Springer}
}
````
