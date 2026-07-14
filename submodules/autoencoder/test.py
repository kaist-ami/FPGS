import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder.dataset_dynerf import Autoencoder_dataset
from model import Autoencoder
import torch.nn.functional as F
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    # default=[384, 128, 64, 32, 16],
                    default=[384, 384, 384, 384, 16],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    # default=[16, 32, 64, 128, 256, 256, 384],
                    default=[512, 512, 512, 512, 384],
                    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = f"{args.checkpoint}/best_ckpt.pth"

    data_dir = f"{dataset_path}/dino_features"
    output_dir = f"{dataset_path}/dino_features_dim3"

    save_path = f'{dataset_path}/dino_features_16'
    os.makedirs(save_path, exist_ok=True)

    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)


    checkpoint = torch.load(ckpt_path)

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    data_imgs = os.listdir(data_dir)
    for i in tqdm(range(len(data_imgs))):
        # if i > 32:
        #     continue
        load_img = os.path.join(data_dir, data_imgs[i])
        feature = torch.load(load_img).float()
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode_normalize(data)  
            pred = model.decode_denormalize(outputs)  

            # outputs /= 15
            torch.save(outputs.cpu(), f'{save_path}/{data_imgs[i]}')
        # PCA
            outputs = pred
            outputs_1d = outputs.reshape(outputs.shape[1],-1).permute(1,0)
            # outputs_1d = pred.reshape(384,-1).permute(1,0)
            U, S, V = torch.pca_lowrank(outputs_1d, q=3)
            # torch.save(V.cpu(), 'pca_mat_block_19.pt')
            # V = torch.load('pca_mat_block_0.pt')
            # V = torch.load('pca_mat_block_0_cos.pt')
            projection = torch.matmul(outputs_1d, V.cuda())
            # projection_norm = (projection - torch.min(projection)) / (torch.max(projection) - torch.min(projection))
            # projection_norm = (projection + 0.75) / (1.5)
            projection_norm = (projection + 20) / (40)
            # projection_norm = projection_norm.reshape(1, 95, 127, 3).permute(0,3,1,2)
            projection_norm = projection_norm.reshape(1,outputs.shape[2],outputs.shape[3],3).permute(0,3,1,2)
            # torchvision.utils.save_image(projection_norm, 'block_13_pca/{}.jpg'.format(img_path[:-4]))
            torchvision.utils.save_image(projection_norm, 'colmap_pca_highdim/{}.jpg'.format(data_imgs[i][:-3]))




