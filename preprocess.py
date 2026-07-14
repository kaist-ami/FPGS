# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import cv2
import random
import colorsys
from tqdm import tqdm

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import submodules.dino.vision_transformer as vits
from submodules.guided_filter.guided_filter import FastGuidedFilter2d
import torch.nn.functional as F

from submodules.autoencoder.dataset_get_tensor import Autoencoder_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from submodules.autoencoder.model import Autoencoder

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
upsample = nn.UpsamplingBilinear2d(scale_factor=8)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=1).mean()



def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


def get_feature_map(model, img):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    features = model.get_features(img.to(device))
    features = features.reshape(1, 113, 113, features.shape[-1])

    return features


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--model_name", default="dino", type=str)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    transform = pth_transforms.Compose([
        pth_transforms.Resize([448,448]),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_path = args.data_path
    data_dir = os.listdir(data_path)
    img_list = []


    img_dir_path = os.path.join(data_path, f'images')
    img_name_list = os.listdir(img_dir_path)
    img_length = len(img_name_list)

    img_sample_path = os.path.join(img_dir_path, img_name_list[0])
    img_sample = cv2.imread(img_sample_path)
    h, w, _ = img_sample.shape
    with torch.no_grad():
        
        downsample = 1
        img_index = 0
        highdim_features = torch.zeros(img_length, 384, h//downsample, w//downsample)
        img_names = [None] * img_length

        for img_name in tqdm(img_name_list):
            img_path = os.path.join(img_dir_path, img_name)

        
            style_img_origin = Image.open(img_path)
            style_img_origin = style_img_origin.convert('RGB')
            w_origin, h_origin = style_img_origin.size
            style_img = transform(style_img_origin).unsqueeze(0).cuda()


            down_height = h//downsample
            down_width = w//downsample


            style_feature_map = get_feature_map(model, style_img.squeeze())
            fliped_style_feature_map = get_feature_map(model, flip_image(style_img).squeeze())
            fliped_style_feature_map_2d = fliped_style_feature_map.reshape(1, 113, 113, 384).permute(0,3,1,2)
            style_feature_map += flip_image(fliped_style_feature_map_2d).permute(0,2,3,1).reshape(style_feature_map.shape)
            style_feature_map /= 2


            radius=30
            eps=1e-3

            GF = FastGuidedFilter2d(radius, eps, s=2)

            tch_img = style_img
            tch_mask = style_feature_map.permute(0,3,1,2)
            tch_mask = F.interpolate(tch_mask, size=(style_img.shape[2], style_img.shape[3]), mode='bilinear')


            out = GF(tch_mask, tch_img)
            if torch.sum(out.isnan()) > 0:
                print('nan! {}'.format(img_path))
            out = torch.nan_to_num(out)
            out = F.interpolate(out, (down_height, down_width), mode='bicubic') # downsampling
            

            highdim_features[img_index] = out
            img_names[img_index] = img_name
            
            img_index += 1


    
    lr = 0.001
    args = parser.parse_args()
    num_epochs = 5
    
    train_dataset = Autoencoder_dataset(highdim_features, downsample=1, crop_down=4)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=24,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=24,
        drop_last=False 
    )
    
    encoder_hidden_dims = [384, 384, 384, 384, 16]
    decoder_hidden_dims = [512, 512, 512, 512, 384]

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, model_name=args.model_name).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(f"{data_path}/{args.model_name}_autoencoder/", exist_ok=True)


    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in tqdm(enumerate(train_loader)):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx

        torch.save(model.state_dict(), f'{data_path}/{args.model_name}_autoencoder/best_ckpt.pth')
                
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))




    with torch.no_grad():
        model.eval()
        # highdim_features = highdim_features.cuda()
        print('Create low-dim features')
        for i in tqdm(range(highdim_features.shape[0])):
            lowdim_feature = model.encode_normalize(highdim_features[i].unsqueeze(0).cuda())
            lowdim_feature_upsample = F.interpolate(lowdim_feature, size=(h, w), mode='bicubic')
            os.makedirs(os.path.join(data_path, f'{args.model_name}_features_16'), exist_ok=True)
            torch.save(lowdim_feature_upsample, os.path.join(data_path, f'{args.model_name}_features_16', img_names[i][:-4] + '.pt'))





 