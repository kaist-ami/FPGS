#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render, render_f
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from submodules.adain.decoder import VGG_linear, decoder_single_128
import torch.nn as nn
import submodules.adain.VGGNet as VGGNet
from submodules.dino.autoencoder.model import Autoencoder
from PIL import Image

from style_transfer_utils import *
import time
import glob

torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)



def sh_style_transfer(gaussians, source_path, transfer_settings):

    cnn_encoder = transfer_settings["cnn_encoder"]
    stylize_iterations = transfer_settings['stylize_iterations']
    semantic_model = transfer_settings["semantic_model"]

    semantic_dim = 384

    mlp_encoder = VGG_linear()
    mlp_encoder.load_state_dict(torch.load('submodules/adain/ckpts/vgg_linear.pth'))
    mlp_encoder.to('cuda:0')
    mlp_encoder.eval()

    vgg_decoder = VGGNet.decoder
    vgg = VGGNet.vgg
    vgg.load_state_dict(torch.load('submodules/adain/ckpts/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg_encoder_cnn = VGGNet.Net(vgg, vgg_decoder)
    vgg_encoder_cnn.to('cuda:0')
    vgg_encoder_cnn.eval()

    mlp_decoder = decoder_single_128()
    mlp_decoder.eval()
    mlp_decoder.load_state_dict(torch.load('submodules/adain/ckpts/decoder.pth'))

    encoder_hidden_dims = [384, 384, 384, 384, 16]
    decoder_hidden_dims = [512, 512, 512, 512, semantic_dim]
                
    
    semantic_autoencoder = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, semantic_model).to("cuda:0")
    ckpt_path = f"{source_path}/dino_autoencoder/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path)
    semantic_autoencoder.load_state_dict(checkpoint)
    semantic_autoencoder.eval()

    ######## make style dictionary ###########

    num_clusters = 10
    blending_ratio = transfer_settings['blending_ratio']
    temperature = transfer_settings['temperature']

    local_vgg_mean_1d_list = []
    local_vgg_std_1d_list = []
    local_semantic_feature_1d_list = []


    with torch.no_grad():

        for style_im_path in transfer_settings['style_img']:
            style_im = Image.open(style_im_path).convert('RGB')

            style_vgg_features = get_vgg_features_cnn(style_im, vgg_encoder_cnn)

            style_semantic_features = get_semantic_features(style_im)

            clustered_vgg_mean, clustered_vgg_std, clustered_semantic_features = style_im_clustering(style_vgg_features, style_semantic_features, num_clusters, blending_ratio)
            local_vgg_mean_1d_list.append(clustered_vgg_mean)
            local_vgg_std_1d_list.append(clustered_vgg_std)
            local_semantic_feature_1d_list.append(clustered_semantic_features)

        local_vgg_mean_1d = torch.cat(local_vgg_mean_1d_list, 0)
        local_vgg_std_1d = torch.cat(local_vgg_std_1d_list, 0)
        style_dino_feature_1d = torch.cat(local_semantic_feature_1d_list, 0).cuda()

        shs = gaussians.get_features
        cov = gaussians.get_covariance()

       
        cov_mat = torch.zeros(cov.shape[0],3,3)
        cov_mat[:,0,:] = cov[:,:3] 

        
        SH_C0 = [0.28209479177387814]

        offset = 0.5

        semantic_features = semantic_autoencoder.decode(gaussians.get_semantic_features.permute(1,0).unsqueeze(1)).squeeze().permute(1,0)

        ####### style matching #######

        num_shs = shs.shape[0]
        softmax = nn.Softmax(dim=1)
        eps = 0.01



        for stylize_iter in range(stylize_iterations):

            shs_defuse = shs[:,0,:]
            shs_defuse = shs_defuse * SH_C0[0] + offset
            shs_defuse_clamp = torch.maximum(torch.tensor(0),shs_defuse)
            shs_defuse_clamp = torch.minimum(torch.tensor(1),shs_defuse_clamp)

            content_features = mlp_encoder(shs_defuse_clamp)

            valid_gaussians = gaussians.get_opacity > 0
            content_mean = torch.mean(content_features[valid_gaussians.squeeze()], 0, keepdim=True)
            content_std = torch.std(content_features[valid_gaussians.squeeze()], 0, keepdim=True)
        

            batch_size = 2 ** 16
            batch_offset = 0

            for i in range(num_shs//batch_size + 1):
                if i * batch_size > num_shs:
                    temp_batch_size = num_shs - (i-1) * batch_size
                else:
                    temp_batch_size = batch_size
                ####### semantic matching #########
                
                content_feature_batch = content_features[batch_offset:batch_offset+temp_batch_size]
                semantic_feature_batch = semantic_features[batch_offset:batch_offset+temp_batch_size]

                corr_mat = torch.matmul(semantic_feature_batch, style_dino_feature_1d.T)
                corr_mat_soft = softmax(corr_mat/temperature)

                weighted_vgg_mean = torch.matmul(corr_mat_soft, local_vgg_mean_1d)
                weighted_vgg_std = torch.matmul(corr_mat_soft, local_vgg_std_1d)
                weighted_vgg_mean = torch.nan_to_num(weighted_vgg_mean)
                weighted_vgg_std = torch.nan_to_num(weighted_vgg_std)

                content_normalized = (content_feature_batch - content_mean) / (content_std + eps)
                adain_features = content_normalized * weighted_vgg_std + weighted_vgg_mean
                adain_shs = mlp_decoder(adain_features)
                shs[batch_offset:batch_offset+temp_batch_size,0,:] = (adain_shs - offset) / SH_C0[0]

                batch_offset += temp_batch_size

        content_mean = []
        content_std = []
    return shs[:,0,:], shs[:,1:,:], content_mean, content_std



def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, transfer_settings, style_transfer_3D=False):
    save_ply = transfer_settings['save_ply']

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    os.makedirs(render_path, exist_ok=True)

    offset = 0.5

    if style_transfer_3D:
        shs_defuse, shs_spec, content_mean, content_std = sh_style_transfer(gaussians, source_path, transfer_settings)
        gaussians._features_dc = shs_defuse.unsqueeze(1)
        gaussians._features_rest = shs_spec


        if save_ply:
            gaussians.save_ply(os.path.join(model_path, 'point_cloud_transferred', transfer_settings['style_dir'].split('/')[-1], "point_cloud.ply"))
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background)["render"]

            os.makedirs(os.path.join(render_path, transfer_settings['style_dir'].split('/')[-1]), exist_ok=True)
            torchvision.utils.save_image(rendering, os.path.join(render_path, transfer_settings['style_dir'].split('/')[-1], '{0:05d}'.format(idx) + ".jpg"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_spiral : bool, transfer_settings, style_transfer_3D=False, ):
    with torch.no_grad():
        
        start_time = time.time()

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        checkpoint = os.path.join(dataset.model_path, f"chkpnt{scene.loaded_iter}.pth")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore_eval(model_params)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        end_time = time.time()
        print(f'Running time : {end_time - start_time}')
        render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, transfer_settings, style_transfer_3D)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_spiral", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cnn_encoder", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--stylize_iterations", default=2, type=int)
    parser.add_argument("--style_dir", default='styles', type=str)
    parser.add_argument("--style_transfer_3D", default=True, type=bool)
    parser.add_argument("--temperature", default=100, type=int)
    parser.add_argument("--blending_ratio", default=0.3, type=float)
    parser.add_argument("--semantic_model", default='dino', type=str)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    transfer_settings = {}
    transfer_settings['cnn_encoder'] = args.cnn_encoder
    transfer_settings['save_ply'] = args.save_ply
    transfer_settings['stylize_iterations'] = args.stylize_iterations
    transfer_settings['temperature'] = args.temperature
    transfer_settings['blending_ratio'] = args.blending_ratio
    transfer_settings['style_dir'] = args.style_dir
    transfer_settings["semantic_model"] = args.semantic_model
    style_imgs = glob.glob(f'{args.style_dir}/*')
    transfer_settings['style_img'] = style_imgs


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_spiral, transfer_settings, args.style_transfer_3D)