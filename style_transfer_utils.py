
import torch
from torchvision import transforms
import torch.nn.functional as F
import submodules.dino.vision_transformer as vits
from submodules.guided_filter.guided_filter import FastGuidedFilter2d
from kmeans_pytorch import kmeans
import math
 


def flip_image_360(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()

        horizon_fliped = img.index_select(3, idx)
        idx = torch.arange(img.size(2)-1, -1, -1).type_as(img).long()

    
    return horizon_fliped.index_select(2, idx)

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)

def get_dino_feature_map(model, img, patch_size = 8):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # attentions = model.get_last_selfattention(img.to(device))
    features = model.get_features(img.to(torch.device("cuda")))
    # features = features.reshape(1, w_featmap, h_featmap, features.shape[-1])
    features = features.reshape(1, 113, 113, features.shape[-1])

    return features

def get_vgg_features(style_im, mlp_encoder):
    vgg_transform = transforms.Compose([
        transforms.Resize([113,113]),
        transforms.ToTensor()
    ])

    style_im_vgg = vgg_transform(style_im).cuda()
    style_im_1d = style_im_vgg.reshape(3,-1).permute(1,0)
    style_feature_1d = mlp_encoder(style_im_1d)


    return style_feature_1d

def get_vgg_features_cnn(style_im, cnn_encoder, channel_num=128):
    vgg_transform = transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor()
    ])

    style_im_vgg = vgg_transform(style_im).cuda()
    if channel_num == 64:
        style_feature_2d = cnn_encoder.encode_for_inr_decoder_64(style_im_vgg.unsqueeze(0))
    elif channel_num == 128:
        style_feature_2d = cnn_encoder.encode_for_inr_decoder_128(style_im_vgg.unsqueeze(0))
    elif channel_num == 256:
        style_feature_2d = cnn_encoder.encode_for_inr_decoder_256(style_im_vgg.unsqueeze(0))
    elif channel_num == 512:
        style_feature_2d = cnn_encoder.encode_for_inr_decoder_512(style_im_vgg.unsqueeze(0))

    style_feature_2d = F.interpolate(style_feature_2d, (113,113), mode='nearest')


    style_feature_1d = style_feature_2d.reshape(style_feature_2d.shape[1], -1)
    style_feature_1d = style_feature_1d.permute(1,0)


    return style_feature_1d

def get_semantic_features(style_im):
    vit_transform = transforms.Compose([
        # transforms.Resize([378,504]),
        transforms.Resize([448,448]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dinov1_model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
    dinov1_model.eval()
    dinov1_model.cuda()
    url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    dinov1_model.load_state_dict(state_dict, strict=True)


    vit_style_img = vit_transform(style_im).unsqueeze(0).cuda()

    dino_out = get_dino_feature_map(dinov1_model, vit_style_img.squeeze())
    fliped_dino_out = get_dino_feature_map(dinov1_model, flip_image(vit_style_img).squeeze())
    style_feature_map = dino_out
    fliped_style_feature_map = fliped_dino_out
    h, w = dino_out.shape[1], dino_out.shape[2]
    fliped_style_feature_map_2d = fliped_style_feature_map.reshape(1, h, w, style_feature_map.shape[-1]).permute(0,3,1,2)
    style_feature_map += flip_image(fliped_style_feature_map_2d).permute(0,2,3,1).reshape(style_feature_map.shape)
    style_feature_map /= 2

    radius=30
    eps=1e-3
    GF = FastGuidedFilter2d(radius, eps, 2)

    tch_img = F.interpolate(vit_style_img, size=(113, 113), mode='nearest')
    tch_mask = style_feature_map.permute(0,3,1,2)
    out = GF(tch_mask, tch_img).permute(0,2,3,1)

    style_semantic_feature_1d = out.reshape(-1, style_feature_map.shape[-1])

    return style_semantic_feature_1d


def style_im_clustering(vgg_features, semantic_features, num_clusters, blending_ratio):
    vgg_feat_dim = vgg_features.shape[-1]
    
    ids, center = kmeans(semantic_features, num_clusters = num_clusters, distance = 'cosine', device = 'cuda')
    local_vgg_mean_cluster = torch.zeros(num_clusters, vgg_feat_dim).cuda()
    local_vgg_std_cluster = torch.zeros(num_clusters, vgg_feat_dim).cuda()
    # local_vgg_cluster = torch.zeros(num_clusters, self.vgg_feat_dim).cuda()

    for i in range(num_clusters):
        local_vgg_mean_cluster[i] = torch.mean(vgg_features[ids == i], 0)
        local_vgg_std_cluster[i] = torch.std(vgg_features[ids == i], 0)
        # local_vgg_cluster[i] = torch.mean(style_vgg_feature_1d_resize[ids == i], 0)
        # local_vgg_cluster[i] = torch.mean(local_vgg_mean_1d[ids == i], 0)

    global_vgg_mean = torch.mean(vgg_features, 0).unsqueeze(0)
    global_vgg_std = torch.std(vgg_features, 0).unsqueeze(0)

    local_vgg_mean_cluster = local_vgg_mean_cluster * (1-blending_ratio) + global_vgg_mean * blending_ratio
    local_vgg_std_cluster = local_vgg_std_cluster * (1-blending_ratio) + global_vgg_std * blending_ratio
    
    local_semantic_cluster = center

    return local_vgg_mean_cluster, local_vgg_std_cluster, local_semantic_cluster

def style_im_clustering_edit(vgg_features, vgg_features_origin, semantic_features, num_clusters, blending_ratio):
    vgg_feat_dim = vgg_features.shape[-1]
    
    ids, center = kmeans(semantic_features, num_clusters = num_clusters, distance = 'cosine', device = 'cuda')
    local_vgg_mean_cluster = torch.zeros(num_clusters, vgg_feat_dim).cuda()
    local_vgg_std_cluster = torch.zeros(num_clusters, vgg_feat_dim).cuda()
    
    local_vgg_mean_cluster_origin = torch.zeros(num_clusters, vgg_feat_dim).cuda()
    local_vgg_std_cluster_origin = torch.zeros(num_clusters, vgg_feat_dim).cuda()

    for i in range(num_clusters):
        local_vgg_mean_cluster[i] = torch.mean(vgg_features[ids == i], 0)
        local_vgg_std_cluster[i] = torch.std(vgg_features[ids == i], 0)

        local_vgg_mean_cluster_origin[i] = torch.mean(vgg_features_origin[ids == i], 0)
        local_vgg_std_cluster_origin[i] = torch.std(vgg_features_origin[ids == i], 0)


    global_vgg_mean = torch.mean(vgg_features, 0).unsqueeze(0)
    global_vgg_std = torch.std(vgg_features, 0).unsqueeze(0)

    local_vgg_mean_cluster = local_vgg_mean_cluster * (1-blending_ratio) + global_vgg_mean * blending_ratio
    local_vgg_std_cluster = local_vgg_std_cluster * (1-blending_ratio) + global_vgg_std * blending_ratio

    local_vgg_mean_cluster_origin = local_vgg_mean_cluster_origin * (1-blending_ratio) + global_vgg_mean * blending_ratio
    local_vgg_std_cluster_origin = local_vgg_std_cluster_origin * (1-blending_ratio) + global_vgg_std * blending_ratio
    
    local_semantic_cluster = center

    return local_vgg_mean_cluster, local_vgg_std_cluster, local_semantic_cluster, local_vgg_mean_cluster_origin, local_vgg_std_cluster_origin


def adain(content_feature, style_feature):
    mean_content = torch.mean(content_feature,0, keepdim = True)
    std_content = torch.std(content_feature,0, keepdim = True)
    mean_style = torch.mean(style_feature,0, keepdim = True)
    std_style = torch.std(style_feature,0, keepdim = True)
    
    adain_feature = ((content_feature -  mean_content) / (std_content + 0.0001)) * std_style + mean_style
    return adain_feature


def euler_from_quaternion(vec):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w  = vec[:,0],vec[:,1],vec[:,2],vec[:,3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.minimum(torch.ones_like(t2), t2)
        t2 = torch.maximum(-torch.ones_like(t2), t2)

        # t2 = +1.0 if t2 > +1.0 else t2
        # t2 = -1.0 if t2 < -1.0 else t2
        
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        out_vec = torch.vstack((roll_x, pitch_y, yaw_z)).T

        return out_vec # in radians
        # return roll_x, pitch_y, yaw_z # in radians


def sh_get_color(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]

    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]


    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
                    

    result += 0.5

    return result

def rot_90_rh(x, dim):
    if dim==0:
        rot_mat = torch.Tensor([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])
    elif dim==1:
        rot_mat = torch.Tensor([[0,0,1],
                                [0,1,0],
                                [-1,0,0]])
    elif dim==2:
        rot_mat = torch.Tensor([[0,-1,0],
                                [1,0,0],
                                [0,0,1]])
    
    # if len(x.shape) == 2:
    #     rot_mat = rot_mat.unsqueeze(0)

    rot_output = torch.matmul(rot_mat.cuda(), x.T).T
    return rot_output


