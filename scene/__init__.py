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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import copy
from scene.cameras import Camera

from utils.my_utils import posetow2c_matrcs

from utils.pose_utils import generate_spiral_path

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.dataset_name = scene_info.ply_path.split('/')[-4]


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getSpiralCameras(self, scale=1.0):
        camera_path = os.path.join('LLFF_cameras', self.dataset_name + '_gaussian.npy')
        # camera_path = os.path.join('LLFF_cameras', self.dataset_name + '.npy')
        
        llff_poses = np.load(camera_path)
        
        # llff_poses = np.load(os.path.join('/home/geonu/gaussian-splatting-16/data/nerf_llff_data', self.dataset_name) + '/poses_bounds.npy')
        # llff_poses = llff_poses[:,:15]
        # llff_poses = llff_poses.reshape(-1,3,5)
        # llff_poses = llff_poses[:,:,:4]


        # ######## StyleRF ########
        # new_poses = np.load("/home/geonu/StyleRF/{}_path.npy".format(self.dataset_name))
        # new_poses = new_poses[:,:3,:]
        # new_poses[:,:,3] *= (0.75/0.89)
        # llff_poses = new_poses
        # # llff_poses[:,:,:3] = new_poses[:,:,:3]

        cam_sample = self.train_cameras[1][0]
        uid = cam_sample.uid
        colmap_id = cam_sample.colmap_id
        FoVx = cam_sample.FoVx
        FoVy = cam_sample.FoVy
        image_name = cam_sample.image_name
        image = cam_sample.original_image

        # llff_poses_15 = np.zeros((llff_poses.shape[0],3,5))
        # llff_poses_15[:,:,:4] = llff_poses
        # llff_poses_15 = llff_poses_15.transpose(1,2,0)
        # # llff_poses_15 = llff_poses_15.transpose(1,2,0)

        # # a = np.load('/home/geonu/gaussian-splatting-16/data/nerf_llff_data/flower/poses_bounds.npy')
        # # llff_poses_15 = a[:,:15].reshape(-1,3,5)
        # # llff_poses_15 = llff_poses_15.transpose(1,2,0)

        # w2c_matriclist = posetow2c_matrcs(llff_poses_15)

    
        ### kplanes poses ###
        llff_poses = np.concatenate([llff_poses[:,  :, 1:2,], llff_poses[:, :, 0:1], -llff_poses[:,:, 2:3],  llff_poses[:,:, 3:4]], axis=2)

        # ### styleRF poses ###
        # llff_poses[:, :, 0:1] = -llff_poses[:, :, 0:1]
        # aver_pose = np.load(os.path.join('LLFF_cameras', self.dataset_name + '_average.npy'))
        # scale_factor = np.load(os.path.join('LLFF_cameras', self.dataset_name + '_scale_factor.npy'))
        # llff_poses[..., 3] *= scale_factor



        llff_poses_homo = np.zeros((llff_poses.shape[0],4,4))
        llff_poses_homo[:,:3,:] = llff_poses
        llff_poses_homo[:,3,3] = 1

        # ### styleRF poses ###
        # llff_poses_homo = np.matmul(aver_pose, llff_poses_homo)
        # llff_poses_homo[:,:,0] *= -1
        # llff_poses_homo[:,:,1] *= -1
        # llff_poses_homo[:,:,2] *= -1



        w2c_matriclist = np.linalg.inv(llff_poses_homo)


    
        cam_out_list = []
        
        # a= generate_spiral_path(np.load(os.path.join('/home/geonu/gaussian-splatting-16/data/nerf_llff_data', self.dataset_name) + '/poses_bounds.npy'))

        # w2c_matriclist = llff_poses
        for i in range(len(w2c_matriclist)):
        # for w2c in a:
            R = llff_poses_homo[i,:3,:3]
            # R[0,:] = -R[0,:]
            T = w2c_matriclist[i,:3,3]

            cam_out_copy = Camera(colmap_id, R, T, FoVx, FoVy, image, None, image_name, uid)

            cam_out_list.append(cam_out_copy)
        
        cam_out_dict = {}
        cam_out_dict[scale] = cam_out_list


        return cam_out_dict[scale]
    
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses: np.ndarray) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg