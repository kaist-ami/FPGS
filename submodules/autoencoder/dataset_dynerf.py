import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import RandomCrop

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, downsample=1, crop_down=1):
        img_dirs = glob.glob(os.path.join(data_dir, 'cam*'))
        self.data_dic = {}
        sample_paths = glob.glob(os.path.join(img_dirs[0], '*.pt'))
        sample_path = sample_paths[0]
        sample_feature = torch.load(sample_path)
        _, channel, h, w = sample_feature.shape
        down_h = h//downsample
        down_w = w//downsample
        self.randomcrop = RandomCrop((h//(downsample*crop_down),w//(downsample*crop_down)))

        data_num = len(img_dirs) * len(sample_paths)        
        data = torch.zeros(data_num, channel, down_h, down_w)

        count = 0
        for img_dir in tqdm(img_dirs):
            imgs_paths = glob.glob(os.path.join(img_dir, '*.pt'))
            for imgs_path in imgs_paths:
                # if i > 32:
                #     continue
                features = torch.load(imgs_path).float()

                if down_h != h:
                    features = torch.nn.functional.interpolate(features, size=(down_h,down_w),  mode='nearest')

                name = imgs_path.split('/')[-1].split('.')[0]
                self.data_dic[name] = features.shape[0] 
                data[count] = features
                count += 1
            # if i == 0:
            #     data = features
            # else:
            #     data = np.concatenate([data, features], axis=0)
        self.data = data

    def __getitem__(self, index):
        # data = torch.tensor(self.data[index])
        data = self.data[index]
        data_cropped = self.randomcrop(data)
        return data_cropped

    def __len__(self):
        return self.data.shape[0] 