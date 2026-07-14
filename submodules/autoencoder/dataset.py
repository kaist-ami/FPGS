import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import RandomCrop

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, downsample=1, crop_down=1):
        data_names = glob.glob(os.path.join(data_dir, '*.pt'))
        self.data_dic = {}
        sample_feature = torch.load(data_names[0])
        _, channel, h, w = sample_feature.shape
        down_h = h//downsample
        down_w = w//downsample
        self.randomcrop = RandomCrop((h//(downsample*crop_down),w//(downsample*crop_down)))

        data_num = len(data_names)        
        data = torch.zeros(data_num, channel, down_h, down_w)


        for i in tqdm(range(len(data_names))):
            # if i > 32:
            #     continue
            features = torch.load(data_names[i]).float()

            features = torch.nn.functional.interpolate(features, size=(down_h,down_w),  mode='nearest')

            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            data[i] = features
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