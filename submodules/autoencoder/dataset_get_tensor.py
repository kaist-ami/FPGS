import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import RandomCrop

class Autoencoder_dataset(Dataset):
    def __init__(self, input_data, downsample=1, crop_down=1):
        # img_dirs = glob.glob(os.path.join(data_dir, 'cam*'))
        self.data_dic = {}
        # sample_paths = glob.glob(os.path.join(img_dirs[0], '*.pt'))
        # sample_path = sample_paths[0]
        # sample_feature = torch.load(sample_path)
        data_num, channel, h, w = input_data.shape
        down_h = h//downsample
        down_w = w//downsample
        self.randomcrop = RandomCrop((h//(downsample*crop_down),w//(downsample*crop_down)))

        # data_num = len(img_dirs)        
        # data = torch.zeros(data_num, channel, down_h, down_w)

        
        if down_h != h:
            input_data = torch.nn.functional.interpolate(input_data, size=(down_h,down_w),  mode='nearest')
            # else:
            #     data = np.concatenate([data, features], axis=0)
        self.data = input_data

    def __getitem__(self, index):
        # data = torch.tensor(self.data[index])
        data = self.data[index]
        data_cropped = self.randomcrop(data)
        return data_cropped

    def __len__(self):
        return self.data.shape[0] 