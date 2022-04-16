from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip


class MRIDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation

        self.transforms = Transformer()
        self.config = config
        self.transforms.register(Normalize(), probability=1.0)

        if config.tf == "all_tf":
            self.transforms.register(Flip(), probability=0.5)
            self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Cutout(patch_size=np.ceil(
                np.array(config.input_size)/4)), probability=0.5)
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=0.5)

        elif config.tf == "cutout":
            self.transforms.register(Cutout(patch_size=np.ceil(
                np.array(config.input_size)/4)), probability=1)

        elif config.tf == "crop":
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=1)

        if training:
            dir_list = os.listdir(config.data_train)
            self.data = [os.path.join(config.data_train, file)
                         for file in dir_list]
            if config.label_train != None:
                self.labels = pd.read_csv(config.label_train)
            else:
                self.labels = None
        elif validation:
            dir_list = os.listdir(config.data_val)
            self.data = [os.path.join(config.data_val, file)
                         for file in dir_list]
            if config.label_val != None:
                self.labels = pd.read_csv(config.label_val)
            else:
                self.labels = None
        dummy = np.load(self.data[0])
        assert dummy.shape == tuple(config.input_size), "3D images must have shape {}".\
            format(config.input_size)

    def collate_fn(self, list_samples):
        if self.labels != None:
            list_x = torch.stack([torch.as_tensor(x, dtype=torch.float)
                                  for (x, y) in list_samples], dim=0)
            list_y = torch.stack([torch.as_tensor(y, dtype=torch.float)
                                  for (x, y) in list_samples], dim=0)

            return (list_x, list_y)
        else:
            list_x = torch.stack([torch.as_tensor(x, dtype=torch.float)
                                  for x in list_samples], dim=0)
            return list_x

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        np.random.seed()
        x1_img = np.load(self.data[idx])
        x2_img = np.load(self.data[idx])
        x1 = self.transforms(x1_img)
        x2 = self.transforms(x2_img)
        # x1 = x1_img
        # x2 = x2_img
        if self.labels != None:
            labels = self.labels[self.config.label_name].values[idx]
        else:
            labels = None
        x = np.stack((x1, x2), axis=0)

        if labels != None:
            return (x, labels)
        else:
            return x

    def __len__(self):
        return len(self.data)
