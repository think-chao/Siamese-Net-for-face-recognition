#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: FaceDatasets.py 
@time: 2019/06/28
@contact: wchao118@gmail.com
@software: PyCharm 
"""

from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import torch
import numpy as np


class FaceData(Dataset):
    def __init__(self, folder_datasets, transform=None):
        self.folder_datasets = folder_datasets
        self.transform = transform

    def __len__(self):
        return len(self.folder_datasets.imgs)

    def __getitem__(self, item):
        img0_tuple = random.choice(self.folder_datasets.imgs)
        is_same_class = random.randint(0, 1)
        if is_same_class:
            label = 0
            while True:
                img1_tuple = random.choice(self.folder_datasets.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            label = 1
            while True:
                img1_tuple = random.choice(self.folder_datasets.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = np.array(label)
        label = torch.from_numpy(label)
        return img0, img1, label
