#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: main.py 
@time: 2019/06/28
@contact: wchao118@gmail.com
@software: PyCharm 
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from SiameseNetwork import FaceNet
import matplotlib.pyplot as plt
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from FaceDatasets import FaceData
import torchvision
import numpy as np
from torch import optim
from PIL import Image
import os


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.float()
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Config():
    training_dir = r"E:\file\data\face"
    testing_dir = r"E:\file\data\face"
    train_batch_size = 16
    epoch = 20


cfg = Config
model = FaceNet()
img_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])


def train():
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = dset.ImageFolder(cfg.training_dir)
    siamese_dataset = FaceData(dataset, transform=transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ]))
    train_loader = DataLoader(siamese_dataset, batch_size=cfg.train_batch_size, shuffle=True)

    for epoch in range(cfg.epoch):
        for iter, sample in enumerate(train_loader):
            optimizer.zero_grad()
            img0, img1, label = sample
            output1, output2 = model(img0, img1)
            loss = criterion(output1, output2, label)
            print('Epoch {} || Iteration {} : loss {}'.format(epoch, iter, int(loss)))
            loss.backward()
            optimizer.step()
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join('./checkpoints', str(epoch) + '.pth'))

    # concatenated = torch.cat((sample[0], sample[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))


def test():
    model.load_state_dict(torch.load('./checkpoints/2.pth')['model'])
    dataset = dset.ImageFolder(cfg.training_dir)
    siamese_dataset = FaceData(dataset, transform=transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ]))
    train_loader = DataLoader(siamese_dataset, batch_size=1, shuffle=True)
    for sample in train_loader:
        img0, img1, label = sample
        output1, output2 = model(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        print(euclidean_distance, label)
        concatenated = torch.cat((sample[0], sample[1]), 0)
        imshow(torchvision.utils.make_grid(concatenated))


def api(face_im):
    face_input = torch.unsqueeze(img_transform(face_im),dim=0)
    model.load_state_dict(torch.load(r'E:\code\FaceRecognition-tensorflow\checkpoints\2.pth')['model'])
    dataset = dset.ImageFolder(cfg.training_dir)
    clss = dataset.class_to_idx.keys()
    for cls in clss:
        print(cls)
        cls_dir = os.path.join(cfg.training_dir, cls)
        for im in os.listdir(cls_dir):
            cls_im = Image.open(os.path.join(cls_dir, im))
            cls_input = torch.unsqueeze(img_transform(cls_im), dim=0)
            output1, output2 = model(face_input, cls_input)
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            print(int(euclidean_distance.detach()))


if __name__ == '__main__':
    pass
