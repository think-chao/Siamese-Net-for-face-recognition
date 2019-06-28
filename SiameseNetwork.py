#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: SiameseNetwork.py 
@time: 2019/06/28
@contact: wchao118@gmail.com
@software: PyCharm 
"""

import torch.nn as nn
import torch


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(1 * 12800, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
        )

    def forward_once(self, x):
        conv_feature = self.cnn1(x)
        conv_feature = conv_feature.view(conv_feature.size()[0], -1)
        output = self.fc(conv_feature)
        return output

    def forward(self, input1, input2):
        feat_1 = self.forward_once(input1)
        feat_2 = self.forward_once(input2)
        return feat_1, feat_2


if __name__ == '__main__':
    input = torch.ones(1, 3, 100, 100)
    face = FaceNet()
