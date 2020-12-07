import glob
from typing import *

import numpy as np
import torch
from skimage import io
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import Balls_CF_Detection

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']


class NotreNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NotreNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))  # 1 convolution pass + 1 max pooling
        return x


if __name__ == "__main__":

    """
    # train_dataset = Balls_CF_Detection ("../mini_balls/train", 20999,
    #     transforms.Normalize([128, 128, 128], [50, 50, 50]))
    train_dataset = Balls_CF_Detection("data/train/train/")

    img,p,b = train_dataset.__getitem__(42)

    print ("Presence:")
    print (p)

    print ("Pose:")
    print (b)
    """

    criterion = None
    optimizer = None
