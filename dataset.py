import glob
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, utils, models

class Balls_CF_Detection(Dataset):
    def __init__(self, dir, transform=None, cache: bool = False):
        """
        Initializes the dataset by reading files in the "dir" folder.
        """
    
        if cache:    
            self.files = [io.imread(f) for f in sorted(glob.glob(f"{dir}*.jpg"))]
            self.labels = [np.load(f) for f in sorted(glob.glob(f"{dir}*.npy"))]
            self.filename = sorted(glob.glob(f"{dir}*"))
        else: 
            self.files = sorted(glob.glob(f"{dir}*.jpg"))
            self.labels = sorted(glob.glob(f"{dir}*.npy"))
            self.filename = sorted(glob.glob(f"{dir}*"))
        
        
        self.files = self.files
        self.labels = self.labels
        self.filename = self.filename
        
        self.dir = dir
        self.cache: bool = cache
        self.transform = transform

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index: int) -> Tuple[ np.ndarray, np.ndarray]:
                 
        # torch.Tensor
        """
        Returns the elements at index "index".
        The result is a tuple (image, binary vector, angles coordinates matrix).
        """
        #self.transform = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #])

        # Pretrained resnet18 model till second last layer as feature extraction.
        #resnet18 = models.resnet18(pretrained=True)
        #modules=list(resnet18.children())[:-1]
        #resnet18=nn.Sequential(*modules)
        
        #for p in resnet18.parameters():
        #    p.requires_grad = False
                
        img = io.imread(self.files[index])
        img = np.asarray(img)
        img = img.astype(np.float32)

        # Dims in: x, y, color
        # should be: color, x, y
        img = np.transpose(img, (2,0,1))
        img = torch.tensor(img)

        img = img/255

        #img = self.transform(img)

        DOWN_SAMPLE = 2
        img = img[:,::DOWN_SAMPLE,::DOWN_SAMPLE]

        # Load presence and bounding boxes and split it up
        if self.cache:
            p_bb = self.labels[index]
        else:   
            p_bb = np.load(self.labels[index])     
        
        p  = p_bb[:,0]
        bb = p_bb[:,1:5]
        
        return img, p

    # Return the dataset size
    def __len__(self) -> int:
        return len(self.files)
