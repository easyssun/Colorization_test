import os
from pathlib import Path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm

import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg19
from torch.utils.data import Dataset, DataLoader

image_size_1 = 256
image_size_2 = 256

class ImageDataset(Dataset):
    ''' 
    
    Class that deals with the Data Loading and preprocessing steps such as image resizing, data augmentation (horizontal flip) 
    and conversion of RGB image to LAB color space with standardization.
    
    '''
    
    def __init__(self,paths,train = True):
        if train == True:
            self.transforms = transforms.Compose([transforms.Resize((image_size_1,image_size_2)),
                                                 transforms.RandomHorizontalFlip()]) # Basic Data Augmentation
        elif train == False:
            self.transforms = transforms.Compose([transforms.Resize((image_size_1,image_size_2))])
            
        self.train = train
        self.paths = paths
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        lab = transforms.ToTensor()(lab)
        L = lab[[0],...]/50 - 1 # Standardizing L space
        ab = lab[[1,2],...]/128 # Standardizing ab space
        
        return {'L': L,
                'ab': ab}