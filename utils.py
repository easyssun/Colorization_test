# import os
# from pathlib import Path
# import glob
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# from tqdm import tqdm

# import PIL
# from PIL import Image
# from skimage.color import rgb2lab, lab2rgb

# import torch
# from torch import nn, optim
# from torchvision import transforms
# from torchvision.models.resnet import resnet18
# from torchvision.models.vgg import vgg19
# from torch.utils.data import Dataset, DataLoader

# image_size_1 = 256
# image_size_2 = 256

# class ImageDataset(Dataset):
#     ''' 
    
#     Class that deals with the Data Loading and preprocessing steps such as image resizing, data augmentation (horizontal flip) 
#     and conversion of RGB image to LAB color space with standardization.
    
#     '''
    
#     def __init__(self,paths,train = True):
#         if train == True:
#             self.transforms = transforms.Compose([transforms.Resize((image_size_1,image_size_2)),
#                                                  transforms.RandomHorizontalFlip()]) # Basic Data Augmentation
#         elif train == False:
#             self.transforms = transforms.Compose([transforms.Resize((image_size_1,image_size_2))])
            
#         self.train = train
#         self.paths = paths
        
#     def __len__(self):
#         return len(self.paths)
    
#     def __getitem__(self,idx):
#         img = Image.open(self.paths[idx]).convert("RGB")
#         img = self.transforms(img)
#         img = np.array(img)
#         lab = rgb2lab(img).astype("float32")
#         lab = transforms.ToTensor()(lab)
#         L = lab[[0],...]/50 - 1 # Standardizing L space
#         ab = lab[[1,2],...]/128 # Standardizing ab space
        
#         return {'L': L,
#                 'ab': ab}
    
import os
from pathlib import Path

# Import glob to get the files directories recursively
import glob

# Import Garbage collector interface
import gc 

# Import OpenCV to transforme pictures
import cv2

# Import Time
import time

# import numpy for math calculations
import numpy as np

# Import pandas for data (csv) manipulation
import pandas as pd

# Import matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
matplotlib.use('TkAgg')  # You can replace 'TkAgg' with another backend if you prefer

# matplotlib.style.use('fivethirtyeight') 
# %matplotlib inline

import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import pytorch_lightning as pl

# Import pytorch to build Deel Learling Models 
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

# from torchsummary import summary

# Import tqdm to show a smart progress meter
from tqdm import tqdm

# Import warnings to hide the unnessairy warniings
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ab_path = "ab/ab/ab1.npy"
l_path = "l/gray_scale.npy"

ab_df = np.load(ab_path)[0:10000]
L_df = np.load(l_path)[0:10000]
dataset = (L_df,ab_df )
# gc.collect()

def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = L  * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = torch.cat([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

# img = np.zeros((224,224,3))
# img[:,:,0] = L_df[0]
# plt.imshow(lab_to_rgb(img,ab_df[0]))

plt.figure(figsize=(30,30))
for i in range(1,16,2):
    print(i)
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = L_df[i]
    plt.title('B&W')
    plt.imshow(lab2rgb(img))
    
    plt.subplot(4,4,i+1)
    img[:,:,1:] = ab_df[i]
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.title('Colored')
    plt.imshow(img)
plt.show()
    
class ImageColorizationDataset(Dataset):
    ''' Black and White (L) Images and corresponding A&B Colors'''
    def __init__(self, dataset, transform=None):
        '''
        :param dataset: Dataset name.
        :param data_dir: Directory with all the images.
        :param transform: Optional transform to be applied on sample
        '''
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset[0])
    
    def __getitem__(self, idx):
        L = np.array(dataset[0][idx]).reshape((224,224,1))
        L = transforms.ToTensor()(L)
        
        ab = np.array(dataset[1][idx])
        ab = transforms.ToTensor()(ab)

        return ab, L
    
