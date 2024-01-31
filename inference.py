from baseline import *
from utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss, L1Loss
from pytorch_msssim import ssim
import math

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device} device")

# # Load the dataset
ab_path = "ab/ab1.npy"
l_path = "l/gray_scale.npy"
ab_df = np.load(ab_path)
L_df = np.load(l_path)

def visualize(model, epoch, train_dataset):
        
    plt.figure(figsize=(50,30))
    for i in range(1,15,3):

        # Original Grayscale image
        plt.subplot(5,3,i)
        img = np.zeros((224,224,3))
        img[:,:,0] = L_df[i]
        plt.title('B&W')
        plt.imshow(lab2rgb(img))
        
        # Original Colored image
        plt.subplot(5,3,i+1)
        img[:,:,1:] = ab_df[i]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        plt.title('Colored')
        plt.imshow(img)

        # Predicted image
        plt.subplot(5,3,i+2)
        input = transforms.ToTensor()(L_df[i]).unsqueeze(0).to(device)
        out = model(input).cpu().detach().numpy()
        out = out[0].transpose((1,2,0))
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[i]
        precited_img[:,:,1:] = out
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)
        # input = transforms.ToTensor()(np.array(L_df[i]).reshape((224,224,1))).to(device)
        # input = input.unsqueeze(0)
        # out = model(input).cpu().detach().numpy()
        # out = out[0].transpose((1,2,0))
        # precited_img = np.zeros((224,224,3))
        # precited_img[:,:,0] = L_df[i]
        # precited_img[:,:,1:] = out
        # precited_img = precited_img.astype('uint8')
        # precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        # plt.title('Predicted')
        # plt.imshow(precited_img)

    plt.savefig('./pic/results' + str(epoch) + '.png')

