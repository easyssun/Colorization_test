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

def visualize(model, epoch, current_datetime=None):

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
        L = np.array(L_df[i]).reshape((224,224,1))
        ab = np.array(ab_df[i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        out = model(L).cpu().detach()
        print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)

    if current_datetime != None:
        if os.path.exists(f"results/{current_datetime}") == False:
            os.makedirs(f"results/{current_datetime}")
        plt.savefig(f"results/{current_datetime}/epoch_{epoch}.png")                                                           
    else:
        if os.path.exists(f"results/default") == False:
            os.makedirs(f"results/default")
        plt.savefig(f"results/default/epoch_{epoch}.png")                                                           

# model = UNet().to(device)
# # model pt file load
# model.load_state_dict(torch.load("model_99.pt"))

# visualize(model, epoch=99)