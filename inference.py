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
        # print(out.shape)
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

    # validation set에 대한 결과를 저장
    plt.figure(figsize=(50,30))
    for i in range(1,15,3):

        # Original Grayscale image
        plt.subplot(5,3,i)
        img = np.zeros((224,224,3))
        img[:,:,0] = L_df[600+i]
        plt.title('B&W')
        plt.imshow(lab2rgb(img))
        
        # Original Colored image
        plt.subplot(5,3,i+1)
        img[:,:,1:] = ab_df[600+i]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        plt.title('Colored')
        plt.imshow(img)

        # Predicted image
        plt.subplot(5,3,i+2)
        L = np.array(L_df[600+i]).reshape((224,224,1))
        ab = np.array(ab_df[600+i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        with torch.no_grad():
            out = model(L).cpu().detach()
        # print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[600+i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)  

    if current_datetime != None:
        if os.path.exists(f"results/{current_datetime}") == False:
            os.makedirs(f"results/{current_datetime}")
        plt.savefig(f"results/{current_datetime}/val_epoch_{epoch}.png")                                                           
    else:
        if os.path.exists(f"results/default") == False:
            os.makedirs(f"results/default")
        plt.savefig(f"results/default/val_epoch_{epoch}.png")                                                    


def visualize_compare(model1, model2, epoch, name=None):

    plt.figure(figsize=(50,30))
    for i in range(1,20,4):

        # Original Grayscale image
        plt.subplot(5,4,i)
        img = np.zeros((224,224,3))
        img[:,:,0] = L_df[i]
        plt.title('B&W')
        plt.imshow(lab2rgb(img))
        
        # Original Colored image
        plt.subplot(5,4,i+1)
        img[:,:,1:] = ab_df[i]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        plt.title('Colored')
        plt.imshow(img)

        # Predicted image of model1
        plt.subplot(5,4,i+2)
        L = np.array(L_df[i]).reshape((224,224,1))
        ab = np.array(ab_df[i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        with torch.no_grad():
            out = model1(L).cpu().detach()
        # print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)
        
        # Predicted image of model2
        plt.subplot(5,4,i+3)
        L = np.array(L_df[i]).reshape((224,224,1))
        ab = np.array(ab_df[i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        with torch.no_grad():
            out = model2(L).cpu().detach()
        # print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)

    if name != None:
        if os.path.exists(f"results/{name}") == False:
            os.makedirs(f"results/{name}")
        plt.savefig(f"results/{name}/train_{epoch}.png")                                                           
    else:
        if os.path.exists(f"results/default") == False:
            os.makedirs(f"results/default")
        plt.savefig(f"results/default/train_{epoch}.png")  

    # validation set에 대한 결과를 저장
    plt.figure(figsize=(50,30))
    for i in range(1,20,4):

        # Original Grayscale image
        plt.subplot(5,4,i)
        img = np.zeros((224,224,3))
        img[:,:,0] = L_df[600+i]
        plt.title('B&W')
        plt.imshow(lab2rgb(img))
        
        # Original Colored image
        plt.subplot(5,4,i+1)
        img[:,:,1:] = ab_df[600+i]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        plt.title('Colored')
        plt.imshow(img)

        # Predicted image of model1
        plt.subplot(5,4,i+2)
        L = np.array(L_df[600+i]).reshape((224,224,1))
        ab = np.array(ab_df[600+i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        out = model1(L).cpu().detach()
        # print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[600+i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img)  

        # Predicted image of model2
        plt.subplot(5,4,i+3)
        L = np.array(L_df[600+i]).reshape((224,224,1))
        ab = np.array(ab_df[600+i])
        
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)
        L = torch.unsqueeze(L, 0)
        ab = torch.unsqueeze(ab, 0)
        L, ab = L.to(device), ab.to(device)
        out = model2(L).cpu().detach()
        # print(out.shape)
        out = out[0].permute(1,2,0)
        
        precited_img = np.zeros((224,224,3))
        precited_img[:,:,0] = L_df[600+i]
        precited_img[:,:,1:] = out * 255
        precited_img = precited_img.astype('uint8')
        precited_img = cv2.cvtColor(precited_img, cv2.COLOR_LAB2RGB)
        plt.title('Predicted')
        plt.imshow(precited_img) 

    if name != None:
        if os.path.exists(f"results/{name}") == False:
            os.makedirs(f"results/{name}")
        plt.savefig(f"results/{name}/validation_{epoch}.png")                                                           
    else:
        if os.path.exists(f"results/default") == False:
            os.makedirs(f"results/default")
        plt.savefig(f"results/default/validation_{epoch}.png")    

model1 = UNet().to(device)
model2 = UNet_CBAM().to(device)
# # model pt file load
epoch = 600
current_datetime1 = "20240213_150951"
current_datetime2 = "20240213_173918"
model1.load_state_dict(torch.load(f"models/{current_datetime1}/model_{epoch}.pt"))
model2.load_state_dict(torch.load(f"models/{current_datetime2}/model_{epoch}.pt"))

# visualize(model, epoch=99)
# visualize(model1, epoch, current_datetime1)
# visualize(model2, epoch, current_datetime2)
visualize_compare(model1, model2, epoch, name="baseline_vs_cbam")
