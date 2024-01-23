from utils import *
from baseline import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import math

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == "__main__":

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    ab_path = "ab/ab1.npy"
    l_path = "l/gray_scale.npy"

    ab_df = np.load(ab_path)[0:10000]
    L_df = np.load(l_path)[0:10000]

    epoch = 10
    learning_rate = 0.0001

    # 1) dataloader 인스턴스화 (불러오기)
    # Prepare the Datasets
    train_dataset = ImageColorizationDataset(dataset = (L_df[:6000], ab_df[:6000]))
    val_dataset = ImageColorizationDataset(dataset = (L_df[6000:8000], ab_df[6000:8000]))
    test_dataset = ImageColorizationDataset(dataset = (L_df[8000:], ab_df[8000:]))

    # Build DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle = True, pin_memory = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle = False, pin_memory = True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle = False, pin_memory = True)

    # 2) 모델 인스턴스화
    model = UNet().to(device)

    # 3) Optimizer, loss function 인스턴스화
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    loss_fn = L1Loss()

    # 4) 데이터로더로 데이터 불러와서 모델에 입력
    for e in tqdm(range(epoch), desc="Epoch"):
        model.train()  # Set the model to training mode
        train_loss = 0

        for d in tqdm(train_loader, desc="Train Loader"):
            noise_img, gt = d
            noise_img, gt = noise_img.to(device), gt.to(device)

            out = model(noise_img)
            
            loss = loss_fn(out, gt)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print("Train loss: ", avg_train_loss)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        total_val_psnr = 0
        total_val_ssim = 0
        num_val_samples = 0

        with torch.no_grad():  # No gradient computation in validation phase
            for d in tqdm(val_loader, desc="Validation Loader"):
                noise_img, gt = d
                noise_img, gt = noise_img.to(device), gt.to(device)

                out = model(noise_img)
                val_loss += loss_fn(out, gt).item()

                # Calculate PSNR and SSIM for validation set
                out = out.cpu()
                gt = gt.cpu()
                for i in range(out.size(0)):
                    psnr_val = calculate_psnr(out[i], gt[i])
                    # ssim_val = ssim(out[i].numpy(), gt[i].numpy(), multichannel=True)
                    
                    total_val_psnr += psnr_val
                    # total_val_ssim += ssim_val
                    num_val_samples += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / num_val_samples
        # avg_val_ssim = total_val_ssim / num_val_samples
        print("Validation loss: ", avg_val_loss)
        print("Average Validation PSNR: ", avg_val_psnr)
        # print("Average Validation SSIM: ", avg_val_ssim)

        model.train()  # Set the model back to training mode

    # Testing phase
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():  # No gradient computation in testing phase
        for d in tqdm(test_loader, desc="Testing Loader"):
            noise_img, gt = d
            noise_img, gt = noise_img.to(device), gt.to(device)
            
            out = model(noise_img)
            test_loss += loss_fn(out, gt).item()

            # Calculate PSNR and SSIM
            out = out.cpu()
            gt = gt.cpu()
            for i in range(out.size(0)):
                psnr_val = calculate_psnr(out[i], gt[i])
                # ssim_val = ssim(out[i].numpy(), gt[i].numpy(), multichannel=True)
                
                total_psnr += psnr_val
                # total_ssim += ssim_val
                num_samples += 1

    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    # avg_ssim = total_ssim / num_samples

    
    print("Test loss: ", avg_test_loss)
    print("Average PSNR: ", avg_psnr)
    # print("Average SSIM: ", avg_ssim
    # )
            
