import wandb
from utils import *
from baseline import *
from inference import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import math
from pytorch_msssim import ssim
# import wandb
import datetime
import os
import torch

# Get the current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a directory name with the current date and time
directory_name = "models/" + current_datetime

# Create the directory if it does not exist
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Function to log data to a file
def log_to_file(message, file_name="training_log.txt"):
    with open(os.path.join(directory_name, file_name), "a") as file:
        file.write(message + "\n")

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


epoch = 1000
learning_rate = 0.0001

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="colorization-project",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learning_rate,
#     "architecture": "baseline",
#     "epochs": epoch,
#     }
# )

if __name__ == "__main__":

    log_file_name = "training_log_" + current_datetime + ".txt"
    log_to_file("Training started at " + current_datetime, log_file_name)

    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    log_to_file(f"Using {device} device", log_file_name)
    
    ab_path = "ab/ab1.npy"
    l_path = "l/gray_scale.npy"

    ab_df = np.load(ab_path)[0:10000]
    L_df = np.load(l_path)[0:10000]
    
    # 1) dataloader 인스턴스화 (불러오기)
    # Prepare the Datasets
    train_dataset = ImageColorizationDataset(dataset = (L_df[:6000], ab_df[:6000]))
    val_dataset = ImageColorizationDataset(dataset = (L_df[6000:8000], ab_df[6000:8000]))
    test_dataset = ImageColorizationDataset(dataset = (L_df[8000:10000], ab_df[8000:10000]))

    # Build DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle = True, pin_memory = True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle = False, pin_memory = True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle = False, pin_memory = True, num_workers=0)
    
    # 2) 모델 인스턴스화
    # model = UNet().to(device)
    model = UNet_with_ResNet().to(device)

    # 3) Optimizer, loss function 인스턴스화
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    loss_fn = L1Loss()

    patience = 200  # Number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')
    epochs_since_improvement = 0

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
        # wandb.log({"Train Loss": avg_train_loss}, step=e)
        log_to_file(f"Epoch {e}: Train Loss: {avg_train_loss}", log_file_name)


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
                    # ssim_val = ssim(out[i], gt[i], data_range=1.0, size_average=False)
                    ssim_val = ssim(out[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=1.0, size_average=False)
                    
                    total_val_psnr += psnr_val
                    total_val_ssim += ssim_val.item()
                    num_val_samples += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / num_val_samples
        avg_val_ssim = total_val_ssim / num_val_samples
        print("Validation loss: ", avg_val_loss)
        print("Average Validation PSNR: ", avg_val_psnr)
        print("Average Validation SSIM: ", avg_val_ssim)

        # wandb.log({"Validation Loss": avg_val_loss, "Validation PSNR": avg_val_psnr, "Validation SSIM": avg_val_ssim}, step=e)
        log_to_file(f"Epoch {e}: Validation Loss: {avg_val_loss}, Validation PSNR: {avg_val_psnr}, Validation SSIM: {avg_val_ssim}", log_file_name)
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Stopping training.")
            break

        # with torch.no_grad():
            # visualize(model, e, current_datetime)
        
        # Save the model in the directory
        model_save_path = os.path.join(directory_name, "model_" + str(e) + ".pt")
        if (e > 250 and e % 10 == 0) or e % 50 == 0:
            torch.save(model.state_dict(), model_save_path)
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
                ssim_val = ssim(out[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=1.0, size_average=False)
                    
                # ssim_val = ssim(out[i].numpy(), gt[i].numpy(), multichannel=True)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                num_samples += 1

    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    
    print("Test loss: ", avg_test_loss)
    print("Average PSNR: ", avg_psnr)
    print("Average SSIM: ", avg_ssim)
            
    # wandb.log({"Test Loss": avg_test_loss, "Test PSNR": avg_psnr, "Test SSIM": avg_ssim})
    log_to_file(f"Test Loss: {avg_test_loss}, Test PSNR: {avg_psnr}, Test SSIM: {avg_ssim}", log_file_name)