import wandb
from utils import *
from baseline import *
from inference import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss, L1Loss, MSELoss
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
learning_rate_generator = 0.0002
learning_rate_discriminator = 0.0002

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    log_to_file(f"Using {device} device", log_file_name)
    
    ab_path = "ab/ab1.npy"
    l_path = "l/gray_scale.npy"

    ab_df = np.load(ab_path)[0:10000]
    L_df = np.load(l_path)[0:10000]
    
    # 1) dataloader 인스턴스화 (불러오기)
    # Prepare the Datasets
    # train_dataset = ImageColorizationDataset(dataset = (L_df[:6000], ab_df[:6000]))
    # val_dataset = ImageColorizationDataset(dataset = (L_df[6000:8000], ab_df[6000:8000]))
    # test_dataset = ImageColorizationDataset(dataset = (L_df[8000:], ab_df[8000:]))
    train_dataset = ImageColorizationDataset(dataset = (L_df[:6000], ab_df[:6000]))
    val_dataset = ImageColorizationDataset(dataset = (L_df[6000:8000], ab_df[6000:8000]))
    test_dataset = ImageColorizationDataset(dataset = (L_df[8000:10000], ab_df[8000:10000]))

    # Build DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle = True, pin_memory = True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle = False, pin_memory = True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle = False, pin_memory = True, num_workers=0)
    
    # 2) 모델 인스턴스화
    # model = UNet().to(device)
    model = UNet_GAN().to(device)

    # 3) Optimizer, loss function 인스턴스화
    optimizer_G = Adam(model.unet.parameters(), lr=learning_rate_generator)
    optimizer_D = Adam(model.discriminator.parameters(), lr=learning_rate_discriminator)

    
    criterion_GAN = BCELoss()
    criterion_pixelwise = L1Loss()

    for e in tqdm(range(epoch), desc="Epoch"):
        model.train()
        train_loss_G = 0
        train_loss_D = 0

        for noise_img, gt in tqdm(train_loader, desc="Train Loader"):
            # valid = torch.ones((noise_img.size(0), 1), requires_grad=False).to(device)
            # fake = torch.zeros((noise_img.size(0), 1), requires_grad=False).to(device)
            valid = torch.ones((noise_img.size(0), 1, 26, 26), requires_grad=False).to(device)
            fake = torch.zeros((noise_img.size(0), 1, 26, 26), requires_grad=False).to(device)


            noise_img, gt = noise_img.to(device), gt.to(device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            generated_images = model.unet(noise_img)

            # Adjusted part: Prepare discriminator inputs
            # Assuming noise_img is the L channel and generated_images are the ab channels
            # Concatenate L channel with generated ab channels for the discriminator's input
            discriminator_input_fake = torch.cat((noise_img, generated_images), 1)
            

            # Flatten the discriminator output before applying BCELoss
            out_discriminator_fake = model.discriminator(discriminator_input_fake)
            # out_discriminator_fake = out_discriminator_fake.view(-1)  # Flatten the output

            loss_GAN = criterion_GAN(out_discriminator_fake, valid)
            loss_pixel = criterion_pixelwise(generated_images, gt)
            
            # Total loss for the generator
            loss_G = 0.001 * loss_GAN + 0.999 * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images for discriminator
            # Here, you need to concatenate the L channel with the real ab channels from gt
            discriminator_input_real = torch.cat((noise_img, gt), 1)
            real_loss = criterion_GAN(model.discriminator(discriminator_input_real), valid)

            # Fake images for discriminator (already concatenated)
            fake_loss = criterion_GAN(model.discriminator(discriminator_input_fake.detach()), fake)
            loss_D = 0.5 * (real_loss + fake_loss)

            loss_D.backward()
            optimizer_D.step()

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()

        avg_train_loss_G = train_loss_G / len(train_loader)
        avg_train_loss_D = train_loss_D / len(train_loader)
        print(f"Epoch {e}, Loss G: {avg_train_loss_G}, Loss D: {avg_train_loss_D}")
        log_to_file(f"Epoch {e}, Loss G: {avg_train_loss_G}, Loss D: {avg_train_loss_D}", log_file_name)


        # Validation phase
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

                generated_images, _ = model(noise_img)  # Ignore discriminator output during validation
                val_loss += criterion_pixelwise(generated_images, gt).item()

                # Calculate PSNR and SSIM for validation set
                generated_images = generated_images.cpu()
                gt = gt.cpu()
                for i in range(generated_images.size(0)):
                    psnr_val = calculate_psnr(generated_images[i], gt[i])
                    ssim_val = ssim(generated_images[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=1.0, size_average=False)
                    
                    total_val_psnr += psnr_val
                    total_val_ssim += ssim_val.item()
                    num_val_samples += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = total_val_psnr / num_val_samples
        avg_val_ssim = total_val_ssim / num_val_samples
        print("Validation loss: ", avg_val_loss)
        print("Average Validation PSNR: ", avg_val_psnr)
        print("Average Validation SSIM: ", avg_val_ssim)

        log_to_file(f"Epoch {e}: Validation Loss: {avg_val_loss}, Validation PSNR: {avg_val_psnr}, Validation SSIM: {avg_val_ssim}", log_file_name)

        model_save_path = os.path.join(directory_name, "model_" + str(e) + ".pt")
        if (e > 250 and e % 10 == 0) or e % 50 == 0:
            torch.save(model.state_dict(), model_save_path)

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
            
            generated_images, _ = model(noise_img)  # Ignore discriminator output during testing
            test_loss += criterion_pixelwise(generated_images, gt).item()

            # Calculate PSNR and SSIM
            generated_images = generated_images.cpu()
            gt = gt.cpu()
            for i in range(generated_images.size(0)):
                psnr_val = calculate_psnr(generated_images[i], gt[i])
                ssim_val = ssim(generated_images[i].unsqueeze(0), gt[i].unsqueeze(0), data_range=1.0, size_average=False)
                
                total_psnr += psnr_val
                total_ssim += ssim_val.item()
                num_samples += 1

    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print("Test loss: ", avg_test_loss)
    print("Average PSNR: ", avg_psnr)
    print("Average SSIM: ", avg_ssim)

    log_to_file(f"Test Loss: {avg_test_loss}, Test PSNR: {avg_psnr}, Test SSIM: {avg_ssim}", log_file_name)
