from utils import *
from baseline import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm 
import numpy as np

if __name__ == "__main__":
    
    ab_path = "ab/ab/ab1.npy"
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
    model = UNet()

    # 3) Optimizer, loss function 인스턴스화
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    loss_fn = L1Loss()

    # 4) 데이터로더로 데이터 불러와서 모델에 입력
    for e in tqdm(range(epoch), desc="Epoch"):
        model.train()  # Set the model to training mode
        train_loss = 0

        for d in tqdm(train_loader, desc="Train Loader"):
            noise_img, gt = d
            out = model(noise_img)
            
            loss = loss_fn(out, gt)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print("Train loss: ", avg_train_loss)

        model.eval()  # Set the model to evaluation mode
        val_loss = 0

        with torch.no_grad():  # No gradient computation in validation phase
            for d in tqdm(val_loader, desc="Validation Loader"):
                noise_img, gt = d
                out = model(noise_img)
                val_loss += loss_fn(out, gt).item()  

        avg_val_loss = val_loss / len(val_loader)
        print("Validation loss: ", avg_val_loss)

        model.train()  # Set the model back to training mode

    # Testing phase
    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.no_grad():  # No gradient computation in testing phase
        for d in tqdm(test_loader, desc="Testing Loader"):
            noise_img, gt = d
            out = model(noise_img)
            test_loss += loss_fn(out, gt).item()

    avg_test_loss = test_loss / len(test_loader)
    print("Test loss: ", avg_test_loss)
            
