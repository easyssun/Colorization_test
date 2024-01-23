from unet import UNet
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm 

if __name__ == "__main__":
    
    img_dir = "C:\\Users\\82102\\OneDrive - gachon.ac.kr\\바탕 화면\\컴퓨터비전 스터디\\data"
    train_file = "train.csv"
    val_file = "val.csv"
    test_file = "test.csv"
    
    epoch = 10
    learning_rate = 0.0001

    # 1) dataloader 인스턴스화 (불러오기)
    train_data = CustomDataset(train_file, img_dir)
    val_data = CustomDataset(val_file, img_dir)
    test_data = CustomDataset(test_file, img_dir)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    # 2) 모델 인스턴스화
    model = UNet()

    # 3) Optimizer, loss function 인스턴스화
    optimzer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MSELoss()

    # 4) 데이터로더로 데이터 불러와서 모델에 입력
    for e in tqdm(range(epoch), desc="Epoch"):
        train_loss = 0
        val_loss = 0

        for d in tqdm(train_dataloader, desc="train loader"):
            noise_img, gt = d
            # print(noise_img.dtype)
            # print(gt.dtype)
            out = model(noise_img)
            
            loss = loss_fn(out, gt)
            
            train_loss += loss
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

        print("Train loss: ", train_loss)
        
        for d in tqdm(val_dataloader):
            noise_img, gt = d
            out = model(noise_img)
            val_loss += loss_fn(noise_img, gt)  
            
        print("Val loss: ", val_loss)
        
            
