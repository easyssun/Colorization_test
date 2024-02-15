import torch
import torch.nn as nn
from attention import CBAM
import numpy as np
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def positional_encoding(shape, num_encoding_functions=16):
    """
    shape: 입력 이미지의 모양 (H, W)
    num_encoding_functions: 사용할 인코딩 함수의 수
    반환 값: Positional Encoding이 적용된 텐서, 크기는 (C', H, W)
    """
    H, W = shape
    position_y = np.tile(np.linspace(-1, 1, H), (W, 1)).T
    position_x = np.tile(np.linspace(-1, 1, W), (H, 1))
    
    encodings = []
    for i in range(num_encoding_functions):
        for fn in [torch.sin, torch.cos]:
            encodings.append(fn(torch.tensor(position_x * (2 ** i), dtype=torch.float32)))
            encodings.append(fn(torch.tensor(position_y * (2 ** i), dtype=torch.float32)))
            
    positional_encodings = torch.stack(encodings, dim=0)
    return positional_encodings

class UNet(nn.Module):

    def __init__(self, output_channels=2):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(128, 16)
        self.cbam3 = CBAM(256, 16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, output_channels, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        # x = torch.cat([x, self.cbam3(conv3)], dim=1)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        # x = torch.cat([x, self.cbam2(conv2)], dim=1)       
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        # x = torch.cat([x, self.cbam1(conv1)], dim=1)   
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.sigmoid(self.final(x))
        return x

class UNet_GAN(nn.Module):
    def __init__(self, output_channels=2):
        super(UNet_GAN, self).__init__()
        self.unet = UNet(output_channels)
        self.discriminator = Discriminator()

    def forward(self, x):
        generated_image = self.unet(x)
        
        discriminator_input = torch.cat((x, generated_image), dim=1)

        validity = self.discriminator(discriminator_input)
        return generated_image, validity
    
class UNet_CBAM(nn.Module):

    def __init__(self, output_channels=2):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(128, 16)
        self.cbam3 = CBAM(256, 16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, output_channels, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, self.cbam3(conv3)], dim=1)
        # x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, self.cbam2(conv2)], dim=1)       
        # x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, self.cbam1(conv1)], dim=1)   
        # x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
class UNet_pos_enc(UNet):
    def __init__(self, output_channels=2, num_encoding_functions=16):
        super().__init__(output_channels=output_channels)
        self.num_encoding_functions = num_encoding_functions
        
        # 첫 번째 double_conv의 in_channels 수정
        self.dconv_down1 = double_conv(1 + 2 * num_encoding_functions * 2, 64)
    
    def forward(self, x):
        # x의 크기: (B, 1, H, W)
        B, _, H, W = x.shape
        pe = positional_encoding((H, W), self.num_encoding_functions).unsqueeze(0).repeat(B, 1, 1, 1)
        pe = pe.to(x.device)
        
        # Positional Encoding과 원본 입력 채널을 합침
        x = torch.cat([x, pe], dim=1)  # 수정된 입력 채널: (B, 1 + 2 * num_encoding_functions * 2, H, W)
        
        # UNet forward의 나머지 부분은 동일
        return super().forward(x)
    
class UNet_pos_enc_intermediate(UNet):
    def __init__(self, output_channels=2, num_encoding_functions=16):
        super().__init__(output_channels=output_channels)
        self.num_encoding_functions = num_encoding_functions
        
        # 첫 번째 double_conv의 in_channels 수정
        self.dconv_down1 = double_conv(1 + 2 * num_encoding_functions * 2, 64)
        
        self.dconv_up3 = double_conv(512 + 256 + 64, 256)  # 수정된 채널 수를 반영
        self.dconv_up2 = double_conv(256 + 128 + 64, 128)  # 수정된 채널 수를 반영
        self.dconv_up1 = double_conv(128 + 64 + 64, 64)  # 수정된 채널 수를 반영

    def forward(self, x):
        # x의 크기: (B, 1, H, W)
        B, _, H, W = x.shape
        initial_pe = positional_encoding((H, W), self.num_encoding_functions).unsqueeze(0).repeat(B, 1, 1, 1)
        initial_pe = initial_pe.to(x.device)
        
        # Positional Encoding과 원본 입력 채널을 합침
        x = torch.cat([x, initial_pe], dim=1)  # 수정된 입력 채널: (B, 1 + 2 * num_encoding_functions * 2, H, W)
        
        # Downsample path
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)

        # Upsample path
        x = self.upsample(x)
        pe_up3 = positional_encoding(conv3.shape[2:], self.num_encoding_functions).unsqueeze(0).repeat(B, 1, 1, 1).to(x.device)
        conv3 = torch.cat([conv3, pe_up3], dim=1)  # Positional Encoding 추가
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        pe_up2 = positional_encoding(conv2.shape[2:], self.num_encoding_functions).unsqueeze(0).repeat(B, 1, 1, 1).to(x.device)
        conv2 = torch.cat([conv2, pe_up2], dim=1)  # Positional Encoding 추가
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        pe_up1 = positional_encoding(conv1.shape[2:], self.num_encoding_functions).unsqueeze(0).repeat(B, 1, 1, 1).to(x.device)
        conv1 = torch.cat([conv1, pe_up1], dim=1)  # Positional Encoding 추가
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        
        return out
