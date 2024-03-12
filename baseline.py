import torch
import torch.nn as nn
from attention import CBAM
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

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

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 미리 학습된 ResNet34 모델 불러오기
        self.resnet = models.resnet34(pretrained=pretrained)
        # ResNet의 마지막 fully connected 레이어를 제거하여 특징 추출기로 사용
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        # ResNet을 통해 특징 추출
        x = self.features(x)
        return x

class UNet_with_ResNet(nn.Module):

    def __init__(self, output_channels=2):
        super().__init__()
        
        self.resnet_feature_extractor = ResNetFeatureExtractor()
        
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(128, 16)
        self.cbam3 = CBAM(256, 16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # Upsample the feature map
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(256 + 512 + 512, 256)
        self.dconv_up2 = double_conv(128 + 512 + 256, 128)
        self.dconv_up1 = double_conv(64 + 512 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, output_channels, 1)
        
        
    def forward(self, x):

        # 단일 채널 입력을 3채널로 확장
        x_expanded = x.repeat(1, 3, 1, 1)  # (B, C, H, W) -> (B, 3, H, W)
        
        # ResNet로 특징 추출
        resnet_feature3 = self.resnet_feature_extractor(x_expanded)
        resnet_feature3 = self.upsample8(resnet_feature3)
        resnet_feature2 = self.upsample(resnet_feature3)
        resnet_feature1 = self.upsample(resnet_feature2)

        # Encoding
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        

        # Decoding
        # print("x shape:", x.shape)
        # print("resnet_feature3 shape:", resnet_feature3.shape)
        # print("conv3 shape:", conv3.shape)
        x = torch.cat([x, resnet_feature3, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        # print("x shape:", x.shape)
        # print("resnet_feature2 shape:", resnet_feature2.shape)
        # print("conv2 shape:", conv2.shape)
        x = torch.cat([x, resnet_feature2, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = torch.cat([x, resnet_feature1, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


# class UNet_text_guided(nn.Module):

#     def __init__(self, output_channels=2):
#         super().__init__()
        
#         self.image_to_text_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        
        
#     def forward(self, x):

#         caption = self.image_to_text_model(x)

from transformers import pipeline
from transformers import CLIPModel, CLIPTokenizer

class UNetTextGuided(nn.Module):
    def __init__(self, output_channels=2):
        super(UNetTextGuided, self).__init__()
        # 예시로 사용된 모델과 토크나이저는 실제 환경에 맞게 조정되어야 합니다.
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # UNet 구조를 정의합니다. 이 부분은 UNet의 구체적인 구현에 따라 달라집니다.
        # 예를 들어, output_channels를 사용하여 마지막 층의 출력 채널을 조정할 수 있습니다.
        
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
        
    def forward(self, images, captions):
        # 이미지를 CLIP 이미지 인코더에 통과시켜 임베딩을 얻습니다.
        image_features = self.clip_model.get_image_features(images)
        
        # 캡션을 토큰화하고 CLIP 텍스트 인코더에 통과시켜 임베딩을 얻습니다.
        text_inputs = self.clip_tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        # 이미지 임베딩과 텍스트 임베딩을 결합합니다. 이는 예시로, 실제 구현에서는 조정이 필요할 수 있습니다.
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # 결합된 임베딩을 UNet에 주입합니다. 이는 UNet의 중간 단계에 추가적인 입력으로 주입하는 방식을 가정합니다.
        # 실제로는 UNet 구조 내에서 적절한 위치를 찾아 결합된 임베딩을 사용해야 할 것입니다.
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out