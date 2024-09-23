import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 모든 이미지를 224x224로 크기 조정
        transforms.ToTensor(),
    ]
)

base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, "./data/train")
test_dir = os.path.join(base_dir, "./data/test")
# 데이터셋을 불러오고 전체 이미지의 RGB 값을 하나의 텐서로 모음
dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

mean = 0.0
std = 0.0
nb_samples = 0

for data, _ in loader:
    batch_samples = data.size(0)  # 배치 사이즈
    data = data.view(batch_samples, data.size(1), -1)  # (B, C, H, W) -> (B, C, H*W)
    mean += data.mean(2).sum(0)  # 채널별 평균
    std += data.std(2).sum(0)  # 채널별 표준편차
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("Mean:", mean)
print("Std:", std)
