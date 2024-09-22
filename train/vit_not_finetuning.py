import os

import torch
import torch.nn as nn
from torchvision import models
from utils.data_load import data_load
from utils.test_model import test_model


def vit_test():
    base_dir = os.path.dirname(__file__)
    train_dir = os.path.join(base_dir, "../data/train")
    test_dir = os.path.join(base_dir, "../data/test")

    num_classes = 10
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = data_load(train_dir, test_dir, batch_size)

    model = models.vit_b_16(pretrained=True)

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    model_path = "./models/vit.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("trained model")
    else:
        print("학습된 가중치가 없습니다. 사전 학습된 모델로 테스트합니다.")
    model.eval()

    test_model(model, test_loader, device)


if __name__ == "__main__":
    vit_test()
