import os

import torch
from torchvision import models
from utils.data_load import data_load
from utils.test_model import test_model

base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, "./data/train")
test_dir = os.path.join(base_dir, "./data/test")


batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = data_load(train_dir, test_dir, batch_size)


def load_resnet50():
    model_resnet = models.resnet50()
    model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, 10)
    model_resnet.load_state_dict(torch.load("./models/resnet50.pth"))
    model_resnet = model_resnet.to(device)
    print("ResNet50 Test Results:")
    resnet_acc = test_model(model_resnet, test_loader, device)
    return resnet_acc


def load_vit():
    model_vit = models.vit_b_16()
    model_vit.heads.head = torch.nn.Linear(model_vit.heads.head.in_features, 10)
    model_vit.load_state_dict(torch.load("./models/vit.pth"))
    model_vit = model_vit.to(device)
    print("\nViT Test Results:")
    vit_acc = test_model(model_vit, test_loader, device)
    return vit_acc


def compare_models():
    resnet_acc = load_resnet50()
    vit_acc = load_vit()

    print("\nComparison of Models:")
    print(f"ResNet50 Accuracy: {resnet_acc:.4f}")
    print(f"ViT Accuracy: {vit_acc:.4f}")

    if resnet_acc > vit_acc:
        print("ResNet50 performed better.")
    elif vit_acc > resnet_acc:
        print("ViT performed better.")
    else:
        print("Both models performed equally well.")


if __name__ == "__main__":
    compare_models()
