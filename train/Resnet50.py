import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights

from utils.data_load import data_load
from utils.test_model import test_model


def resnet_train():
    base_dir = os.path.dirname(__file__)
    train_dir = os.path.join(base_dir, "../data/train")
    test_dir = os.path.join(base_dir, "../data/test")

    num_classes = 10
    batch_size = 32
    num_epochs = 15
    learning_rate = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = data_load(train_dir, test_dir, batch_size)

    # model = models.resnet50(pretrained=True)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(device, model, train_loader, criterion, optimizer, num_epochs)

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/resnet50.pth")

    test_model(model, test_loader, device)


def train_model(device, model, train_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    print("Starting training...", flush=True)
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            print(inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(
            f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model
