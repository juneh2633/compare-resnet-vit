from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def data_load(train_dir, test_dir, batch_size=32):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.6482, 0.5635, 0.4257], [0.2171, 0.2315, 0.2546]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.6482, 0.5635, 0.4257], [0.2171, 0.2315, 0.2546]
                ),
            ]
        ),
    }

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
