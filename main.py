from compare import compare_models
from train.Resnet50 import resnet_train
from train.ViT import vit_train
from train.vit_not_finetuning import vit_test


def main():
    vit_train()
    resnet_train()


if __name__ == "__main__":
    main()
