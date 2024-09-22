from compare import compare_models
from train.Resnet50 import resnet_train
from train.ViT import vit_train


def main():

    while 1:
        print("1. train ")
        print("2. compare ")
        a = int(input("input 1 or 2 \n"))
        if a == 1:
            print("1. vit ")
            print("2. resnet50 ")
            print("3. all ")
            x = int(input("input 1 or 2 \n"))
            if x == 1:
                vit_train()
            elif x == 2:
                resnet_train()
            elif x == 3:
                vit_train()
                resnet_train()

            break
        elif a == 2:
            compare_models()
            break


if __name__ == "__main__":
    main()
