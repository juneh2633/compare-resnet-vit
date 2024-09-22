import torch


def test_model(model, test_loader, device):
    model.eval()
    running_corrects = 0
    class_names = [
        "apple",
        "avocado",
        "banana",
        "cherry",
        "kiwi",
        "mango",
        "orange",
        "pineapple",
        "strawberries",
        "watermelon",
    ]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                print(
                    f"Image {i + 1} answer : {class_names[labels[i].item()]}, Predicted: {class_names[preds[i].item()]}"
                )
                print("")
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc
