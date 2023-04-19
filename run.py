import torch
import torch.nn as nn
from resnet import iresnet34
import argparse
import torchvision
from optimizers import _SGD_, _RMSprop_, _Adam_, SGD, RMSprop, Adam
from termcolor import colored
from tqdm import tqdm
from time import time


# train loop using autmatic mixed precision
def train(model, dataset, optimizer, Loss, schedueler, epochs):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        epoch_start = time()
        running_loss = 0.0
        for i, data in enumerate(dataset):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = Loss(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(
            f"Epoch: {epoch + 1}, average loss: {running_loss / len(dataset):.4f}, time: {time() - epoch_start:.2f} seconds"
        )
        running_loss = 0.0
        # schedueler.step()


def test(model, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on test images: {(100 * correct / total):.2f}")

    return 100 * correct / total


if __name__ == "__main__":
    # hyperparameters
    EPOCHS = 5
    LR = 1e-3
    MOMENTUM = 0.9
    BATCH_SIZE = 256

    # argparse
    parser = argparse.ArgumentParser(description="Compare optimizers")
    parser.add_argument(
        "--preimplemented",
        action="store_true",
        help="if true, use pytorch implemented optimizers",
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR"],
        default="MNIST CIFAR",
        action="store",
        help="Controls which dataset is used. Defaults to using both.",
    )
    args = parser.parse_args()

    # load datasets
    dataset_list = []
    dataset_idx = {
        "MNIST": (torchvision.datasets.MNIST, (0.1307,), (0.3081,)),
        "CIFAR": (torchvision.datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    }

    for dataset_name in args.dataset.split():
        print(colored(f"loading dataset {dataset_name}...", "green"))
        dataset, mean, std = dataset_idx[dataset_name]
        train_loader = torch.utils.data.DataLoader(
            dataset(
                "data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean, std),
                    ]
                ),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset(
                "data",
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean, std),
                    ]
                ),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )

        dataset_list.append((dataset_name, train_loader, test_loader))

    # initialize optimizers
    if args.preimplemented:
        optimizer_list = [("SGD", SGD), ("RMSprop", RMSprop), ("Adam", Adam)]
    else:
        optimizer_list = [("SGD", _SGD_), ("RMSprop", _RMSprop_), ("Adam", _Adam_)]

    # standard cross entropy
    loss = nn.CrossEntropyLoss()

    # train and test
    for dataset_name, trainset, testset in dataset_list:
        for name, Opt in optimizer_list:
            # initialize model
            if dataset_name == "MNIST":
                model = iresnet34(image_channels=1)
            else:
                model = iresnet34(image_channels=3)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if name in ["SGD", "RMSprop"]:
                optimizer = Opt(model.parameters(), lr=LR, momentum=MOMENTUM)
            else:
                optimizer = Opt(model.parameters(), lr=LR)
            print(
                colored(
                    f"training  on dataset {dataset_name} with optimizer {name}...",
                    "green",
                )
            )
            train(model, trainset, optimizer, loss, None, EPOCHS)
            test(model, testset)
