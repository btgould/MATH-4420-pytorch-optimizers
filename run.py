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
def train(model, dataset, optimizer, Loss, scheduler, epochs, testset=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    loss_list = []
    acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        epoch_start = time()
        running_loss = 0.0
        correct, total = 0, 0
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

            # metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            loss_list.append(loss.item())

        print(
            f"Epoch: {epoch + 1}, average loss: {running_loss / len(dataset):.4f}, acc: {correct*100/total:.2f}, time: {time() - epoch_start:.2f} seconds"
        )
        # track train and test accuracy
        acc_list.append(correct * 100 / total)
        if testset is not False:
            test_acc_list.append(test(model, testset, Loss)[0])
            model.train()
        
        # scheduler update - exp decay
        if scheduler is not None:
            optimizer.defaults["lr"] *= .9
            print(optimizer.defaults["lr"])

  

    return loss_list, acc_list, test_acc_list


def test(model, dataset, Loss, imgset="test"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = Loss(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on {imgset} images: {(100 * correct / total):.2f}, test loss: {running_loss / len(dataset):.4f}")
    return 100 * correct / total, running_loss / len(dataset)


def get_args():
     # argparse
    parser = argparse.ArgumentParser(description="Compare optimizers")
    parser.add_argument(  # TODO: I think we should change this behavior to: If no arg is given, use both custom and pytorch. With this arg, use only pytorch.
        "--preimplemented",
        action="store_true",
        help="if true, use only pytorch implemented optimizers",
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR"],
        default="MNIST CIFAR",
        action="store",
        help="Controls which dataset is used. Defaults to using both.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["SGD", "SGD+scheduler","RMSprop", "Adam"],
        default="SGD SGD+scheduler RMSprop Adam",
        action="store",
        help="""Controls which optimizers are tested. Defaults to all 3. 
                By default, only the custom implementation will be used. 
                Add --preimplemented to use the pytorch versions instead""",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for. Defaults to 5.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size to use. Defaults to 256.",
    )

    args = parser.parse_args()
    return args

def get_dataset_list(args):
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
            batch_size=args.batch_size,
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )

        dataset_list.append((dataset_name, train_loader, test_loader))
    return dataset_list



if __name__ == "__main__":
    # get args
    args=get_args()

    # hyperparameters
    EPOCHS = args.epochs
    LR = 1e-3
    MOMENTUM = 0.9
    BATCH_SIZE = args.batch_size
    WEIGHT_DECAY = 1e-4

    dataset_list=get_dataset_list(args)

    

    # initialize optimizers
    optimizer_list = []

    if args.preimplemented:
        SGD_list = [("SGD", SGD)]
        RMSprop_list = [("RMSprop", RMSprop)]
        Adam_list = [("Adam", Adam)]
    else:
        SGD_list = [("SGD", _SGD_)]
        RMSprop_list = [("RMSprop", _RMSprop_)]
        Adam_list = [("Adam", _Adam_)]

    optimizer_idx = {"SGD": SGD_list, "RMSprop": RMSprop_list, "Adam": Adam_list}
    for optimizer_name in args.optimizer.split():
        optimizer_list.extend(optimizer_idx[optimizer_name])

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
                optimizer = Opt(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            else:
                optimizer = Opt(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            print(
                colored(
                    f"training on dataset {dataset_name} with optimizer {name}...",
                    "green",
                )
            )
            train(model, trainset, optimizer, loss, None, EPOCHS)
            test(model, testset)
