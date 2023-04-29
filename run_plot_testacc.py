import argparse
import os
import torch, torchvision
import torch.nn as nn
from resnet import iresnet34
from run import train, test, get_dataset_list, get_args
from torch.optim.lr_scheduler import PolynomialLR
from optimizers import _SGD_, _RMSprop_, _Adam_, SGD, RMSprop, Adam
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np

if __name__ == "__main__":
    # get args
    args=get_args()

    # hyperparameters
    EPOCHS = args.epochs
    LR = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    dataset_list=get_dataset_list(args)

    # initialize optimizers
    optimizer_list = []

    if args.preimplemented:
        SGD_list = [("SGD", SGD, False)]
        SGD_schedule_list = [("SGD with scheduler", SGD, True)]
        RMSprop_list = [("RMSprop", RMSprop, False)]
        Adam_list = [("Adam", Adam, False)]
    else:
        SGD_list = [("SGD", _SGD_, False)]
        SGD_schedule_list = [("SGD with scheduler", _SGD_, True)]
        RMSprop_list = [("RMSprop", _RMSprop_, False)]
        Adam_list = [("Adam", _Adam_, False)]

    optimizer_idx = {"SGD": SGD_list, "SGD+scheduler": SGD_schedule_list, "RMSprop": RMSprop_list, "Adam": Adam_list}
    for optimizer_name in args.optimizer.split():
        optimizer_list.extend(optimizer_idx[optimizer_name])

    # standard cross entropy
    loss = nn.CrossEntropyLoss()

    fig, ax = plt.subplots(len(dataset_list),len(optimizer_list))
    if len(optimizer_list) == 1 and len(dataset_list) == 1:
        ax = np.array(ax)
    elif len(optimizer_list) == 1:
        ax = ax.reshape(-1,1)
    if len(dataset_list) == 1:
        ax = ax.reshape(1,-1)

    fig2, ax2 = plt.subplots(1,len(dataset_list))

    # train and test
    for data_idx, (dataset_name, trainset, testset) in enumerate(dataset_list):
        for opt_idx, (name, Opt, schedule_on) in enumerate(optimizer_list):
            # initialize model
            if dataset_name == "MNIST":
                model = iresnet34(image_channels=1)
                lim = (94,100)
            else:
                model = iresnet34(image_channels=3)
                lim = (50, 99)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if name in ["SGD", "SGD with scheduler", "RMSprop"]:
                optimizer = Opt(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            elif name in ["Adam"]:
                optimizer = Opt(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            else:
                raise
            print(
                colored(
                    f"training on dataset {dataset_name} with optimizer {name}",
                    "green",
                )
            )

            if schedule_on:
                opt_scheduler = True # use exponential decay in train loop
            else:
                opt_scheduler = None
                        
            loss_list, acc_list, test_acc_list = train(model, trainset, optimizer, loss, opt_scheduler, EPOCHS, testset=testset)

            ax[data_idx,opt_idx].plot(acc_list, label=f'train accuracy')
            ax[data_idx,opt_idx].plot(test_acc_list, label=f'test accuracy')
            ax[data_idx,opt_idx].set_title(f'{dataset_name}; {name}')
            ax[data_idx,opt_idx].set_xlabel('Epochs')
            ax[data_idx,opt_idx].set_ylim(lim)
            ax[data_idx,opt_idx].legend()


            ax2[data_idx].plot(loss_list, label=f'{name}')
            ax2[data_idx].set_title(f'{dataset_name}')
            ax2[data_idx].legend()
            ax2[data_idx].set_xlabel('Iterations')
    fig.set_size_inches(4*len(optimizer_list), 5*len(dataset_list))
    plt.figure(1)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    version = 'torch' if args.preimplemented else 'ours'
    plt.savefig(f'plots/testloss_{version}.png')


    plt.figure(2)
    fig2.set_size_inches(9, 7)
    plt.tight_layout()
    plt.savefig(f'plots/loss_{version}.png')
