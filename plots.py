import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from warnings import warn

matplotlib.use("qtagg")


def plot_conv_rate(loss, file_name="fig", show=False):
    plt.clf()
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.title("Convergence Rate")

    plt.savefig(file_name)

    if show:
        warn("Showing plot will block remaining execution")
        plt.show()


penguin_means = {
    "MNIST": [18.35, 18.43, 14.98],
    "CIFAR": [38.79, 48.83, 47.50],
}


def __multi_bar_plot__(data, groups, y_label, title, width=0.25):
    x = np.arange(len(groups))  # the label locations
    index = 0

    _, ax = plt.subplots(layout="constrained")

    for attribute, measurement in data.items():
        offset = width * index
        _ = ax.bar(x + offset, measurement, width, label=attribute)
        index += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width, groups)
    ax.legend(loc="upper left", ncols=3)

    plt.show()


def plot_test_accuracies(data, optimizer_list, file_name="fig"):
    __multi_bar_plot__(
        data, optimizer_list, "Accuracy (%)", "Optimizer Test Performance"
    )
    plt.savefig(file_name)
