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
