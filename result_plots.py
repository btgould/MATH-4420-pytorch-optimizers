import pickle  
import matplotlib.pyplot as plt


def read_data():
    with open('plots/plot_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def loss_plot(data):
    fig, ax = plt.subplots(1,2)
    for (dataset, opt), values in data.items():
        loss_list, _, _, _ = values
        data_idx = 0 if dataset == 'MNIST' else 1
        ax[data_idx].plot(loss_list, label=f'{opt}')
        ax[data_idx].set_title(f'{dataset}')
        ax[data_idx].legend()
        ax[data_idx].set_xlabel('Iterations')

    fig.set_size_inches(9, 5)
    plt.tight_layout()
    plt.savefig(f'plots/loss.png')

def acc_plot(data):
    fig, ax = plt.subplots(2,4)
    opt_list = ['SGD', 'SGD with scheduler', 'RMSprop', 'Adam']
    
    for (dataset, opt), values in data.items():
        _, acc_list, test_acc_list, _ = values
        data_idx, lim = (0, (94,100)) if dataset == 'MNIST' else (1, (50, 99))
        opt_idx = opt_list.index(opt)
        ax[data_idx,opt_idx].plot(acc_list, label=f'train accuracy', marker='o', markersize=1, linewidth=1)
        ax[data_idx,opt_idx].plot(test_acc_list, label=f'test accuracy')
        ax[data_idx,opt_idx].set_title(f'{dataset}; {opt}')
        ax[data_idx,opt_idx].set_xlabel('Epochs')
        ax[data_idx,opt_idx].set_ylim(lim)
        ax[data_idx,opt_idx].legend()

    fig.set_size_inches(4*len(opt_list), 10)
    plt.tight_layout()
    plt.savefig(f'plots/acc.png')

def moment_plot(data):
    fig, ax = plt.subplots(1,2)
    for (dataset, opt), values in data.items():
        if opt != 'Adam':
            continue
        _, _, _, moments = values
        first_moment_list, second_moment_list = moments
        # print max moments
        print(f'{dataset}: {max(first_moment_list)}, {max(second_moment_list)}')
        loss_list, _, _, _ = values
        data_idx = 0 if dataset == 'MNIST' else 1
        ax[data_idx].plot(first_moment_list, label=f'first moment')
        ax[data_idx].set_title(f'{dataset}')
        ax[data_idx].set_xlabel('Iterations')
        ax[data_idx].set_ylabel('first moment')
        ax[data_idx].tick_params(axis='y', labelcolor='blue')
        # plot second moment on seperate axis
        second = ax[data_idx].twinx()
        second.plot(second_moment_list, label=f'second moment', color='orange', alpha=1)
        second.set_ylabel('second moment')
        second.tick_params(axis='y', labelcolor='orange')
        ax[data_idx].legend()
        second.legend()

    fig.set_size_inches(9, 5)
    plt.tight_layout()
    plt.savefig(f'plots/moment.png')

def make_all():
    data = read_data()
    loss_plot(data)
    acc_plot(data)
    moment_plot(data)


if __name__ == "__main__":
    make_all()