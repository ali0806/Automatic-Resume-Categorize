import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(train_acc, val_acc):
    percentages_train = [100 * value for value in train_acc]
    percentages_val = [100 * value for value in val_acc]

    x = np.arange(len(train_acc))

    plt.figure(figsize=(20, 6))
    plt.plot(x, percentages_train, marker='o', label='Train')
    plt.plot(x, percentages_val, marker='s', label='Validation')
    plt.xticks(x, range(1, len(train_acc) + 1))
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(loss_list):
    x = np.arange(len(loss_list))

    plt.figure(figsize=(20, 6))
    plt.plot(x, loss_list, marker='o', label='Loss')
    plt.xticks(x, range(1, len(loss_list) + 1))
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
