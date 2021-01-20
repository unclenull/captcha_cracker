import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_metrics(history, dense_count, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    epochs = len(history.history['loss'])
    history = pd.DataFrame(history.history)

    loss_fields = ['loss', 'val_loss']
    if dense_count > 1:
        accuracy_fields = []
        for i in range(dense_count):
            ix = '' if i == 0 else '_' + str(i)
            accuracy_fields.append('dense{}_accuracy'.format(ix))
            accuracy_fields.append('val_dense{}_accuracy'.format(ix))
            loss_fields.append('dense{}_loss'.format(ix))
            loss_fields.append('val_dense{}_loss'.format(ix))
    else:
        accuracy_fields = ['accuracy', 'val_accuracy']

    history[accuracy_fields].plot(
        ax=axes[0],
        figsize=(8, 5),
        grid=True,
        xticks=(np.arange(0, epochs, 1) if epochs < 10 else None),
        yticks=np.arange(0, 1, 0.1),
        ylim=(0, 1),
        xlabel='Epoch',
        ylabel='Accuracy',
    )
    history[loss_fields].plot(
        ax=axes[1],
        figsize=(8, 5),
        grid=True,
        xticks=(np.arange(0, epochs, 1) if epochs < 10 else None),
        xlabel='Epoch',
        ylabel='Loss',
    )
    plt.savefig(save_path)
    plt.show()
