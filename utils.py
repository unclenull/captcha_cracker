import numpy as np
from math import ceil
import argparse
import string
import matplotlib.pyplot as plt
import pandas as pd
import cv2

DIR_DATASET = 'dataset'
DIR_DATASET_BASE = 'dataset_base'
DIR_MODELS = 'model'
DIR_MODELS_BASE = 'model_base'


def parse_args(extra=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='including digits')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='including lowercase chars.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='including uppercase chars.')
    parser.add_argument(
        '-L', '--length',
        default=4,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '-b', '--batch_size',
        default=32,
        type=int,
        help='batch size for all operations')
    parser.add_argument(
        '--base_dataset_dir',
        default=DIR_DATASET_BASE,
        type=str,
        help='where the fake captchas are saved.')
    parser.add_argument(
        '--dataset_dir',
        default=DIR_DATASET,
        type=str,
        help='where the real captchas are saved.')
    parser.add_argument(
        '--model_dir',
        default=DIR_MODELS,
        type=str,
        help='where the model for real is saved.')
    parser.add_argument(
        '--base_model_dir',
        default=DIR_MODELS_BASE,
        type=str,
        help='where the model for synthetic is saved.')

    if extra is not None:
        for arg in extra:
            parser.add_argument(*arg[0], **arg[1] if len(arg) > 1 else {})

    FLAGS, unparsed = parser.parse_known_args()

    # import pdb; pdb.set_trace()
    len_extra = len(unparsed)
    if len_extra > 0:
        extra = {}
        i = 0
        while i < len_extra:
            extra[unparsed[i][2:]] = unparsed[i + 1]
            i += 2
    else:
        extra = None

    classes = ''
    name = ''
    if FLAGS.digit:
        classes += string.digits
        name += 'd'
    if FLAGS.upper:
        classes += string.ascii_uppercase
        name += 'u'
    if FLAGS.lower:
        classes += string.ascii_lowercase
        name += 'l'

    if len(classes) == 0:
        print('No char space set')
        exit()

    FLAGS.classes = classes
    FLAGS.classes_name = f'{FLAGS.length}_{name}'
    FLAGS.base_dataset_path = f'{FLAGS.base_dataset_dir}/{FLAGS.classes_name}'
    FLAGS.dataset_path = f'{FLAGS.dataset_dir}/{FLAGS.classes_name}'
    FLAGS.base_model_path = f'{FLAGS.base_model_dir}/{FLAGS.classes_name}_base.h5'
    FLAGS.model_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}.h5'
    return FLAGS, extra


def parse_filename_label(filename, classes):
    return parse_label(filename[-8:-4], classes)


def parse_label(text, classes):
    y_list = []
    for i in text:
        y_list.append(classes.index(i))
    return y_list


def data_generator_from_fs(images, batch_size, img_shape, letter_count, classes, no_first=False):
    total = len(images)
    if total < batch_size:
        batch_size = total
    if not no_first:
        # this first batch only returns shape
        yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * letter_count

    while True:
        # epoch begins
        np.random.shuffle(images)
        x, y = [], []
        for img in images:
            x.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
            y.append(parse_filename_label(img, classes))
            if len(x) >= batch_size:
                x = np.array(x) / 255
                x = np.expand_dims(x, -1)
                y = np.array(y)
                if letter_count > 1:
                    y = [y[:, i] for i in range(letter_count)]
                else:  # array will trigger error
                    y = y[:, 0]
                yield x, y
                x, y = [], []
        # epoch ends
        if len(x) > 0:  # the last batch doesn't have enough
            y = np.array(y)
            if letter_count > 1:
                y = [y[:, i] for i in range(letter_count)]
            else:  # array will trigger error
                y = y[:, 0]
            yield np.array(x), y


def data_generator_from_gen(Generator, batch_size, img_shape, letter_count, classes, no_first=False):
    if not no_first:
        # this first batch only returns shape
        yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * letter_count

    gen = Generator(img_shape, letter_count, classes).generate

    while True:
        # epoch begins
        x, y = [], []
        for _ in range(batch_size):
            img, text = gen()
            x.append(img)
            y.append(parse_label(text, classes))

        x = np.array(x) / 255
        x = np.expand_dims(x, -1)
        y = np.array(y)
        if letter_count > 1:
            y = [y[:, i] for i in range(letter_count)]
        else:  # array will trigger error
            y = y[:, 0]
        yield x, y


def show_metrics(history, char_count, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    epochs = len(history.history['loss'])
    history = pd.DataFrame(history.history)

    loss_fields = ['loss', 'val_loss']
    if char_count > 1:
        accuracy_fields = []
        for i in range(char_count):
            accuracy_fields.append(f'c{i}_accuracy')
            accuracy_fields.append(f'val_c{i}_accuracy')
            loss_fields.append(f'c{i}_loss')
            loss_fields.append(f'val_c{i}_loss')
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


def get_label_string(tensor, classes):
    return ''.join(map(lambda ix: classes[ix], tensor))


def show_test(images_test, labels_true, labels_pred, classes):
    plots = []
    for i, label in enumerate(labels_true):
        plots.append((images_test[i], labels_pred[i], labels_true[i], np.array_equal(labels_pred[i], label)))

    if len(plots) == 0:
        return

    n_cols = 8
    n_rows = ceil(len(plots) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5 * 3))
    for i, f in enumerate(plots):
        ax = axes.flat[i]
        ax.imshow(f[0], cmap='gray')
        ax.set_xlabel('{}({})'.format(get_label_string(f[1], classes), get_label_string(f[2], classes)))
        if not f[3]:
            ax.xaxis.label.set_color('red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(hspace=12)
    plt.tight_layout()
    plt.show()
