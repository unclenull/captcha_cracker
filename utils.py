import imp
import glob
import os
import numpy as np
from math import ceil
import argparse
import string
import matplotlib.pyplot as plt
import pandas as pd
import cv2

DIR_DATASET = 'dataset'
DIR_MODELS = 'model'

FOLDER_TMP_AUG = 'tmp/showcase_aug'


def clear_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(f'{folder}/{f}')
    if not os.path.exists(folder):
        os.makedirs(folder)


def parse_args(extra=None, args=None):
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
        '-c', '--classes',
        type=str,
        help='specify explicit chars')
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
        '--dataset_dir',
        default=DIR_DATASET,
        type=str,
        help='where the real captchas are stored.')
    parser.add_argument(
        '--model_dir',
        default=DIR_MODELS,
        type=str,
        help='where the trained model is to be saved.')
    parser.add_argument(
        '--gen', '--generator',
        type=str,
        help='custom generator path')
    parser.add_argument(
        '--g-c', '--gen-conf',
        type=str,
        help='config file path for the builtin generator')

    if extra is not None:
        for arg in extra:
            parser.add_argument(*arg[0], **arg[1] if len(arg) > 1 else {})

    FLAGS, unparsed = parser.parse_known_args(args)

    len_extra = len(unparsed)
    if len_extra > 0:
        extra = {}
        i = 0
        while i < len_extra:
            extra[unparsed[i][2:]] = unparsed[i + 1]
            i += 2
    else:
        extra = None

    # import pdb; pdb.set_trace()
    classes = FLAGS.classes or ''
    name = ''
    if not classes:
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
    FLAGS.classes_name = f'{FLAGS.length}_{classes}'
    FLAGS.dataset_path = f'{FLAGS.dataset_dir}/{FLAGS.classes_name}'
    FLAGS.model_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}.h5'
    return FLAGS, extra


def parse_filename_label(filename, length, classes):
    """
    xxxxx_label.png
    """
    return parse_label(filename[-4 - length:-4], classes)


def parse_label(text, classes):
    y_list = []
    for i in text:
        y_list.append(classes.index(i))
    return y_list


def normalize(images):
    """
    (-1, 1) * 0.98
    """
    images = images * 0.98 / 127.5
    images -= 1
    if images.shape[-1] != 1:
        images = np.expand_dims(images, -1)
    return images


def data_generator_from_fs(images, batch_size, img_shape, length, classes, no_first=False):
    total = len(images)
    if total < batch_size:
        batch_size = total
    if not no_first:
        # this first batch only returns shape
        yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * length

    while True:
        # epoch begins
        np.random.shuffle(images)
        x, y = [], []
        for img in images:
            x.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
            y.append(parse_filename_label(img, length, classes))
            if len(x) >= batch_size:
                x = normalize(x)
                y = np.array(y)
                if length > 1:
                    y = [y[:, i] for i in range(length)]
                else:  # array will trigger error
                    y = y[:, 0]
                yield x, y
                x, y = [], []
        # epoch ends
        if len(x) > 0:  # the last batch doesn't have enough
            y = np.array(y)
            if length > 1:
                y = [y[:, i] for i in range(length)]
            else:  # array will trigger error
                y = y[:, 0]
            yield np.array(x), y


def data_generator_from_syn(gen, batch_size, img_shape, length, classes, no_first=False):
    if not no_first:
        # this first batch only returns shape
        yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * length

    while True:
        # epoch begins
        x, y = [], []
        for _ in range(batch_size):
            img, text = gen()
            x.append(img)
            y.append(parse_label(text, classes))

        import pdb; pdb.set_trace()
        x = normalize(x)
        y = np.array(y)
        if length > 1:
            y = [y[:, i] for i in range(length)]
        else:  # array will trigger error
            y = y[:, 0]
        yield x, y


def create_generator(FLAGS, both=False, no_first=False):
    if callable(FLAGS.gen):
        return FLAGS.gen()
    elif os.path.isfile(FLAGS.gen):
        image_syn = imp.load_source('custom.generator', FLAGS.gen)
        img_shape = (FLAGS.height, FLAGS.width, 1)
        image_syn = image_syn(img_shape, FLAGS.length, FLAGS.classes).generate
    elif os.path.isfile(FLAGS.get_conf):
        image_syn, img_shape = get_builtin_image_gen()

    if image_syn is not None:
        gen = data_generator_from_syn(image_syn, FLAGS.batch_size, img_shape, FLAGS.length, FLAGS.classes, FLAGS.no_first)
        gen.img_shape = img_shape
        gen.count = FLAGS.samples
        if both:
            gen2 = data_generator_from_syn(image_syn, FLAGS.batch_size, (
                FLAGS.height, FLAGS.width, 1), FLAGS.length, FLAGS.classes, FLAGS.no_first)
            gen2.img_shape = img_shape
            gen2.count = FLAGS.samples
            gen = (gen, gen2)
    elif os.path.isdir(FLAGS.dataset_dir):
        samples = glob.glob(f'{FLAGS.dataset_path}/test/*')
        img_shape = cv2.imread(samples[0], cv2.IMREAD_GRAYSCALE).shape
        img_shape = list(img_shape)
        img_shape.append(1)
        gen = data_generator_from_fs(samples, FLAGS.batch_size, img_shape, FLAGS.length, FLAGS.classes, FLAGS.no_first)
        gen.img_shape = img_shape
        gen.count = len(samples)

        if both:
            samples = glob.glob(f'{FLAGS.dataset_path}/train/*')
            gen2 = data_generator_from_fs(samples, FLAGS.batch_size, img_shape, FLAGS.length, FLAGS.classes, FLAGS.no_first)
            gen2.count = len(samples)
            gen2.img_shape = img_shape
            gen = (gen2, gen)
    else:
        return None

    return gen


def get_builtin_image_gen():
    pass


def show_metrics(history, length, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    epochs = len(history.history['loss'])
    history = pd.DataFrame(history.history)

    loss_fields = ['loss', 'val_loss']
    if length > 1:
        accuracy_fields = []
        for i in range(length):
            accuracy_fields.append(f'c{i}_acc')
            accuracy_fields.append(f'val_c{i}_acc')
            loss_fields.append(f'c{i}_loss')
            loss_fields.append(f'val_c{i}_loss')
    else:
        accuracy_fields = ['acc', 'val_acc']

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


def get_label_string(tensors, classes):
    return ''.join(map(lambda ix: classes[ix], tensors)) if isinstance(tensors, tuple) else classes[tensors]


def show_test(images_test, labels_true, labels_pred, classes):
    plots = []
    for i, label in enumerate(labels_true):
        cap = '{}({})'.format(get_label_string(labels_pred[i], classes), get_label_string(labels_true[i], classes))
        plots.append((images_test[i], cap, not np.array_equal(labels_pred[i], label)))

    if len(plots) == 0:
        return
    plot_images(plots)


def plot_images(plots):
    """
    plots: ((image, label, [isHighlight]), )
    """
    n_cols = 8
    n_rows = ceil(len(plots) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5 * 3))
    for i, f in enumerate(plots):
        ax = axes.flat[i]
        ax.imshow(f[0], cmap='gray')
        ax.set_xlabel(f[1])
        if len(f) > 2 and f[2]:
            ax.xaxis.label.set_color('red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(hspace=12)
    plt.tight_layout()
    plt.show()
