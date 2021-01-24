import numpy as np
from math import ceil
import argparse
import string
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import cv2

DIR_DATASET = 'dataset'
DIR_DATASET_BASE = 'dataset_base'
DIR_MODELS = 'models'
DIR_MODELS_BASE = 'models_base'


def parse_args(extra=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '-w', '--size',
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
        help='where the generated captchas will be saved.')
    parser.add_argument(
        '--dataset_dir',
        default=DIR_DATASET,
        type=str,
        help='where the captchas will be saved.')
    parser.add_argument(
        '--models_dir',
        default=DIR_MODELS,
        type=str,
        help='where the models will be saved.')
    parser.add_argument(
        '--base_models_dir',
        default=DIR_MODELS_BASE,
        type=str,
        help='where the base models will be saved.')

    if extra is not None:
        for arg in extra:
            parser.add_argument(*arg[0], **arg[1] if len(arg) > 1 else {})

    FLAGS, unparsed = parser.parse_known_args()

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
    FLAGS.classes_name = f'{FLAGS.size}_{name}'
    FLAGS.base_dataset_path = f'{FLAGS.base_dataset_dir}/{FLAGS.classes_name}'
    FLAGS.dataset_path = f'{FLAGS.dataset_dir}/{FLAGS.classes_name}'
    FLAGS.base_model_path = f'{FLAGS.base_models_dir}/{FLAGS.classes_name}.h5'
    FLAGS.model_path = f'{FLAGS.models_dir}/{FLAGS.classes_name}.h5'
    return FLAGS, extra


def parse_label(filename, classes):
    real_num = filename[-8:-4]
    y_list = []
    for i in real_num:
        y_list.append(classes.index(i))
    return y_list


def data_generator(images, batch_size, img_shape, letter_count, classes, no_first=False):
    if not no_first:
        # this first batch only returns shape
        yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * letter_count

    while True:
        # epoch begins
        np.random.shuffle(images)
        x, y = [], []
        for img in images:
            x.append(cv2.imread(img))
            y.append(parse_label(img, classes))
            if len(x) >= batch_size:
                try:
                    x = preprocess_input(np.array(x).astype(float))
                except Exception:
                    import pdb
                    pdb.set_trace()
                y = np.array(y)
                yield x, [y[:, i] for i in range(letter_count)]
                x, y = [], []
        # epoch ends
        if len(x) > 0:  # the last batch doesn't have enough
            y = np.array(y)
            yield x, [y[:, i] for i in range(letter_count)]


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
        ax.imshow(f[0])
        ax.set_xlabel('{}({})'.format(get_label_string(f[1], classes), get_label_string(f[2], classes)))
        if not f[3]:
            ax.xaxis.label.set_color('red')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(hspace=12)
    plt.tight_layout()
    plt.show()


def create_model(with_weights, img_shape, FLAGS):
    input_image = Input(shape=img_shape)

    base_model = Xception(
        input_tensor=input_image,
        weights='imagenet' if with_weights else None,
        include_top=False,
        pooling='avg'
    )

    predicts = [Dense(len(FLAGS.classes), activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(FLAGS.size)]

    model = Model(inputs=input_image, outputs=predicts)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=f'{FLAGS.base_models_dir}/{FLAGS.classes_name}.png')
    return model
