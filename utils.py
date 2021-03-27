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
from tensorflow.keras.models import load_model
from synthesizer import Captcha
from reflection_padding_2D import ReflectionPadding2D
from tensorflow_addons.layers import InstanceNormalization  # noqa

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
        '-b', '--batch-size',
        default=32,
        type=int,
        help='batch size for all operations')
    parser.add_argument(
        '-e', '--epochs',
        default=200,
        type=int)
    parser.add_argument(
        '--dataset-dir',
        default=DIR_DATASET,
        type=str,
        help='where the real captchas are stored.')
    parser.add_argument(
        '--model-dir',
        default=DIR_MODELS,
        type=str,
        help='where the trained model is to be saved.')
    parser.add_argument(
        '--gen', '--generator',
        type=str,
        help='custom generator path')
    parser.add_argument(
        '--syn', '--synthesizer',
        type=str,
        help='custom synthesizer path')
    parser.add_argument(
        '--syn-conf',
        default='',
        type=str,
        help='config file path for the builtin synthesizer')

    if extra is not None:
        for arg in extra:
            parser.add_argument(*arg[0], **arg[1] if len(arg) > 1 else {})

    FLAGS, unparsed = parser.parse_known_args(args)
    if len(unparsed) > 0:
        extra = {}
        for arg in unparsed:
            arg = arg.split('=')
            extra[arg[0]] = arg[1]
    else:
        extra = None

    name = classes = FLAGS.classes or ''
    if not classes:
        if os.path.isfile(FLAGS.syn_conf):
            FLAGS.syn_conf = imp.load_source('captcha.conf', FLAGS.syn_conf)
            if FLAGS.syn_conf.charset_classes is not None:
                if '0' in FLAGS.syn_conf.charset_classes:
                    classes += string.digits
                    name += 'd'
                if 'a' in FLAGS.syn_conf.charset_classes:
                    classes += string.ascii_uppercase
                    name += 'u'
                if 'A' in FLAGS.syn_conf.charset_classes:
                    classes += string.ascii_lowercase
                    name += 'l'
        else:
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
    else:
        print('Dataset is {name}')

    FLAGS.classes = classes
    FLAGS.classes_name = f'{FLAGS.length}_{name}'
    FLAGS.dataset_path = f'{FLAGS.dataset_dir}/{FLAGS.classes_name}'
    FLAGS.model_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}.h5'
    FLAGS.refiner_forward_model_path = f'{FLAGS.model_dir}/refiner_forward_{FLAGS.classes_name}.h5'
    FLAGS.refiner_backward_model_path = f'{FLAGS.model_dir}/refiner_backward_{FLAGS.classes_name}.h5'
    FLAGS.transferred_model_path = f'{FLAGS.model_dir}/transferred_{FLAGS.classes_name}.h5'
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
    images = np.array(images) * 0.98 / 127.5
    images -= 1
    if images.shape[-1] != 1:
        images = np.expand_dims(images, -1)
    return images


def normalize_batches(images):  # list of objects
    shape = list(images[0].shape)
    shape.insert(0, -1)
    images = np.vstack(images).reshape(shape)
    images = normalize(images)
    return images


def data_generator_from_fs(images, batch_size, length, classes, img_shape=None):
    total = len(images)
    if total < batch_size:
        batch_size = total
    while True:
        # epoch begins
        np.random.shuffle(images)
        x, y = [], []
        for imgpath in images:
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            if img_shape:
                img = cv2.resize(img, (img_shape[1], img_shape[0]))
            x.append(img)
            y.append(parse_filename_label(imgpath, length, classes))
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
            x = normalize(x)
            y = np.array(y)
            if length > 1:
                y = [y[:, i] for i in range(length)]
            else:  # array will trigger error
                y = y[:, 0]
            yield np.array(x), y


def data_generator_from_syn(syn, batch_size, length, classes):
    while True:
        # epoch begins
        x, y = [], []
        batches = syn(batch_size)
        for b in batches:
            img, text = b
            x.append(img)
            y.append(parse_label(text, classes))

        # import pdb; pdb.set_trace()
        # x = normalize(x)
        y = np.array(y)
        if length > 1:
            y = [y[:, i] for i in range(length)]
        else:  # array will trigger error
            y = y[:, 0]
        yield np.array(x), y


def get_recognizer_generator(FLAGS, both=False):
    image_syn = None
    if FLAGS.gen is not None:
        if callable(FLAGS.gen):
            image_gen = FLAGS.gen()
        elif os.path.isfile(FLAGS.gen):
            image_gen = imp.load_source('custom.generator', FLAGS.gen).default().get_batch
        else:
            print('FLAGS.gen is invalid')
            exit()

        def image_syn(*args):
            batches = image_gen(*args)
            images = normalize_batches(batches[:, 0])
            labels = batches[:, 1]
            batches = np.array([(images[i], labels[i]) for i in range(batches.shape[0])])
            return batches

        img_shape = (FLAGS.height, FLAGS.width, 1)
    else:
        rs = get_refined_image_gen(FLAGS, both)
        if rs is not None:
            image_syn, img_shape = rs

    if image_syn is not None:
        gen = data_generator_from_syn(image_syn, FLAGS.batch_size, FLAGS.length, FLAGS.classes)
        gen = (gen, ceil(FLAGS.samples * FLAGS.test_ratio))
        if both:
            gen2 = data_generator_from_syn(image_syn, FLAGS.batch_size, FLAGS.length, FLAGS.classes)
            gen = ((gen2, FLAGS.samples), gen)
    elif os.path.isdir(FLAGS.dataset_dir):
        samples = glob.glob(f'{FLAGS.dataset_path}/test/*')
        img_shape = cv2.imread(samples[0], cv2.IMREAD_GRAYSCALE).shape
        img_shape = list(img_shape)
        if FLAGS.height:
            img_shape[0] = FLAGS.height
        if FLAGS.width:
            img_shape[1] = FLAGS.width
        # import pdb; pdb.set_trace()
        gen = data_generator_from_fs(
            samples, FLAGS.batch_size, FLAGS.length, FLAGS.classes,
            img_shape=img_shape if FLAGS.height or FLAGS.width else None
        )
        gen = (gen, len(samples))

        if both:
            samples = glob.glob(f'{FLAGS.dataset_path}/train/*')
            gen2 = data_generator_from_fs(
                samples, FLAGS.batch_size, FLAGS.length, FLAGS.classes,
                img_shape=img_shape if FLAGS.height or FLAGS.width else None
            )
            gen = ((gen2, len(samples)), gen)
        img_shape.append(1)
    else:
        return None

    return gen, img_shape


def get_refined_image_gen(FLAGS, noRefine):
    rs = get_synthesizer(FLAGS)
    if rs is None:
        return

    syn_gen, img_shape = rs
    refiner = None
    if not noRefine:
        refiner = load_model(FLAGS.refiner_forward_model_path, custom_objects=get_refiner_custom_objects())

    def new_gen(batch_size):
        batches = syn_gen.get_batch(batch_size)
        images = normalize_batches(batches[:, 0])
        if refiner:
            images = refiner.predict_on_batch(images)
        labels = batches[:, 1]
        batches = np.array([(images[i], labels[i]) for i in range(batches.shape[0])])
        return batches

    return new_gen, img_shape


def get_refiner_custom_objects():
    return {'ReflectionPadding2D': ReflectionPadding2D}


def get_synthesizer(FLAGS):
    if FLAGS.syn is None:
        return

    if callable(FLAGS.syn):
        return FLAGS.syn()
    elif os.path.isfile(FLAGS.syn):
        synthesizer = imp.load_source('custom.synthesizer', FLAGS.syn).default()
        img_shape = (synthesizer.height, synthesizer.width, 1)
        return synthesizer, img_shape
    elif FLAGS.syn_conf is not None:
        conf = FLAGS.syn_conf
        captcha_gen = Captcha(**conf)
        return captcha_gen, (conf.height, conf.width, 1)


def show_metrics(history, length, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    keys = history.history.keys()
    epochs = len(history.history['loss'])
    history = pd.DataFrame(history.history)

    if length > 1:
        loss_fields = [k for k in keys if k.endswith('_loss')]
        loss_fields.remove('val_loss')
        accuracy_fields = [k for k in keys if k.endswith('_acc')]
    else:
        loss_fields = ['loss', 'val_loss']
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
        grid=True,
        xticks=(np.arange(0, epochs, 1) if epochs < 10 else None),
        xlabel='Epoch',
        ylabel='Loss',
    )
    plt.savefig(save_path)
    plt.waitforbuttonpress()


def get_label_string(tensors, classes):
    return ''.join(map(lambda ix: classes[ix], tensors)) if isinstance(tensors, np.ndarray) else classes[tensors]


def show_test(images_test, labels_true, labels_pred, classes):
    total = len(images_test)
    if total == 0:
        print('No data test to show.')
        return

    plots = []
    positives = 0
    for i, label in enumerate(labels_true):
        cap = '{}({})'.format(get_label_string(labels_pred[i], classes), get_label_string(labels_true[i], classes))
        correct = np.array_equal(labels_pred[i], label)
        if correct:
            positives += 1
        plots.append((images_test[i], cap, not correct))

    percent = round(positives / total, 4) * 100
    plot_images(plots, title=f'Accuracy: {percent}%({positives}/{total})')


def plot_images(plots, cols=8, title=None):
    """
    plots: ((image, label, [isHighlight]), )
    """
    n_rows = ceil(len(plots) / cols)
    fig, axes = plt.subplots(n_rows, cols)
    for i, f in enumerate(plots):
        ax = axes.flat[i]
        if isinstance(f, tuple):
            img = f[0]
            ax.set_xlabel(f[1])
            if len(f) > 2 and f[2] is True:
                ax.xaxis.label.set_color('red')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            img = f
            ax.axis(False)
        ax.imshow(img, cmap='gray')

    count = len(plots)
    if count < len(axes.flat):
        for ax in axes.flat[count:]:
            ax.axis(False)

    fig.subplots_adjust(hspace=12)
    if title is not None:
        fig.canvas.set_window_title(title)
    plt.tight_layout()
    plt.show()
    # plt.waitforbuttonpress()
