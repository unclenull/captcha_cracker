"""
Train single chars with Mnist alike datasets.
This is an alternative to SVM.
"""

import argparse
from math import ceil
import string
import numpy as np
from emnist import extract_training_samples
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import show_test, clear_folder, normalize, FOLDER_TMP_AUG
from recognizer import train as base_train, parse_args as base_parse_args


MODEL_DIR = 'model/single'


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--digits',
        action='store_true',
        help='including digits')
    parser.add_argument(
        '-l', '--letters',
        action='store_true',
        help='including alpha chars (26, upper and lower mixed).')
    parser.add_argument(
        '-b', '--batch_size',
        default=32,
        type=int,
        help='batch size for all operations')
    parser.add_argument(
        'cmd',
        nargs='?',
        help='train, test, show_aug, predict'
    )
    FLAGS, unparsed = parser.parse_known_args(args)
    return (FLAGS, unparsed)


def create_augmentor():
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        fill_mode="nearest"
    )


def create_generator(images, labels, batch_size):
    aug = create_augmentor()
    gen = aug.flow(images, labels, batch_size=batch_size)
    gen.img_shape = images[0].shape
    gen.count = len(images)
    return gen


def show_aug():
    clear_folder(FOLDER_TMP_AUG)

    aug = create_augmentor()
    images = get_dataset()[0]
    aug.flow(
        images,
        batch_size=FLAGS.batch_size,
        save_to_dir=FOLDER_TMP_AUG,
    ).next()


def get_classes():
    if FLAGS.digits and FLAGS.letters:
        dataset = 'balanced'  # 47
        classes = string.digits + string.ascii_uppercase + string.ascii_lowercase.translate({ord(i): None for i in 'cijklmopsuvwxyz'})
    elif FLAGS.digits:
        dataset = 'digits'
        classes = string.digits
    elif FLAGS.letters:
        dataset = 'letters'
        classes = string.ascii_uppercase
    else:
        print('No data set specified')
        exit()

    return dataset, classes


def get_dataset():
    dataset, classes = get_classes()
    images, labels = extract_training_samples(dataset)
    images = normalize(images)
    return images, labels, dataset, classes


def train(*args, overrides=None):
    if len(args) > 0:
        global FLAGS
        FLAGS, extra = parse_args(args)
    images, labels, dataset, classes = get_dataset()
    if overrides is not None:  # {label: generator}
        for label in overrides.keys():
            gen = overrides[label]
            label_index = classes.index(label)
            indices = np.where(labels == label_index)[0]
            if hasattr(gen, 'RATIO'):
                indices = np.random.choice(indices, int(len(indices) * gen.RATIO))
            sample_images = normalize(gen.next())
            # import pdb; pdb.set_trace()
            len_samples = sample_images.shape[0]
            ix_sample = 0
            print(f'CCCCCCCCCCCCCC:{len(indices)}')
            for ix in indices:
                if len_samples == ix_sample:
                    sample_images = normalize(gen.next())
                    len_samples = sample_images.shape[0]
                    ix_sample = 0
                images[ix] = sample_images[ix_sample]
                ix_sample += 1

    (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)
    images = None
    labels = None
    # indices = np.where(testY == label_index)[0]
    # plots = []
    # for i in np.random.choice(indices, 40):
    #     plots.append((testX[i], testY[i]))
    # from utils import plot_images
    # plot_images(plots)
    # exit()

    gen_train = create_generator(trainX, trainY, FLAGS.batch_size)
    gen_test = create_generator(testX, testY, FLAGS.batch_size)

    base_flags = base_parse_args(['-d'])[0]
    base_flags.gen = lambda: (gen_train, gen_test)
    base_flags.classes = classes
    base_flags.classes_name = dataset
    base_flags.model_dir = MODEL_DIR
    base_flags.model_path = f'{base_flags.model_dir}/{base_flags.classes_name}.h5'
    base_flags.length = 1
    base_flags.batch_size = FLAGS.batch_size

    base_train(base_flags)


def test():
    images, labels, dataset, classes = get_dataset()
    model_path = f'{MODEL_DIR}/{dataset}.h5'
    model = load_model(model_path)
    (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)
    gen = create_generator(testX, testY, FLAGS.batch_size)
    # import pdb; pdb.set_trace()

    # evaluate all
    print('Evaluating')
    model.evaluate(gen, steps=ceil(len(testX) / FLAGS.batch_size), verbose=1)

    # show a batch
    images_test, labels_true = next(gen)
    max_count = 40

    labels_pred = model.predict_on_batch(images_test[:max_count])
    labels_pred = np.array(labels_pred).argmax(axis=-1)

    show_test(images_test, labels_true[:max_count], labels_pred, classes)


class Predictor():
    def __init__(self, *args):
        global FLAGS
        FLAGS, extra = parse_args(args)
        dataset, classes = get_classes()
        self.classes = classes
        if hasattr(extra, 'model_path'):
            model_path = extra.model_path
        else:
            model_path = f'{MODEL_DIR}/{dataset}.h5'
        self.model = load_model(model_path)
        print('Make sure images are of BLACK background')

    def next(self, images):
        images = np.array(images)
        if images.shape[-2:] != (28, 28):
            print('The image shape should be (28, 28)')
            exit()
        images = normalize(images)
        one_hots = self.model.predict(images)
        indices = np.array(one_hots).argmax(axis=-1)
        return ''.join(map(lambda ix: self.classes[ix], indices))


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args()

    globals()[FLAGS.cmd]()
