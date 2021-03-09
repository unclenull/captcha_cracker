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
from utils import show_test, clear_folder, FOLDER_TMP_AUG
from recognizer import train as base_train, parse_args as base_parse_args


MODEL_DIR = 'model/single'


def parse_args():
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
        help='train, test'
    )
    FLAGS, unparsed = parser.parse_known_args()
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


def get_dataset():
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

    images, labels = extract_training_samples(dataset)
    images = np.expand_dims(images, -1)
    return images, labels, dataset, classes


def train():
    images, labels, dataset, classes = get_dataset()
    (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)

    gen_train = create_generator(trainX, trainY, FLAGS.batch_size)
    gen_test = create_generator(testX, testY, FLAGS.batch_size)

    base_flags = base_parse_args(['-d', 'foo'])[0]
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

    labels_pred = model.predict_on_batch(images_test)
    labels_pred = np.array(labels_pred).argmax(axis=-1)

    show_test(images_test, labels_true, labels_pred, classes)


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args()

    globals()[FLAGS.cmd]()
