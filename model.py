import os
import numpy as np
import cv2
from math import ceil
from pandas import DataFrame
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import parse_args, parse_label, show_metrics, show_test, create_model


def _create_generator(augment_args=None):
    args = dict(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        channel_shift_range=0.2,
    )

    if augment_args is not None:
        args.update(augment_args)

    return ImageDataGenerator(**args)


def _get_gen(folder, files, target_size, FLAGS, augment_args):
    df = DataFrame()
    df['filename'] = files
    labels = np.array([parse_label(img, FLAGS.classes) for img in files])
    y_col = []
    for i in range(FLAGS.size):
        clm = f'class{i}'
        y_col.append(clm)
        df[clm] = labels[:, i]

    return _create_generator(augment_args).flow_from_dataframe(
        df,
        y_col=y_col,
        directory=folder,
        target_size=target_size,
        batch_size=FLAGS.batch_size,
        classes=FLAGS.classes,
        class_mode='multi_output'
    )


def _get_shape(folder):
    filename = ''
    for root, dirs, files in os.walk(folder):
        for name in files:
            filename = name
            break
        break
    return cv2.imread(f'{folder}/{filename}').shape


def aug(FLAGS, extra):
    folder = f'{FLAGS.dataset_path}/'
    _create_generator({'preprocessing_function': None}) \
        .flow_from_directory(
            folder,
            save_to_dir='tmp',
            target_size=_get_shape(f'{folder}/train')[:2]
        ).next()


def train(FLAGS, extra):
    train_folder = f'{FLAGS.dataset_path}/train/'
    test_folder = f'{FLAGS.dataset_path}/test/'
    train_samples = os.listdir(train_folder)
    nb_train = len(train_samples)
    test_samples = os.listdir(test_folder)
    nb_test = len(test_samples)
    img_shape = cv2.imread(f'{train_folder}/{train_samples[0]}').shape

    if not os.path.exists(FLAGS.models_dir):
        os.mkdir(FLAGS.models_dir)

    callbacks = [
        ModelCheckpoint(
            filepath=FLAGS.model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=2,
            verbose=1
        )
    ]

    model = create_model(False, img_shape, FLAGS)
    model.load_weights(FLAGS.base_model_path)
    model.summary()

    history = model.fit(
        _get_gen(train_folder, train_samples, img_shape[:2], FLAGS, extra),
        steps_per_epoch=ceil(nb_train / FLAGS.batch_size),
        epochs=99,
        validation_data=_get_gen(test_folder, test_samples, img_shape[:2], FLAGS, extra),
        validation_steps=ceil(nb_test / FLAGS.batch_size),
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.size, f"{FLAGS.models_dir}/{FLAGS.classes_name}_metrics.png")


def test(FLAGS, extra):
    model = load_model(FLAGS.model_path)

    test_folder = f'{FLAGS.dataset_path}/test/'
    test_samples = os.listdir(test_folder)
    nb_test = len(test_samples)
    img_shape = cv2.imread(f'{test_folder}/{test_samples[0]}').shape

    # evaluate all
    print('Evaluating')
    gen = _get_gen(test_folder, test_samples, img_shape[:2], FLAGS, extra)
    model.evaluate(gen, steps=ceil(nb_test / FLAGS.batch_size), verbose=1)

    # show a batch
    show_size = 32
    gen = _get_gen(test_folder, test_samples, img_shape[:2], FLAGS, extra)
    images_test, labels_test_raw = next(gen)
    labels_true = [np.array(labels_test_raw)[:, i] for i in range(show_size)]

    labels_pred_raw = model.predict_on_batch(images_test)
    labels_pred = np.array(labels_pred_raw).argmax(axis=-1)
    labels_pred = [np.array(labels_pred)[:, i] for i in range(show_size)]
    # import pdb; pdb.set_trace()

    show_test(images_test, labels_true, labels_pred, FLAGS.classes)


if __name__ == '__main__':
    FLAGS, extra = parse_args([(('cmd',),)])

    globals()[FLAGS.cmd](FLAGS, extra)
