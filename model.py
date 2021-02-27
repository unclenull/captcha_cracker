import os
import numpy as np
import cv2
from math import ceil
from pandas import DataFrame
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout
from utils import parse_args, parse_label, show_metrics, show_test


def _create_generator(augment_args=None, preprocess=True):
    if augment_args is False:
        args = {}
        augment_args = None
    else:
        args = dict(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            channel_shift_range=0.2,
        )

    if preprocess is True:
        args['preprocessing_function'] = preprocess_input

    if augment_args is not None:
        args.update(augment_args)

    return ImageDataGenerator(**args)


def _get_gen(folder, files, target_size, augment_args):
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


def show_aug():
    folder = f'{FLAGS.dataset_path}/'
    folder_target = 'aug_showcase'
    if not os.path.exists(folder_target):
        os.mkdir(folder_target)

    _create_generator(EXTRA if FLAGS.augment else False, False) \
        .flow_from_directory(
            folder,
            save_to_dir=folder_target,
            target_size=_get_shape(f'{folder}/train')[:2]
        ).next()  # noqa

    print(f'Sample augmented images are saved in {folder_target}')


def create_model():
    if FLAGS.retrain:
        if FLAGS.trainable == 0:
            print('No trainable is set for retraining')
            exit()
        model = load_model(FLAGS.model_path)
    else:
        model = load_model(FLAGS.base_model_path)
        if FLAGS.new:
            x = model.layers[- 1 - FLAGS.size].output
            x = Flatten(name="flatten")(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.2, name='dropout_refining')(x)
            predicts = [Dense(len(FLAGS.classes), name=f'c{i}', activation='softmax')(x) for i in range(FLAGS.size)]
            model = Model(inputs=model.input, outputs=predicts)

            FLAGS.trainable = 0  # warm up

    layers_count = len(model.layers)
    if FLAGS.trainable > 0:
        endIndex = ceil(layers_count * (1 - FLAGS.trainable))
    else:  # only the classifiers
        endIndex = layers_count - FLAGS.size
    print(f'Trainable layers count: {layers_count - endIndex}')

    for layer in model.layers[:endIndex]:
        layer.trainable = False

    for layer in model.layers[endIndex:]:
        layer.trainable = not isinstance(layer, BatchNormalization)

    model.compile(
        optimizer=Adam(lr=FLAGS.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    plot_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}_model.png'
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=plot_path)
    return model


def train():
    train_folder = f'{FLAGS.dataset_path}/train/'
    test_folder = f'{FLAGS.dataset_path}/test/'
    train_samples = os.listdir(train_folder)
    nb_train = len(train_samples)
    test_samples = os.listdir(test_folder)
    nb_test = len(test_samples)

    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

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
            patience=FLAGS.patience,
            verbose=1
        )
    ]

    model = create_model()
    img_shape = model.input.shape[1:]

    history = model.fit(
        _get_gen(train_folder, train_samples, img_shape[:2], EXTRA if FLAGS.augment else False),
        steps_per_epoch=ceil(nb_train / FLAGS.batch_size),
        epochs=999,
        validation_data=_get_gen(test_folder, test_samples, img_shape[:2], EXTRA if FLAGS.augment else False),
        validation_steps=ceil(nb_test / FLAGS.batch_size),
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.size, f"{FLAGS.model_dir}/{FLAGS.classes_name}_metrics.png")


def test():
    model = load_model(FLAGS.model_path)
    # model = load_model('model/4_dul - 副本.h5')
    # model = load_model(FLAGS.base_model_path)
    # import pdb; pdb.set_trace()

    test_folder = f'{FLAGS.dataset_path}/test/'
    test_samples = os.listdir(test_folder)
    nb_test = len(test_samples)
    img_shape = model.input.shape[1:]

    # evaluate all
    print('Evaluating')
    gen = _get_gen(test_folder, test_samples, img_shape[:2], False)
    model.evaluate(gen, steps=ceil(nb_test / FLAGS.batch_size), verbose=1)

    # show a batch
    gen = _get_gen(test_folder, test_samples, img_shape[:2], False)
    images_test, labels_test_raw = next(gen)
    labels_true = [np.array(labels_test_raw)[:, i] for i in range(FLAGS.batch_size)]

    labels_pred_raw = model.predict_on_batch(images_test)
    labels_pred = np.array(labels_pred_raw).argmax(axis=-1)
    labels_pred = [np.array(labels_pred)[:, i] for i in range(FLAGS.batch_size)]

    show_test(images_test, labels_true, labels_pred, FLAGS.classes)


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args([
        (
            ('-n', '--new'),
            {
                'action': 'store_true',
                'help': 'create new model classifiers',
            }
        ),
        (
            ('-t', '--trainable'),
            {
                'default': 0,
                'type': float,
                'help': 'ratio of trainable layers',
            }
        ),
        (
            ('--lr', '--learning-rate'),
            {
                'default': 0.0001,
                'type': float,
            }
        ),
        (
            ('-p', '--patience'),
            {
                'default': 3,
                'type': int,
            }
        ),
        (
            ('-a', '--augment'),
            {
                'action': 'store_true',
            }
        ),
        (
            ('-r', '--retrain'),
            {
                'action': 'store_true',
                'help': 'further training with new trainable ratio',
            }
        ),
        (('cmd', ), {'help': 'train, test, show_aug'}),
    ])

    globals()[FLAGS.cmd]()
