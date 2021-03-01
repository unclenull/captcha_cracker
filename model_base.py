import os
import sys
import numpy as np
import glob
import cv2
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from utils import parse_args, data_generator_from_fs, data_generator_from_gen, show_metrics, show_test


def _gen_data(gen, images, img_shape, no_first=False):
    return gen(
        images,
        img_shape=img_shape,
        no_first=no_first,
        batch_size=FLAGS.batch_size,
        letter_count=FLAGS.length,
        classes=FLAGS.classes,
    )


def create_model(img_shape):
    # Alex
    input_tensor = Input(shape=img_shape)

    # 1st Fully Connected Layer
    x = Conv2D(32, 3, activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # 2nd Fully Connected Layer
    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # 3rd Fully Connected Layer
    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)

    # 4th Fully Connected Layer
    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)

    # 5th Fully Connected Layer
    x = Conv2D(128, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Passing it to a Fully Connected layer
    x = Flatten()(x)

    # 1st Fully Connected Layer
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # 2nd Fully Connected Layer
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # 3rd Fully Connected Layer
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predicts = [Dense(len(FLAGS.classes), name=f'c{i}', activation='softmax')(x) for i in range(FLAGS.length)]

    model = Model(inputs=input_tensor, outputs=predicts)

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    plot_path = f'{FLAGS.base_model_dir}/{FLAGS.classes_name}_model_base.png'
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=plot_path)
    return model


def train():
    if os.path.isdir(FLAGS.base_dataset_dir):
        train_samples = glob.glob(f'{FLAGS.base_dataset_path}/train/*')
        img_shape = cv2.imread(train_samples[0], cv2.IMREAD_GRAYSCALE).shape
        img_shape = np.expand_dims(img_shape, -1)
        train_gen = _gen_data(data_generator_from_fs, train_samples, img_shape)
        steps_train = ceil(len(train_samples) / FLAGS.batch_size)

        test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*')
        test_gen = _gen_data(data_generator_from_fs, test_samples, img_shape)
        steps_test = ceil(len(test_samples) / FLAGS.batch_size),
    elif os.path.isfile(f'{FLAGS.base_dataset_dir}.py'):
        sys.path.insert(0, os.getcwd())
        gen = __import__(f'{FLAGS.base_dataset_dir}').Generator
        img_shape = FLAGS.height, FLAGS.width, 1
        train_gen = _gen_data(data_generator_from_gen, gen, img_shape)
        test_gen = _gen_data(data_generator_from_gen, gen, img_shape)
        steps_train = FLAGS.samples // FLAGS.batch_size
        steps_test = steps_train * FLAGS.test_ratio
    else:
        print('no dataset')
        exit()

    if not os.path.exists(FLAGS.base_model_dir):
        os.mkdir(FLAGS.base_model_dir)

    callbacks = [
        ModelCheckpoint(
            filepath=FLAGS.base_model_path,
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

    model = create_model(img_shape)
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=999,
        validation_data=test_gen,
        validation_steps=steps_test,
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.length, f"{FLAGS.base_model_dir}/{FLAGS.classes_name}_metrics_base.png")

    # re-save a light-weight model
    model = create_model(img_shape)
    model.load_weights(FLAGS.base_model_path)
    model.save(FLAGS.base_model_path)


def test():
    model = load_model(FLAGS.base_model_path)
    if os.path.isdir(FLAGS.base_dataset_dir):
        test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*')
        steps = ceil(len(test_samples) / FLAGS.batch_size)
        img_shape = cv2.imread(test_samples[0]).shape
        img_shape = np.expand_dims(img_shape, -1)
        gen = _gen_data(data_generator_from_fs, test_samples, img_shape)
    elif os.path.isfile(f'{FLAGS.base_dataset_dir}.py'):
        sys.path.insert(0, os.getcwd())
        gen = __import__(f'{FLAGS.base_dataset_dir}').Generator
        img_shape = FLAGS.height, FLAGS.width, 1
        gen = _gen_data(data_generator_from_gen, gen, img_shape)
        steps = FLAGS.samples // FLAGS.batch_size * FLAGS.test_ratio
    else:
        print('no dataset')
        exit()

    # evaluate all
    print('Evaluating')
    model.evaluate(gen, steps=steps, verbose=1)

    # show a batch
    images_test, labels_test_raw = next(gen)
    count = len(images_test)
    labels_true = [np.array(labels_test_raw)[:, i] for i in range(count)]

    labels_pred_raw = model.predict_on_batch(images_test)
    # labels_pred_raw = model.predict(images_test)
    labels_pred = np.array(labels_pred_raw).argmax(axis=-1)
    labels_pred = [np.array(labels_pred)[:, i] for i in range(count)]

    show_test(images_test, labels_true, labels_pred, FLAGS.classes)


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args([
        (
            ('-w', '--width'),
            {
                'default': 100,
                'type': float,
                'help': 'image width',
            }
        ),
        (
            ('--height',),
            {
                'default': 50,
                'type': float,
                'help': 'image height',
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
            ('-s', '--samples'),
            {
                'default': 50000,
                'type': int,
            }
        ),
        (
            ('-t', '--test-ratio'),
            {
                'default': 0.2,
                'type': float,
                'help': 'ratio of count of images for evaluating',
            }
        ),
        (('cmd', ), {'help': 'train, test'}),
    ])

    globals()[FLAGS.cmd]()
