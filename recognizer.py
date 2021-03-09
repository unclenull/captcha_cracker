"""
Recongize captchas (from filesystem or generator) via AlexNet
"""

import os
import numpy as np
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from utils import parse_args as base_parse_args, create_generator, show_metrics, show_test


def create_model(img_shape):
    # Alex
    input_tensor = Input(shape=img_shape)

    # 1st Fully Connected Layer
    x = Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # 2nd Fully Connected Layer
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # 3rd Fully Connected Layer
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # 4th Fully Connected Layer
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # 5th Fully Connected Layer
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
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
        metrics=[SparseCategoricalAccuracy(name='acc')]
    )
    model.summary()
    plot_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}_model.png'
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=plot_path)
    return model


def _gen_data(both=False, no_first=False):
    return create_generator(
        FLAGS,
        both,
        no_first=no_first,
    )


def train(flags=None):
    if flags is not None:
        global FLAGS
        FLAGS = flags

    gens = _gen_data(True)
    if gens is None:
        print('no dataset')
        exit()
    else:
        (train_gen, test_gen) = gens
        img_shape = train_gen.img_shape
        steps_train = ceil(train_gen.count / FLAGS.batch_size)
        steps_test = ceil(test_gen.count / FLAGS.batch_size)

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

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

    model = create_model(img_shape)
    # import pdb; pdb.set_trace()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=999,
        validation_data=test_gen,
        validation_steps=steps_test,
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.length, f"{FLAGS.model_dir}/{FLAGS.classes_name}_metrics.png")

    # re-save a light-weight model
    model = create_model(img_shape)
    model.load_weights(FLAGS.model_path)
    model.save(FLAGS.model_path)


def test():
    model = load_model(FLAGS.model_path)
    gen = _gen_data()
    if gen is None:
        print('no dataset')
        exit()

    steps = ceil(gen.count / FLAGS.batch_size)

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


def predict(image):
    pass


def parse_args(args=None):
    return base_parse_args([
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
                'help': 'how many samples to generate',
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
    ], args)


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args()
    globals()[FLAGS.cmd]()
