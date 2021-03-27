"""
Recongize captchas (from filesystem or generator) via AlexNet
"""

import os
import numpy as np
from math import ceil
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from utils import parse_args as base_parse_args, get_recognizer_generator, show_metrics, show_test, normalize, get_refiner_custom_objects


def create_model(img_shape, target_model_only=False):
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

    model = Model(inputs=input_tensor, outputs=predicts, name='Alex')

    train_model = None
    if FLAGS.syn is not None and not target_model_only:
        refiner = load_model(FLAGS.refiner_forward_model_path, custom_objects=get_refiner_custom_objects())
        x = model(refiner.output)
        train_model = Model(inputs=refiner.input, outputs=x)
        refiner.trainable = False

    (train_model or model).compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=[SparseCategoricalAccuracy(name='acc')]
    )

    model.summary()
    plot_path = f'{FLAGS.model_dir}/{FLAGS.classes_name}_model.png'
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=plot_path)
    if train_model:
        return train_model, model
    else:
        return model, None


def _gen_data(both=False):
    return get_recognizer_generator(
        FLAGS,
        both,
    )


class RefinedCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.target_model = model
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current = logs.get('val_loss')
        print(current)
        if self.best is not None and self.best <= current:
            return

        self.best = current
        self.target_model.save_weights(FLAGS.model_path)


class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Starting training;")

    def on_train_end(self, logs=None):
        print("Stop training;")

    def on_epoch_begin(self, epoch, logs=None):
        print("Start epoch {} of training;".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training;".format(epoch))

    def on_test_begin(self, logs=None):
        print("Start testing;")

    def on_test_end(self, logs=None):
        print("Stop testing;")

    def on_test_batch_begin(self, batch, logs=None):
        print("...Evaluating: start of batch {};".format(batch))

    def on_test_batch_end(self, batch, logs=None):
        print("...Evaluating: end of batch {};".format(batch))


def train(flags=None):
    if flags is not None:
        global FLAGS
        FLAGS = flags

    gens, img_shape = _gen_data(True)
    if gens is None:
        print('no dataset')
        exit()
    else:
        ((train_gen, train_count), (test_gen, test_count)) = gens
        steps_train = ceil(train_count / FLAGS.batch_size)
        steps_test = ceil(test_count / FLAGS.batch_size)

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    callbacks = [
        CustomCallback(),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=FLAGS.patience,
            verbose=1
        )
    ]

    train_model, target_model = create_model(img_shape)

    if FLAGS.syn is not None:
        callbacks.append(RefinedCallback(target_model))
    else:
        callbacks.append(
            ModelCheckpoint(
                filepath=FLAGS.model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        )
    # import pdb; pdb.set_trace()

    history = train_model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=FLAGS.epochs,
        validation_data=test_gen,
        validation_steps=steps_test,
        callbacks=callbacks,
    )

    show_metrics(history, FLAGS.length, f"{FLAGS.model_dir}/{FLAGS.classes_name}_metrics.png")

    # re-save a light-weight model
    model, _ = create_model(img_shape, True)
    model.load_weights(FLAGS.model_path)
    model.save(FLAGS.model_path)
    print('Training done.')


def test():
    model = load_model(FLAGS.model_path)
    rs = _gen_data()
    if rs is None:
        print('no dataset')
        exit()
    (gen, count), img_shape = rs
    # import pdb; pdb.set_trace()
    steps = ceil(count / FLAGS.batch_size)

    # evaluate all
    print('=================Evaluating=================')
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
    normalize(image)
    pass


def parse_args(args=None):
    return base_parse_args([
        (
            ('-w', '--width'),
            {
                'type': int,
                'help': 'image width',
            }
        ),
        (
            ('--height',),
            {
                'type': int,
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
        (('cmd', ), {'nargs': '?', 'help': 'train, test'}),
    ], args)


if __name__ == '__main__':
    FLAGS, EXTRA = parse_args()
    globals()[FLAGS.cmd]()
