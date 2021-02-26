import os
import numpy as np
import glob
import cv2
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from utils import parse_args, data_generator, show_metrics, show_test


def _gen_data(images, img_shape, no_first=False):
    return data_generator(
        img_shape=img_shape,
        images=images,
        no_first=no_first,
        batch_size=FLAGS.batch_size,
        letter_count=FLAGS.size,
        classes=FLAGS.classes,
    )


def create_model(img_shape):
    base_model = Xception(
        weights=None,
        include_top=False,
        pooling='avg'
    )
    # import pdb; pdb.set_trace()
    input_layer = Input(shape=img_shape)
    x = base_model(input_layer)
    x = Dropout(0.5)(x)

    predicts = [Dense(len(FLAGS.classes), activation='softmax')(x) for i in range(FLAGS.size)]

    model = Model(inputs=input_layer, outputs=predicts)

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
    train_samples = glob.glob(f'{FLAGS.base_dataset_path}/train/*.jpg')
    test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*.jpg')
    nb_train = len(train_samples)
    nb_test = len(test_samples)
    img_shape = cv2.imread(train_samples[0]).shape

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
            patience=2,
            verbose=1
        )
    ]

    model = create_model(img_shape)
    history = model.fit(
        _gen_data(train_samples, img_shape),
        steps_per_epoch=ceil(nb_train / FLAGS.batch_size),
        epochs=99,
        validation_data=_gen_data(test_samples, img_shape),
        validation_steps=ceil(nb_test / FLAGS.batch_size),
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.size, f"{FLAGS.base_model_dir}/{FLAGS.classes_name}_metrics_base.png")

    # re-save a light-weight model
    model = create_model(img_shape)
    model.load_weights(FLAGS.base_model_path)
    model.save(FLAGS.base_model_path)


def test():
    model = load_model(FLAGS.base_model_path)
    test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*')
    nb_test = len(test_samples)
    img_shape = cv2.imread(f'{test_samples[0]}').shape

    # evaluate all
    print('Evaluating')
    gen = _gen_data(test_samples, img_shape)
    model.evaluate(gen, steps=ceil(nb_test / FLAGS.batch_size), verbose=1)

    # show a batch
    gen = _gen_data(test_samples, img_shape, True)
    images_test, labels_test_raw = next(gen)
    labels_true = [np.array(labels_test_raw)[:, i] for i in range(FLAGS.batch_size)]

    labels_pred_raw = model.predict_on_batch(images_test)
    labels_pred = np.array(labels_pred_raw).argmax(axis=-1)
    labels_pred = [np.array(labels_pred)[:, i] for i in range(FLAGS.batch_size)]
    # import pdb; pdb.set_trace()

    show_test(images_test, labels_true, labels_pred, FLAGS.classes)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args([(('cmd', ), {'help': 'train, test'})])

    globals()[FLAGS.cmd]()
