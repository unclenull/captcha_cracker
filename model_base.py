import numpy as np
import glob
import cv2
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from utils import parse_args, data_generator, show_metrics, show_test, create_model


def gen_data(images, batch_size, img_shape):
    return data_generator(
        images=images,
        batch_size=FLAGS.batch_size,
        img_shape=img_shape,
        letter_count=FLAGS.size,
        classes=FLAGS.classes,
        preprocess_input=preprocess_input,
    )


def train(FLAGS):
    train_samples = glob.glob(f'{FLAGS.base_dataset_path}/train/*.jpg')
    test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*.jpg')
    nb_train = len(train_samples)
    nb_test = len(test_samples)
    img_shape = cv2.imread(train_samples[0]).shape

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

    model = create_model(True, img_shape, FLAGS)
    history = model.fit(
        gen_data(train_samples, FLAGS.batch_size, img_shape),
        steps_per_epoch=ceil(nb_train / FLAGS.batch_size),
        epochs=99,
        validation_data=gen_data(test_samples, FLAGS.batch_size, img_shape),
        validation_steps=ceil(nb_test / FLAGS.batch_size),
        callbacks=callbacks,
    )

    print('Training done.')
    show_metrics(history, FLAGS.size, f"{FLAGS.base_models_dir}/{FLAGS.classes_name}_metrics_base.png")


def test():
    model = load_model(FLAGS.base_model_path)
    test_samples = glob.glob(f'{FLAGS.base_dataset_path}/test/*.jpg')
    nb_test = len(test_samples)

    # evaluate all
    print('Evaluating')
    gen = data_generator(test_samples, FLAGS.batch_size)
    model.evaluate(gen, steps=ceil(nb_test / FLAGS.batch_size), verbose=1)

    # show a batch
    show_size = 32
    gen = data_generator(test_samples, show_size, True)
    images_test, labels_test_raw = next(gen)
    labels_true = [np.array(labels_test_raw)[:, i] for i in range(show_size)]

    labels_pred_raw = model.predict_on_batch(images_test)
    labels_pred = np.array(labels_pred_raw).argmax(axis=-1)
    labels_pred = [np.array(labels_pred)[:, i] for i in range(show_size)]
    # import pdb; pdb.set_trace()

    show_test(images_test, labels_true, labels_pred, FLAGS.classes)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args([(('cmd',),)])

    globals()[FLAGS.cmd](FLAGS)
