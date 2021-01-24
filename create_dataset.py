import os
from captcha.image import ImageCaptcha
from random import randint
from utils import parse_args


def gen_captcha(count, width, classes, folder):
    for j in range(count):
        if j % 100 == 0:
            print(j)

        while True:
            chars = ''
            for i in range(width):
                rand_num = randint(0, 61)
                chars += classes[rand_num]
            filename = '{}/{}.jpg'.format(folder, chars)
            if os.path.exists(filename):
                continue
            image = ImageCaptcha().generate_image(chars)
            try:
                image.save(filename)
            except Exception:
                print('Invalid: {}'.format(filename))
                continue

            break


if __name__ == '__main__':
    extra = [(
        ('-c', '--count'),
        dict(
            type=int,
            help='total number of captchas'
        )
    ), (
        ('-t', '--test'),
        dict(
            default=0.1,
            type=float,
            help='ratio of test dataset.'
        )
    )]

    FLAGS = parse_args(extra)
    classes, base_dataset_dir, base_dataset_name, width = FLAGS.classes, FLAGS.base_dataset_dir, FLAGS.base_dataset_name, FLAGS.width

    len_classes = len(classes)
    if len_classes == 0:
        print('No char range set!')
        exit()

    count = FLAGS.count
    if count is None:
        count = len_classes * 1000

    count_test = int(count * FLAGS.test)
    count_train = count - count_test

    runs = []
    if count_train > 0:
        runs.append((count_train, '{}/{}/train'.format(base_dataset_dir, base_dataset_name)))
    if count_test > 0:
        runs.append((count_test, '{}/{}/test'.format(base_dataset_dir, base_dataset_name)))

    for c, f in runs:
        print('Generating into {}... (count: {}, width: {}, classes: {})'.format(f, c, width, len_classes))
        command = input('Type Enter/n:')
        if command != 'n':
            if not os.path.exists(f):
                os.makedirs(f)
            gen_captcha(count=c, width=width, classes=classes, folder=f)
