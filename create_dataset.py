import os
import argparse
from captcha.image import ImageCaptcha
from random import randint
import string


def gen_captcha(count, captcha_len, classes, folder):
    for j in range(count):
        if j % 100 == 0:
            print(j)

        while True:
            chars = ''
            for i in range(captcha_len):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--count',
        type=int,
        help='total number of captchas')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')
    FLAGS, unparsed = parser.parse_known_args()
# count = 50000
count = 1
captcha_len = 4
    list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

classes = string.digits + string.ascii_lowercase + string.ascii_uppercase
gen_captcha(count, captcha_len, classes, folder)
