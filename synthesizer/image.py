import numpy as np
import random
from PIL import Image, ImageFilter
from PIL.ImageDraw import Draw


def smooth():
    def drawer(image, text):
        return image.filter(ImageFilter.SMOOTH)

    return drawer


def curve(color="#5C87B2", width=4, number=6):
    from .bezier import make_bezier

    if not callable(color):
        c = getrgb(color)

        def color():
            return c

    def drawer(image, text):
        dx, height = image.size
        dx = dx / number
        path = [(dx * i, random.randint(0, height)) for i in range(1, number)]
        bcoefs = make_bezier(number - 1)
        points = []
        for coefs in bcoefs:
            points.append(
                tuple(
                    sum([coef * p for coef, p in zip(coefs, ps)])
                    for ps in zip(*path)
                )
            )
        draw = Draw(image)
        draw.line(points, fill=color(), width=width)
        return image

    return drawer


def noise(number=50, color="#EEEECC", level=2):
    if not callable(color):
        c = getrgb(color)

        def color():
            return c

    def drawer(image, text):
        width, height = image.size
        dx = width / 10
        width = width - dx
        dy = height / 10
        height = height - dy
        draw = Draw(image)
        for _ in range(number):
            x = int(random.uniform(dx, width))
            y = int(random.uniform(dy, height))
            draw.line(((x, y), (x + level, y)), fill=color(), width=level)
        return image

    return drawer


def text(
    fonts=None,
    select_font=None,
    drawings=None,
    color=None,
    stroke=(0, None),
    even=None,
    offset_x=None,
    offset_y=None,
    offset_char_x=None,
    offset_char_y=None,
    padding_x=None,
    padding_y=None,
):
    offset_x = offset_x()
    offset_y = offset_y()
    stroke_size, stroke_color = stroke

    if offset_char_y is True:  # random
        def get_offset_char_y(i, count, max_y):
            return random.choice(range(max_y))
    elif callable(offset_char_y):
        get_offset_char_y = offset_char_y
    else:
        def get_offset_char_y(i, count, max_y):
            return int(max_y / 2)

    if not even:
        if isinstance(offset_char_x, int):  # (-int, +int)
            def get_offset_char_x(i, count, max_x):
                if i > 0:
                    min_x = -offset_char_x
                else:
                    min_x = 0
                if i < count - 1:
                    max_x += offset_char_x
                if min_x >= max_x:
                    print(f'Horizontal space is not enough!!! {min_x} >= {max_x}')
                    exit()
                return random.choice(range(min_x, max_x))
        elif isinstance(offset_char_x, tuple):
            def get_offset_char_x(i, count, max_x):
                if i > 0:
                    min_x = offset_char_x[0]
                else:
                    min_x = 0
                if i < count - 1:
                    max_x += offset_char_x[1]
                if min_x >= max_x:
                    print(f'Horizontal space is not enough!!! {min_x} >= {max_x}')
                    exit()
                return random.choice(range(min_x, max_x))
        elif callable(offset_char_x):
            get_offset_char_x = offset_char_x

    def get_dimen(image):
        width, height = image.size
        width = width - padding_x * 2
        if width < 0:
            print(f'Horizontal space {width} is not enough, padding x {padding_x} is too large!!!')
            exit()
        height = height - padding_y * 2
        if height < 0:
            print(f'Vertical space {height} is not enough, padding y {padding_y} is too large!!!')
            exit()
        return width, height

    def copy_even(image, chars):
        # import pdb; pdb.set_trace()
        width, height = get_dimen(image)
        if callable(offset_char_x):
            _offset_char_x = offset_char_x()
        else:
            _offset_char_x = offset_char_x
        count = len(chars)
        width_chars = sum(c.size[0] for c in chars) + _offset_char_x * (count - 1)
        left = padding_x + offset_x + int((width - width_chars) / 2)
        for i, c in enumerate(chars):
            c_width, c_height = c.size
            # import pdb; pdb.set_trace()
            space_y = height - c_height
            if space_y < 0:
                print(f'Vertical space is not enough!!! {i}: {height} <= {c_height}')
                exit()
            top = padding_y + offset_y + get_offset_char_y(i, count, space_y)
            image.paste(c, (left, top), c)
            # mask = c.convert("L").point(lambda i: i * 1.8)
            # image.paste(c, (left, top), mask)
            left += c_width + _offset_char_x

    def copy_odd(image, chars):
        width, height = get_dimen(image)
        count = len(chars)
        char_space_x = round(width / count)
        left = padding_x + offset_x

        for i, c in enumerate(chars):
            print(np.array(c))
            c_width, c_height = c.size
            space_y = height - c_height
            if space_y < 0:
                print(f'Vertical space is not enough!!! {i}: {height} <= {c_height}')
                exit()
            top = padding_y + offset_y + get_offset_char_y(i, count, space_y)
            max_x = char_space_x - c_width
            # mask = c.convert("L")
            # print(i, '---------------------')
            # print(np.array(mask))
            # mask = mask.point(lambda i: i * 1.8)
            # print(i, '=====================')
            # print(np.array(mask))
            image.paste(c, (left + get_offset_char_x(i, count, max_x), top), c)
            left += char_space_x

    def drawer(image, text):
        draw = Draw(image)
        char_images = []
        height_fixed = None
        min_y = 0
        if len(fonts) == 1:
            font = fonts[0]
            if offset_char_y is False:  # keep em only for single font
                offsets = [font.getbbox(c, stroke_width=stroke_size) for c in text]
                offsets = np.array(offsets)
                min_y = min(offsets[:, 1])
                max_y = max(offsets[:, 3])
                height_fixed = max_y - min_y
        for i, c in enumerate(text):
            if select_font is not None:
                font = select_font(i, fonts)
            else:
                font = random.choice(fonts)
            # import pdb; pdb.set_trace()
            c_width, c_height = draw.textsize(c, font=font, stroke_width=stroke_size)
            char_image = Image.new("RGBA", (c_width, height_fixed or c_height), (0,) * 4)
            char_draw = Draw(char_image)
            char_draw.text((-font.getoffset(c)[0], -min_y), c, font=font, fill=color(i, c), stroke_width=stroke_size, stroke_fill=stroke_color)
            if offset_char_y is not False:
                char_image = char_image.crop(char_image.getbbox())
            for drawing in drawings:
                char_image = drawing(i, c, char_image)
            char_images.append(char_image)

        if even:
            copy_even(image, char_images)
        else:
            copy_odd(image, char_images)
        return image

    return drawer


# region: text drawers


def warp(dx_factor=0.27, dy_factor=0.21):
    def drawer(i, c, image):
        width, height = image.size
        dx = width * dx_factor
        dy = height * dy_factor
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        image2 = Image.new("RGB", (width + abs(x1) + abs(x2), height + abs(y1) + abs(y2)))
        image2.paste(image, (abs(x1), abs(y1)))
        width2, height2 = image2.size
        return image2.transform(
            (width, height),
            Image.QUAD,
            (
                x1,
                y1,
                -x1,
                height2 - y2,
                width2 + x2,
                height2 + y2,
                width2 - x2,
                -y1,
            ),
        )

    return drawer


def rotate(angle=25):
    def drawer(i, c, image):
        return image.rotate(
            random.uniform(-angle, angle), Image.BILINEAR, expand=1
        )

    return drawer


def resize(cfg):
    if callable(cfg):
        resize = cfg
    else:
        if isinstance(cfg[0], float):
            def resize(width, height):
                return round(width * cfg[0]), round(height * cfg[1])
        elif isinstance(cfg[0], int):
            def resize(width, height):
                return cfg[0], cfg[1]
        else:
            print('Invalid arg "resize"')
            exit()

    def drawer(i, c, image):
        return image.resize((resize(*image.size)))

    return drawer
