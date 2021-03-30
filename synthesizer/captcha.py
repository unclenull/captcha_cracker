import os
import string
import random
from cached_property import cached_property
from PIL import Image
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from .image import (
    curve,
    noise,
    smooth,
    text,
    resize,
    warp,
    rotate,
)


def arr_to_dict(arr):
    dc = {}
    for f in arr:
        dc[f.__name__] = f
    return dc


Char_filters = arr_to_dict([resize, rotate, warp])
Filters = arr_to_dict([curve, noise, smooth])


class Captcha():
    def __init__(
        self,

        # background
        width=160,
        height=60,
        background=None,  # rgb/func

        # text
        lines=1,  # lines of text
        charset_classes=None,  # 'dul'
        charset=string.digits,
        length=4,
        fonts=None,
        font_sizes=None,  # int, tuple, range
        select_font=None,  # function
        color=(0,) * 3,  # tuple/func(i, char)
        stroke=(0, None),  # size, color

        # position
        offset_x=0,
        offset_y=0,
        padding_x=0,
        padding_y=0,
        even=True,  # consistent spaces
        offset_char_x=0,
        #   even: int/func;
        #   odd(extend random range):
        #     int: => (min=-int, max:+int)
        #     (=min, +max)
        #     func(i, count, max_x): offset_x
        offset_char_y=0,  # False: keep em; True: random; func(i, count, max); others in the middle

        char_filters=None,
        # {
        #   'resize': None,  # for Non-proportional (for proportional use select-font); (int, int)/(float, float)/func(size)
        #   'rotate': None,
        #   'warp': None,  # distortion
        #   'extra': None,  # func(image)
        # }

        # whole
        filters=None,
        # specify order
        # {
        #
        #   'text': True,
        #   'noise': None,
        #   'curve': None,  # occluding line
        #   'smooth' None,
        #   'extra': func,  # custom filter
        # }
    ):
        self.width = width
        self.height = height
        self._background = background
        if isinstance(background, tuple):
            self.bg_color = background
        else:
            self.bg_color = (255,) * 3

        self.lines = lines  # lines of text (TODO)
        if charset_classes is not None:
            self.charset = ''
            if '0' in charset_classes:
                self.charset += string.digits
            if 'a' in charset_classes:
                self.charset += string.ascii_uppercase
            if 'A' in charset_classes:
                self.charset += string.ascii_lowercase
        self.length = length
        self._fonts = fonts
        self._font_sizes = font_sizes
        self.select_font = select_font
        self._color = color
        self.stroke = stroke

        self.even = even
        self._offset_x = offset_x
        self._offset_y = offset_y
        self.offset_char_x = offset_char_x
        self.offset_char_y = offset_char_y
        self.padding_x = padding_x
        self.padding_y = padding_y

        self.char_filters = char_filters
        self.filters = filters or {'text': True}

    @cached_property
    def background(self):
        if callable(self._background):
            return self._background
        else:
            return None

    @cached_property
    def default_font_path(self):
        from os.path import join, dirname, abspath
        return join(dirname(abspath(__file__)), 'CourierNew-Bold.ttf')

    @cached_property
    def fonts(self):
        if self._fonts is None:
            self._fonts = [self.default_font_path]

        return tuple(
            [
                truetype(name, size)
                for name in self._fonts
                for size in self.font_sizes
            ]
        )

    @cached_property
    def font_sizes(self):
        if self._font_sizes is None:
            self._font_sizes = (self.height,)
        if isinstance(self._font_sizes, int):
            self._font_sizes = (self._font_sizes,)
        return self._font_sizes

    @cached_property
    def color(self):
        if not callable(self._color):
            def color(*args):
                return self._color
            return color
        return self._color

    @cached_property
    def offset_x(self):
        if not callable(self._offset_x):
            def offset_x(*args):
                return self._offset_x
            return offset_x
        return self._offset_x

    @cached_property
    def offset_y(self):
        if isinstance(self._offset_y, int):
            def offset_y(*args):
                return self._offset_y
            return offset_y
        return self._offset_y

    @cached_property
    def drawings(self):
        text_drawings = []
        if self.char_filters:
            for k in self.char_filters.keys():
                if k in Char_filters:
                    text_drawings.append(Char_filters[k](self.char_filters[k]))
                else:  # custom
                    text_drawings.append(self.char_filters[k])

        drawings = []
        if self.background is not None:
            drawings.append(self.background)
        if self.filters:
            for k in self.filters.keys():
                if k == 'text':
                    drawings.append(
                        text(
                            fonts=self.fonts,
                            select_font=self.select_font,
                            drawings=text_drawings,
                            color=self.color,
                            stroke=self.stroke,
                            even=self.even,
                            offset_x=self.offset_x,
                            offset_y=self.offset_y,
                            offset_char_x=self.offset_char_x,
                            offset_char_y=self.offset_char_y,
                            padding_x=self.padding_x,
                            padding_y=self.padding_y
                        )
                    )

                elif k in Filters:
                    drawings.append(Filters[k](self.filters[k]))
                else:  # custom
                    drawings.append(self.filters[k])

        return drawings

    def get_one(self, chars=None, no_label=False):
        if chars is None:
            chars = random.choices(self.charset, k=self.length)

        image = Image.new("RGB", (self.width, self.height), self.bg_color)
        for drawing in self.drawings:
            image = drawing(image, chars)
            assert image
        if no_label:
            return image
        else:
            return image, chars

    def get_batch(self, batch_size, chars=None, no_label=False):
        arrs = []
        for _ in range(batch_size):
            arrs.append(self.get_one(chars, no_label=no_label))
        return arrs

    def test_fontsize(self, path=None, size=None):
        if size is None:
            print('No font size is given')
            exit()
        # import pdb; pdb.set_trace()
        if path is None:
            path = self.default_font_path
        font = truetype(path, size)
        font_cap = truetype(path, int(size / 3))

        text = 'awbp69E'
        count = len(text)
        offset_x = size
        width_canvas = size * count + offset_x * 2
        height_canvas = size * 3
        line_color = (100, 230, 100)
        bg_canvas = (230,) * 3
        bg_char = (0,) * 3
        bg_charbox = (90,) * 3
        color_cap = (0, 0, 200)
        canvas = Image.new("RGB", (width_canvas, height_canvas), bg_canvas)
        draw_canvas = Draw(canvas)
        draw_canvas.text((0, 0), f'font-size: {size}', font=font_cap, fill=(200, 0, 0))
        draw_canvas.line((0, size - 1, width_canvas, size - 1), line_color)
        draw_canvas.line((0, size * 2 + 1, width_canvas, size * 2 + 1), line_color)
        height_cap = font_cap.getsize('w')[1]
        draw_canvas.text((size / 2, size - height_cap), 'w', font=font_cap, fill=color_cap)
        draw_canvas.text((size / 2, size * 2), 'h', font=font_cap, fill=color_cap)

        for c in text:
            c_width, c_height = font.getsize(c, stroke_width=2)
            char_image = Image.new("RGB", (c_width, c_height), bg_char)
            draw_char = Draw(char_image)
            bbox = font.getbbox(c, stroke_width=2)
            draw_char.rectangle(bbox, fill=bg_charbox)
            draw_char.text((-bbox[0], 0), c, font=font, fill=(255,) * 3, stroke_width=2, stroke_fill=(255, 0, 0))    # -1 -> 1
            canvas.paste(char_image, (offset_x, size))
            (left, top, right, bottom) = bbox
            width = right - left
            height = bottom - top
            draw_canvas.text((offset_x, size - height_cap), str(width), font=font_cap, fill=color_cap)
            draw_canvas.text((offset_x, size * 2), str(c_height), font=font_cap, fill=color_cap)
            draw_canvas.text((offset_x, size * 2 + height_cap), str(height), font=font_cap, fill=color_cap)
            offset_x += c_width + 1

        canvas.show()
