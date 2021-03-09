import os
import string
import random
from cached_property import cached_property
from comp import Draw, Image, ImageFilter, getrgb, truetype
from image import (
    background,
    curve,
    noise,
    # offset,
    rotate,
    smooth,
    text,
    warp,
)

HEIGHT = 60


class Captcha(object):
    def __init__(
        self,

        width=160,
        height=HEIGHT,
        background=(255, 255, 255),
        noise=None,
        curve=None,  # occluding line
        extra_bg=None,  # other background processes

        lines=1,  # lines of text
        char_set=string.digits,
        length=4,
        fonts=['CourierNew-Bold'],
        font_sizes=[HEIGHT],
        color=None,
        rotate=None,
        resize=None,
        warp=None,  # distortion
        xspace=None,  # horizontal spacing
        yspace=None,  # vertical spacing
        extra_text=None,  # other text processes
    ):
        self.width = width
        self.height = height
        self._background = background
        self._noise = noise
        self._curve = curve  # occluding line
        self.extra_bg = extra_bg  # other background processes

        self.lines = lines  # lines of text (TODO)
        self.char_set = char_set
        self.length = length
        self._fonts = fonts
        self.font_sizes = font_sizes
        self.color = color
        self.rotate = rotate
        self.resize = resize
        self.warp = warp  # distortion
        self.xspace = xspace  # horizontal spacing
        self.yspace = yspace  # vertical spacing
        self.extra_text = extra_text  # other text processes

        self.init()

    def init(self):
        if callable(self._background):
            Captcha.background = property(self._background)
        else:
            self.background = self._background

    @cached_property
    def fonts(self):
        fontsdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fonts')

        for i, f in enumerate(self._fonts):
            if '/' not in f:
                self._fonts[i] = os.path.join(fontsdir, f'{f}.ttf')

        return self._fonts

    @cached_property
    def drawings(self):
        text_drawings = []
        if self.warp is not None:
            text_drawings.append(self.warp)
        if self.resize is not None:
            text_drawings.append(self.resize)
        if self.rotate is not None:
            text_drawings.append(self.rotate)
        if self.extra_text is not None:
            text_drawings.append(self.extra_text)

        _drawings = [
            text(
                fonts=self.fonts,
                font_sizes=self.font_sizes,
                drawings=text_drawings,
            ),
        ]
        return _drawings

    def generate(self, chars=None):
        if chars is None:
            chars = random.choices(self.char_set, k=self.length)

        image = Image.new("RGB", (self.width, self.height), self.background)
        for drawing in self.drawings:
            image = drawing(image, chars)
            assert image
        return image
