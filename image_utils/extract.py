from PIL import Image
import traceback
import numpy as np
import cv2
from math import ceil
from .split import split_dropfall


# pixels: numpy.array[int(0,255)], binary image pixels
# count: int, number of chars to extract
# splitRecursive: bool
def extract(pixels, count, splitRecursive=False):
    # Image.fromarray(pixels).show()
    rows, cols = pixels.shape
    maskFull = np.zeros(np.array(pixels.shape) + np.array([2, 2]), dtype='uint8')
    widthMax = cols // count
    widthMin = widthMax // 2
    fills = []

    def split(img):
        ls = []
        img1, img2 = split_dropfall(img)
        if img1.shape[1] < widthMin or img2.shape[1] < widthMin:
            raise Exception('failed to split')

        img1 = extract(img1, 1)[0]
        img2 = extract(img2, 1)[0]

        if img1.shape[1] > widthMax:
            if splitRecursive:
                ls += split_dropfall(img1)
            else:
                raise Exception('failed to split')
        else:
            ls.append(img1)
        if img2.shape[1] > widthMax:
            if splitRecursive:
                ls += split_dropfall(img2)
            else:
                raise Exception('failed to split')
        else:
            ls.append(img2)

        return ls

    for r in range(rows):
        for c in range(cols):
            if not pixels[r, c] and not maskFull[r+1, c+1]:
                mask = np.zeros(np.array(pixels.shape) + np.array([2, 2]), dtype='uint8')
                rect = cv2.floodFill(
                    pixels, mask, (c, r), None,
                    flags=4 | 255 << 8 | cv2.FLOODFILL_MASK_ONLY)[3]
                x, y, w, h = rect
                # Image.fromarray(mask).show()
                fills.append((rect, np.invert(mask[y+1:y+1+h, x+1:x+1+w])))  # turn to white background
                maskFull[mask == 255] = True

    out = []
    while len(fills) and len(out) < count:
        m = max(fills, key=lambda e: e[0][2])
        fills.remove(m)
        [x, y, w, h], img = m
        # Image.fromarray(img).show()
        if w > widthMax:
            try:
                print('splitting...')
                ls = split(img)
            except Exception:
                traceback.print_exc()
                return None
            for img in ls:
                out.append((x, img))
        else:
            out.append((x, img))
    out.sort(key=lambda e: e[0])

    if len(out) < count:
        return None
    return list(map(lambda e: e[1], out))


# pixels: numpy.array[int(0,255)], binary image pixels
# count: int, number of chars to extract
# length: target char square length, if None returned unwrapped
# splitRecursive: bool
# noRotate: bool, don't rotate char upright
# return numpy.array[int(0,255)]
def extract_wrapped(pixels, count, length=16, splitRecursive=False, noRotate=False):
    if length is None:
        length = pixels.shape[0]

    def createChar(img):
        h, w = img.shape
        if not noRotate:
            contour = cv2.findContours(
                    np.invert(img),  # work on non-zero uint8 foreground
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )[0][0]
            rect = cv2.minAreaRect(contour)
            center, size, angle = rect

            if angle != -90:
                w, h = size
                w = ceil(w)
                h = ceil(h)
                if w > h:
                    n = w
                    w = h
                    h = n
                    angle += 90

                box = cv2.boxPoints((center, (w, h), angle))

                # temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # cv2.drawContours(temp, [box], 0, (0, 0, 255), 2)
                # Image.fromarray(temp).show()

                src_pts = box.astype("float32")
                dst_pts = np.array([
                        [0, h-1],
                        [0, 0],
                        [w-1, 0],
                        [w-1, h-1],
                    ],
                    dtype="float32")

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_AREA, borderValue=255)
                # Image.fromarray(img).show()

        char = np.full([length, length], 255, dtype='uint8')  # square
        nw = nh = length
        if h > w:
            nw = w * length // h
        else:
            nh = h * length // w
        top = (length-nh)//2
        left = (length-nw)//2
        # Image.fromarray(img).show()
        char[top:top+nh, left:left+nw] = cv2.resize(
            img,
            (nw, nh),
            interpolation=cv2.INTER_NEAREST
        )
        char = cv2.threshold(char, None, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Image.fromarray(char).show()
        return char

    # Image.fromarray(pixels).show()
    imgs = extract(pixels, count)
    if imgs:
        for ix, img in enumerate(imgs):
            # Image.fromarray(img).show()
            imgs[ix] = createChar(img)

    return imgs
