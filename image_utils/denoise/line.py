# pixels: numpy.array(int(0, 255)), binary image pixels
# threshhold: int, minimum length of the line to remove

import numpy as np
import cv2

WHITE = 255
BLACK = 0
VERTICAL = 0
HORIZONTAL = 1
THRESHOLD = 20


def _denoise_line(pixels, colorChar, mainAxis=HORIZONTAL, threshold=THRESHOLD):
    shape = pixels.shape
    # Temp = np.ones(shape)
    # from PIL import Image
    # Image.fromarray(pixels).show()

    ColorBg = ~colorChar & 255
    AXISES_MAIN = shape[mainAxis]
    AXISES_CROSS = shape[1-mainAxis]
    AxisesCrossHit = [None]*AXISES_MAIN
    AxisMainStart = 0
    Visited = np.full(shape, False)

    def getIndex(axisMain, axisCross):
        return (axisCross, axisMain) if mainAxis == HORIZONTAL \
                else (axisMain, axisCross)

    def genLine(axisMain):
        if axisMain < 1:
            print('Invalid index: ', axisMain)
            exit()

        hasMore = False
        hit = False

        if axisMain < AXISES_MAIN:
            # straight forward
            axisCross = AxisesCrossHit[axisMain-1]

            if pixels[getIndex(axisMain, axisCross)] == colorChar:
                AxisesCrossHit[axisMain] = axisCross
                hasMore = True
                hit = genLine(axisMain+1)
                if hit:
                    removeDot(axisMain)
            # ignore branches, they are chars at most cases
            else:
                if (axisCross > 0 and
                        pixels[getIndex(axisMain, axisCross-1)] == colorChar):
                    AxisesCrossHit[axisMain] = axisCross-1
                    hasMore = True
                    hit = genLine(axisMain+1)
                    if hit:
                        removeDot(axisMain)
                if (axisCross < AXISES_CROSS-1 and
                        pixels[getIndex(axisMain, axisCross+1)] == colorChar):
                    AxisesCrossHit[axisMain] = axisCross+1
                    hasMore = True
                    hit = genLine(axisMain+1)
                    if hit:
                        removeDot(axisMain)

        if not hasMore:
            # if axisMain - AxisMainStart > threshold:
                # print(AxisMainStart, axisMain)
            return (axisMain - AxisMainStart > threshold)
        Visited[getIndex(axisMain, axisCross)] = True
        return hit

    def removeDot(axisMain):
        axisCross = AxisesCrossHit[axisMain]
        foundOneSide = False
        # Temp[getIndex(axisMain, axisCross)] = 0

        for c in [-1, 1]:
            if axisCross + c >= AXISES_CROSS or axisCross + c < 0:
                break

            for m in range(-1, 2):
                if axisMain + m >= AXISES_MAIN or axisMain + m < 0:
                    continue

                if pixels[getIndex(axisMain + m, axisCross + c)] == colorChar:
                    if foundOneSide:
                        # preserve if cross line exists
                        return
                    else:  # switch to another side
                        foundOneSide = True
                        break

        pixels[getIndex(axisMain, axisCross)] = ColorBg

    # start
    for axisMain in range(AXISES_MAIN):
        AxisMainStart = axisMain
        for axisCross in range(AXISES_CROSS):
            if (pixels[getIndex(axisMain, axisCross)] == colorChar
                    and not Visited[axisCross, axisMain]):
                AxisesCrossHit[axisMain] = axisCross
                hit = genLine(axisMain+1)
                if hit:
                    removeDot(axisMain)
    # from PIL import Image
    # Image.fromarray(Temp.astype('bool')).show()
    return pixels


def denoise_line_black_hor(pixels, threshold=THRESHOLD):
    return _denoise_line(
            pixels, threshold=threshold,
            colorChar=BLACK, mainAxis=HORIZONTAL)


def denoise_line_black_ver(pixels, threshold=THRESHOLD):
    return _denoise_line(
            pixels, threshold=threshold,
            colorChar=BLACK, mainAxis=VERTICAL)


def denoise_line_white_hor(pixels, threshold=THRESHOLD):
    return _denoise_line(
            pixels, threshold=threshold,
            colorChar=WHITE, mainAxis=HORIZONTAL)


def denoise_line_white_ver(pixels, threshold=THRESHOLD):
    return _denoise_line(
            pixels, threshold=threshold,
            colorChar=WHITE, mainAxis=VERTICAL)
