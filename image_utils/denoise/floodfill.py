import numpy as np
import cv2


# pixels: numpy.array[int(0,255)], binary image pixels with black background
# threshhold: int, maximum number of adjacent front-color pixels to remove
def denoise_floodfill(pixels, threshold=5):
    shape = pixels.shape
    rows, cols = shape
    maskAll = np.zeros(np.array(shape) + np.array([2, 2]), dtype='uint8')

    for r in range(rows):
        for c in range(cols):
            if pixels[r, c] and not maskAll[r+1, c+1]:
                mask = np.zeros(np.array(shape) + np.array([2, 2]), dtype='uint8')
                rect = cv2.floodFill(
                        pixels, mask, (c, r), None,
                        flags=4 | 1 << 8 | cv2.FLOODFILL_MASK_ONLY)[3]
                x, y, w, h = rect
                imgCut = mask[y+1:y+1+h, x+1:x+1+w]
                if np.count_nonzero(imgCut) <= threshold:
                    pixels[mask[1:rows+1, 1:cols+1].astype('bool')] = 0  # set to background
                else:
                    maskAll[mask.astype('bool')] = 1
    return pixels
