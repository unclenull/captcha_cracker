# pixels: numpy.array[int(0,255)], binary image pixels with black background
# threshhold: int, minimum number of surrounding black pixels to remove
def denoise_eight(pixels, threshold=3):
    rows, cols = pixels.shape
    arrToMark = [[None for i in range(cols)] for j in range(rows)]

    def check(row, col):
        total_bg = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                # assume it's black when out of range
                if row + x > rows - 1 or col + y > cols - 1 or row + x < 0 or col + y < 0 \
                        or not pixels[row + x, col + y]:
                    total_bg += 1
                if total_bg >= threshold:
                    arrToMark[row][col] = True
                    return

    for row in range(rows):
        for col in range(cols):
            point = pixels[row, col]
            if point:
                check(row, col)

    for row in range(rows):
        for col in range(cols):
            if arrToMark[row][col]:
                pixels[row, col] = 0  # set to background
    return pixels
