# pixels: numpy.array[int(0,255)], binary image pixels with white background
# threshhold: int, minimum number of surrounding white pixels to remove
def denoise_eight(pixels, threshold=3):
    WHITE = 255
    threshold *= WHITE
    rows, cols = pixels.shape
    arrToMark = [[None for i in range(cols)] for j in range(rows)]

    def check(row, col):
        sum = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                # assume it's white when out of range
                if row + x > rows - 1 or col + y > cols - 1 or \
                        row + x < 0 or col + y < 0:
                    sum += WHITE
                else:
                    sum += pixels[row + x, col + y]
                if sum >= threshold:
                    arrToMark[row][col] = True
                    return

    for row in range(rows):
        for col in range(cols):
            point = pixels[row, col]
            if point == 0:
                check(row, col)

    for row in range(rows):
        for col in range(cols):
            if arrToMark[row][col]:
                pixels[row, col] = WHITE
    return pixels
