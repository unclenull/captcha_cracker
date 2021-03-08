from PIL import Image
from itertools import groupby
import numpy as np
np.set_printoptions(threshold=9999999999)


def _get_start_col_from_min_hist(pixels):
    rows, cols = pixels.shape
    mid = cols // 2
    offset = cols // 6
    temp = 1-pixels[:rows//2, mid-offset:mid+offset]
    histogram_vert = list(np.sum(temp, 0))
    index = histogram_vert.index(min(histogram_vert))

    start_x_temp = mid - offset
    max_x_temp = temp.shape[1] - 1

    if (
        temp[0, index]  # current black
        and not pixels[0, start_x_temp + index-1]  # left black
        and index < max_x_temp and temp[0, index+1]  # right black
    ):  # move to the right most white if exists within temp
        new_index = index + 2
        while (new_index < max_x_temp):
            if not temp[0, new_index]:
                index = new_index
                break
            new_index += 1

    return start_x_temp + index


def _get_weight_index(index, cur_p):
    row, col = cur_p
    return [
            None, (row+1, col-1), (row+1, col),
            (row+1, col+1), (row, col+1), (row, col-1)
           ][index]


def _get_end_route(pixels, start_col):
    left_limit = 0
    rows, cols = pixels.shape
    right_limit = cols - 1
    end_route = []
    cur_p = (0, start_col)
    last_p = cur_p
    end_route.append(cur_p)
    # is_last_black = False

    while cur_p[0] < (rows-1):
        sum_w = 0
        max_w = 0
        cur_row, cur_col = cur_p

        for i in range(1, 6):
            if (i == 1 or i == 5) and cur_col == left_limit:
                continue
            if (i == 3 or i == 4) and cur_col == right_limit:
                continue
            cur_w = pixels[_get_weight_index(i, cur_p)]//255 * (6-i)
            sum_w += cur_w
            if max_w < cur_w:
                max_w = cur_w
        if sum_w == 0:  # all black
            max_w = 0
        elif sum_w == 15:
            max_w = 6

        if max_w == 1:
            next_col = cur_col - 1
            next_row = cur_row
        elif max_w == 2:
            next_col = cur_col + 1
            next_row = cur_row
        elif max_w == 3:
            next_col = cur_col + 1
            next_row = cur_row + 1
        elif max_w == 4:
            next_col = cur_col
            next_row = cur_row + 1
        elif max_w == 5:
            if cur_col < right_limit and pixels[cur_row+1, cur_col+1]:
                # No.3 is white
                # pick the nearest to black
                count = 2
                while cur_col-count > 0 and cur_col+count < right_limit:
                    if not pixels[cur_row+1, cur_col-count]:
                        next_col = cur_col - 1
                        next_row = cur_row + 1
                        break
                    elif not pixels[cur_row+1, cur_col+count]:
                        next_col = cur_col + 1
                        next_row = cur_row + 1
                        break
                    count += 1
                else:
                    next_col = cur_col
                    next_row = cur_row + 1
            else:
                next_col = cur_col - 1
                next_row = cur_row + 1
        elif max_w == 6:
            next_col = cur_col
            next_row = cur_row + 1
        elif max_w == 0:
            next_col = cur_col
            next_row = cur_row + 1
            # inertia only the first time
            # if is_last_black or last_p[1] == cur_col:
            #     next_col = cur_col
            #     next_row = cur_row + 1
            # elif last_p[1] < cur_col:
            #     next_col = cur_col + 1
            #     next_row = cur_row + 1
            # elif last_p[1] > cur_col:
            #     next_col = cur_col - 1
            #     next_row = cur_row + 1

            # is_last_black = True
        else:
            raise Exception("get end route error")

        if last_p[1] == next_col and last_p[0] == next_row:  # return back
            next_col = cur_col
            next_row = cur_row + 1
            # inertia:
            # if next_col < cur_col:
            #     next_col = cur_col + 1
            #     next_row = cur_row + 1
            # else:
            #     next_col = cur_col - 1
            #     next_row = cur_row + 1

        last_p = cur_p

        if next_col > right_limit:
            next_col = right_limit
            next_row = cur_row + 1
        elif next_col < left_limit:
            next_col = left_limit
            next_row = cur_row + 1
        cur_p = (next_row, next_col)
        end_route.append(cur_p)
    return end_route


# half split
# pixels: numpy.array[int(0, 255)], binary image pixels with white background
def split_dropfall(pixels):
    # Image.fromarray(pixels).show()
    # exit()
    rows, cols = pixels.shape
    start_col = _get_start_col_from_min_hist(pixels)

    end_route = _get_end_route(pixels, start_col)

    # img = Image.fromarray(pixels).convert('RGB')
    # img = np.array(img)
    # for i in end_route:
    #     img[tuple(i)] = [255, 0, 0]
    # Image.fromarray(img).show()
    # #Image.fromarray(img).save('mark.png')
    # # exit()

    end_route = [max(list(points)) for _, points in groupby(end_route, lambda x:x[0])]  # remove duplicate rows

    # find the widest black

    col_max = 0
    shortened_cols = []
    for dot in end_route:
        if pixels[dot]:  # shrink to black when white
            while dot[1] > col_max and pixels[dot]:
                dot = (dot[0], dot[1]-1)
        col_max = max(col_max, dot[1])
        shortened_cols.append(dot[1])

    img1 = np.full((rows, col_max+1), 255, dtype='uint8')
    for r in range(rows):
        c = shortened_cols[r]
        img1[r, 0:c+1] = pixels[r, 0:c+1]
    # Image.fromarray(img1.astype('bool')).save('a1.tif')
    # Image.fromarray(img1.astype('bool')).show()

    expanded_cols = []
    col_min = cols - 1
    for dot in end_route:
        if pixels[dot]:  # expand to black when white
            while dot[1] < col_min and pixels[dot]:
                dot = (dot[0], dot[1]+1)
        else:  # move to the next col
            dot = (dot[0], min(dot[1]+1, cols - 1))
        col_min = min(col_min, dot[1])
        expanded_cols.append(dot[1])

    width = cols - col_min
    img2 = np.full((rows, width), 255, dtype='uint8')
    for r in range(rows):
        c = expanded_cols[r]
        img2[r, c-col_min:] = pixels[r, c:]

    # Image.fromarray(img2).save('a2.tif')
    # Image.fromarray(img2).show()

    return [img1, img2]
