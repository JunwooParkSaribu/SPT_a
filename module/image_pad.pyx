#cython: infer_types=True
from libc.stdlib cimport malloc, free
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,::1] image_overlap(double[:,::1] img1, double[:,::1] img2, int div):
    cdef int row_max, col_max
    cdef Py_ssize_t r, c

    row_max = img1.shape[0]
    col_max = img1.shape[1]

    for r in range(row_max):
        for c in range(col_max):
            img1[r][c] = (img1[r][c] + img2[r][c]) / div
    return img1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double contig_image_mean(double[:,::1] img):
    cdef int count, row_max, col_max
    cdef double sum
    cdef Py_ssize_t r, c

    sum = 0.0
    count = 0
    row_max = img.shape[0]
    col_max = img.shape[1]

    for r in range(row_max):
        for c in range(col_max):
            sum += img[r][c]
            count += 1
    if count != 0:
        return sum / count
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double image_mean(double[:,:] img):
    cdef int count, row_max, col_max
    cdef double sum
    cdef Py_ssize_t r, c

    sum = 0.0
    count = 0
    row_max = img.shape[0]
    col_max = img.shape[1]

    for r in range(row_max):
        for c in range(col_max):
            sum += img[r][c]
            count += 1
    if count != 0:
        return sum / count
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double image_std(double[:,:] img):
    cdef int count, row_max, col_max
    cdef double var, mean
    cdef Py_ssize_t r, c

    row_max = img.shape[0]
    col_max = img.shape[1]
    var = 0.0
    count = 0
    mean = image_mean(img)

    for r in range(row_max):
        for c in range(col_max):
            var += (img[r][c] - mean) * (img[r][c] - mean)
            count += 1
    if count != 0:
        return (var / count) ** 0.5
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double contig_image_std(double[:,::1] img):
    cdef int count, row_max, col_max
    cdef double var, mean
    cdef Py_ssize_t r, c

    row_max = img.shape[0]
    col_max = img.shape[1]
    var = 0.0
    count = 0
    mean = image_mean(img)

    for r in range(row_max):
        for c in range(col_max):
            var += (img[r][c] - mean) * (img[r][c] - mean)
            count += 1
    if count != 0:
        return (var / count) ** 0.5
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] boundary_smoothing(double[:,::1] img, int[::1] row_indice, int[::1] col_indice):
    cdef int border_max
    cdef int repeat_n
    cdef int erase_space
    cdef Py_ssize_t border, r, c, i
    cdef int height, width, row_min, row_max, col_min, col_max, index, row, col, row_slice_0, row_slice_1, col_slice_0, col_slice_1

    border_max = 50
    erase_space = 2
    repeat_n = 2
    height = img.shape[0]
    width = img.shape[1]
    index = 0

    cdef double[:, ::1] img_view = img
    cdef int *center_xy = <int *>malloc(border_max * border_max * sizeof(int))

    for border in range(border_max):
        row_min = max(0, row_indice[0] + border)
        row_max = min(height - 1, row_indice[1] - border)
        col_min = max(0, col_indice[0] + border)
        col_max = min(width - 1, col_indice[1] - border)
        for col in range(col_min, col_max+1):
            center_xy[index] = row_min
            index += 1
            center_xy[index] = col
            index += 1
        for row in range(row_min, row_max+1):
            center_xy[index] = row
            index += 1
            center_xy[index] = col_max
            index += 1
        for col in range(col_max, col_min-1, -1):
            center_xy[index] = row_max
            index += 1
            center_xy[index] = col
            index += 1
        for row in range(row_max, row_min-1, -1):
            center_xy[index] = row
            index += 1
            center_xy[index] = col_min
            index += 1

    for _ in range(repeat_n):
        for i in range(index):
            if i % 2 == 0:
                r = center_xy[i]
            else:
                c = center_xy[i]
                row_slice_0 = max(0, r-erase_space)
                row_slice_1 = min(height, r+erase_space+1)
                col_slice_0 = max(0, c-erase_space)
                col_slice_1 = min(width, c+erase_space+1)
                img_view[r][c] = contig_image_mean(img_view[row_slice_0:row_slice_1, col_slice_0:col_slice_1])
    free(center_xy)
    return img


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,::1] add_block_noise(double[:,:,::1] imgs, int extend):
    cdef int gap, row_max, col_max, img_nb, c, r, i, csize, rsize, rand_index
    cdef double m, std
    cdef Py_ssize_t x, y
    cdef int[:] row_indice, col_indice
    gap = extend//2

    img_nb = imgs.shape[0]
    row_max = imgs[0].shape[0]
    col_max = imgs[0].shape[1]
    row_indice = np.arange(0, row_max, gap, dtype=np.intc)
    col_indice = np.arange(0, col_max, gap, dtype=np.intc)

    cdef double[:, :, ::1] img_view = imgs
    cdef double *crop_means = <double *>malloc(img_nb * sizeof(double))
    cdef double *crop_stds = <double *>malloc(img_nb * sizeof(double))

    for i in range(img_nb):
        for c in col_indice:
            crop_img = img_view[i][row_indice[1]: row_indice[1] + gap, c: min(col_max - gap, c + gap)]
            if crop_img.shape[1] == 0:
                break

            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)
            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=(gap * (min(col_max - gap, c+gap) - c)))
            for x in range(row_indice[0], row_indice[0] + gap):
                for y in range(c, min(col_max - gap, c+gap)):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for r in row_indice:
            crop_img = img_view[i][r:min(row_max - gap, r + gap), col_max - 2 * gap: col_max - gap]
            if crop_img.shape[0] == 0:
                break

            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)
            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=((min(row_max-gap, r + gap) - r) * gap))
            for x in range(r, min(row_max-gap, r + gap)):
                for y in range(col_max - gap, col_max):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for c in col_indice[::-1]:
            crop_img = img_view[i][row_max - 2 * gap:row_max - gap, c: min(col_max, c + gap)]
            if crop_img.shape[1] == 0:
                continue

            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)
            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=(gap * (min(col_max, c+gap) - c)))
            for x in range(row_max - gap, row_max):
                for y in range(c, min(col_max, c+gap)):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for r in row_indice:
            crop_img = img_view[i][r:min(row_max, r + gap), col_indice[1]: col_indice[1] + gap]

            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)
            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=((min(row_max, r + gap) - r) * gap))
            for x in range(r, min(row_max, r + gap)):
                for y in range(col_indice[0], col_indice[0] + gap):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for c in col_indice[1:-1]:
            csize = min(col_max, c + 2 * gap) - c - gap

            crop_img = image_overlap(img_view[i][row_indice[0]:row_indice[0]+gap, c-csize: c], img_view[i][row_indice[0]:row_indice[0]+gap, c+gap: c+gap+csize], 2)
            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)

            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=(gap * (min(col_max, c+gap) - c)))
            for x in range(row_indice[0], row_indice[0] + gap):
                for y in range(c, min(col_max, c+gap)):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for r in row_indice[1:-1]:
            rsize = min(row_max, r + 2 * gap) - r - gap

            crop_img = image_overlap(img_view[i][r - rsize: r, col_max - 2 * gap: col_max - gap], img_view[i][r + gap: r+gap+rsize, col_max - 2*gap: col_max - gap], 2)
            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)

            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=((min(row_max, r + gap) - r) * gap))
            for x in range(r, min(row_max, r + gap)):
                for y in range(col_max - gap, col_max):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for c in col_indice[1:-1]:
            csize = min(col_max, c + 2 * gap) - c - gap

            crop_img = image_overlap(img_view[i][row_max - 2*gap:row_max - gap, c-csize: c], img_view[i][row_max - 2*gap:row_max - gap, c+gap: c+gap+csize], 2)
            m = contig_image_mean(crop_img)
            std = contig_image_std(crop_img)

            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=(gap * (min(col_max, c+gap) - c)))
            for x in range(row_max - gap, row_max):
                for y in range(c, min(col_max, c+gap)):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

        for r in row_indice[1:-1]:
            rsize = min(row_max, r + 2 * gap) - r - gap

            crop_img = image_overlap(img_view[i][r - rsize: r, col_indice[0]: col_indice[0] + gap], img_view[i][r + gap: r+gap+rsize, col_indice[0]: col_indice[0] + gap], 2)
            m = contig_image_mean(crop_img)
            std = image_std(crop_img)

            rand_index = 0
            rands = np.random.normal(loc=m, scale=std, size=((min(row_max, r + gap) - r) * gap))
            for x in range(r, min(row_max, r + gap)):
                for y in range(col_indice[0], col_indice[0] + gap):
                    img_view[i][x][y] = rands[rand_index]
                    rand_index += 1

    free(crop_means)
    free(crop_stds)
    return imgs


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,::1] likelihood(crop_imgs, double[:, ::1] gauss_grid, double[::1] bg_squared_sums, bg_means, int window_size1, int window_size2):
    cdef int surface_window, index
    cdef double g_squared_sum, g_mean
    cdef Py_ssize_t i, j
    g_bar = np.zeros([window_size1 * window_size2], dtype=np.double)
    cdef double[::1] g_bar_view = g_bar 
    surface_window = window_size1 * window_size2
    g_mean = contig_image_mean(gauss_grid)
    g_squared_sum = 0

    index = 0
    for i in range(window_size1):
        for j in range(window_size2):
            g_bar_view[index] = gauss_grid[i][j] - g_mean
            index += 1

    for i in range(window_size1 * window_size2):
        g_squared_sum += g_bar_view[i] * g_bar_view[i]

    i_hat = (crop_imgs - bg_means.reshape(crop_imgs.shape[0], 1, 1))
    i_hat = i_hat @ g_bar / g_squared_sum
    i_hat = np.maximum(np.zeros(i_hat.shape), i_hat)
    L = ((surface_window / 2.) * np.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    return L.reshape(crop_imgs.shape[0], crop_imgs.shape[1], 1)
