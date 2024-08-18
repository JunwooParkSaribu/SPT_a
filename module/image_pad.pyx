#cython: infer_types=True
from libc.stdlib cimport malloc, free
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef add_block_noise(imgs, extend):
    gap = extend//2
    row_indice = range(0, len(imgs[0]), gap)
    col_indice = range(0, len(imgs[0][0]), gap)
    for c in col_indice:
        crop_img = imgs[:, row_indice[1]:row_indice[1]+gap, c: min(len(imgs[0][0]) - gap, c+gap)]
        if crop_img.shape[2] == 0:
            break
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(gap, min(len(imgs[0][0]) - gap, c+gap) - c)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, row_indice[0]:row_indice[0] + gap, c: min(len(imgs[0][0]) - gap, c+gap)] = np.array(ret_img_stack)
    for r in row_indice:
        crop_img = imgs[:, r:min(len(imgs[0]) - gap, r + gap), len(imgs[0][0]) - 2*gap: len(imgs[0][0]) - gap]
        if crop_img.shape[1] == 0:
            break
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(min(len(imgs[0])-gap, r + gap) - r, gap)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, r:min(len(imgs[0])-gap, r + gap), len(imgs[0][0]) - gap: len(imgs[0][0])] = np.array(ret_img_stack)
    for c in col_indice[::-1]:
        crop_img = imgs[:, len(imgs[0]) - 2*gap:len(imgs[0]) - gap, c: min(len(imgs[0][0]), c+gap)]
        if crop_img.shape[2] == 0:
            continue
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(gap, min(len(imgs[0][0]), c+gap) - c)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, len(imgs[0]) - gap:len(imgs[0]), c: min(len(imgs[0][0]), c+gap)] = np.array(ret_img_stack)
    for r in row_indice:
        crop_img = imgs[:, r:min(len(imgs[0]), r + gap), col_indice[1]: col_indice[1] + gap]
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(min(len(imgs[0]), r + gap) - r, gap)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, r:min(len(imgs[0]), r + gap), col_indice[0]: col_indice[0] + gap] = np.array(ret_img_stack)

    for c in col_indice[1:-1]:
        csize = min(len(imgs[0][0]), c + 2 * gap) - c - gap
        crop_img = (imgs[:, row_indice[0]:row_indice[0]+gap, c-csize: c]
                    + imgs[:, row_indice[0]:row_indice[0]+gap, c+gap: c+gap+csize]) / 2
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(gap, min(len(imgs[0][0]), c+gap) - c)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, row_indice[0]:row_indice[0] + gap, c: min(len(imgs[0][0]), c+gap)] = np.array(ret_img_stack)
    for r in row_indice[1:-1]:
        rsize = min(len(imgs[0]), r + 2 * gap) - r - gap
        crop_img = (imgs[:, r - rsize: r, len(imgs[0][0]) - 2*gap: len(imgs[0][0]) - gap]
                    + imgs[:, r + gap: r+gap+rsize, len(imgs[0][0]) - 2*gap: len(imgs[0][0]) - gap]) / 2
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(min(len(imgs[0]), r + gap) - r, gap)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, r:min(len(imgs[0]), r + gap), len(imgs[0][0]) - gap: len(imgs[0][0])] = np.array(ret_img_stack)
    for c in col_indice[1:-1]:
        csize = min(len(imgs[0][0]), c + 2 * gap) - c - gap
        crop_img = (imgs[:, len(imgs[0]) - 2*gap:len(imgs[0]) - gap, c-csize: c]
                    + imgs[:, len(imgs[0]) - 2*gap:len(imgs[0]) - gap, c+gap: c+gap+csize]) / 2
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(gap, min(len(imgs[0][0]), c+gap) - c)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, len(imgs[0]) - gap:len(imgs[0]), c: min(len(imgs[0][0]), c+gap)] = np.array(ret_img_stack)
    for r in row_indice[1:-1]:
        rsize = min(len(imgs[0]), r + 2 * gap) - r - gap
        crop_img = (imgs[:, r - rsize: r, col_indice[0]: col_indice[0] + gap]
                    + imgs[:, r + gap: r+gap+rsize, col_indice[0]: col_indice[0] + gap]) / 2
        crop_means = np.mean(crop_img, axis=(1, 2))
        crop_stds = np.std(crop_img, axis=(1, 2))
        ret_img_stack = []
        for m, std in zip(crop_means, crop_stds):
            ret_img_stack.append(np.random.normal(loc=m, scale=std, size=(min(len(imgs[0]), r + gap) - r, gap)))
        ret_img_stack = np.array(ret_img_stack)
        imgs[:, r:min(len(imgs[0]), r + gap), col_indice[0]: col_indice[0] + gap] = np.array(ret_img_stack)
    return imgs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef double[:,:,::1] image_cropping(extended_imgs, int extend, int window_size0, int window_size1, int shift):
    cropped_imgs = []
    cdef int start_row, end_row, start_col, end_col, row_size, col_size, nb_imgs
    cdef int [::1] row_indice, col_indice
    cdef Py_ssize_t n, r, c, index
    nb_imgs = extended_imgs.shape[0]
    row_size = extended_imgs.shape[1]
    col_size = extended_imgs.shape[2]
    start_row = int(extend/2 - (window_size1-1)/2)
    end_row = row_size - window_size1 - start_row + 1
    start_col = int(extend/2 - (window_size0-1)/2)
    end_col = col_size - window_size0 - start_col + 1
    row_indice = np.arange(start_row, end_row, shift, dtype=np.intc)
    col_indice = np.arange(start_col, end_col, shift, dtype=np.intc)

    cropped_imgs = np.zeros([nb_imgs, len(row_indice) * len(col_indice), window_size0* window_size1], dtype=np.double)
    index = 0
    for r in row_indice:
        for c in col_indice:
            cropped_imgs[:, index] = extended_imgs[:, r:r + window_size1, c:c + window_size0].reshape(-1, window_size0*window_size1)
            index += 1
    return cropped_imgs
