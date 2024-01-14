import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numba import njit
from numba.typed import List as nbList
from ImageModule import read_tif
from timeit import default_timer as timer


#images = read_tif('RealData/20220217_aa4_cel8_no_ir.tif')
#images = read_tif('SimulData/receptor_7_low.tif')
#images = read_tif('tif_trxyt/receptor_7_mid.tif')
images = read_tif('tif_trxyt/U2OS-H2B-Halo_0.25%50ms_field1.tif')
#images = read_tif("C:/Users/jwoo/Desktop/U2OS-H2B-Halo_0.25%50ms_field1.tif")
print(images[0].shape)


THRESHOLDS = [.2, .2, .3, .3]
P0 = [2., 2., 0., 0., 0.1]
GAUSS_SEIDEL_DECOMP = 10
WINDOW_SIZES = [(5, 5), (7, 7), (11, 11), (15, 15)]
RADIUS = [1, 3, 5, 7]
DIV_Q = 2
images = images[:10]


def region_max_filter(maps, window_size, threshold):
    indices = []
    r_start_index = int((window_size[1]-1) / 2)
    col_start_index = int((window_size[0]-1) / 2)
    args_map = maps > threshold
    maps = maps * args_map
    img_n, row, col = np.where(args_map == True)
    for n, r, c in zip(img_n, row, col):
        if maps[n][r][c] == np.max(maps[n, max(0, r-r_start_index):min(maps.shape[1]+1, r+r_start_index+1),
                                   max(0, c-col_start_index):min(maps.shape[2]+1, c+col_start_index+1)]):
            indices.append([n, r, c])
    return np.array(indices)


@njit
def subtract_pdf(ext_imgs, pdfs, indices, window_size, bg_means, extend):
    for pdf, (n, r, c) in zip(pdfs, indices):
        pdf = np.ascontiguousarray(pdf).reshape(window_size)
        row_indice = np.array([r - int((window_size[1]-1)/2), r + int((window_size[1]-1)/2)]) + int(extend/2)
        col_indice = np.array([c - int((window_size[0]-1)/2), c + int((window_size[0]-1)/2)]) + int(extend/2)
        ext_imgs[n][row_indice[0]:row_indice[1]+1, col_indice[0]:col_indice[1]+1] -= pdf
        ext_imgs[n] = boundary_smoothing(ext_imgs[n], row_indice, col_indice)
    return np.maximum(ext_imgs, (np.ones(ext_imgs.shape).T * bg_means).T)


@njit
def boundary_smoothing(img, row_indice, col_indice):
    center_xy = []
    row_min = max(0, row_indice[0]-1)
    row_max = min(img.shape[0]-1, row_indice[1]+1)
    col_min = max(0, col_indice[0]-1)
    col_max = min(img.shape[1]-1, col_indice[1]+1)
    for col in range(col_min, col_max+1):
        center_xy.append([row_min, col])
    for row in range(row_min, row_max+1):
        center_xy.append([row, col_max])
    for col in range(col_max, col_min-1, -1):
        center_xy.append([row_max, col])
    for row in range(row_max, row_min-1, -1):
        center_xy.append([row, col_min])
    for r, c in center_xy:
        img[r][c] = np.mean(img[max(0, r-1):min(img.shape[0], r+2), max(0, c-1):min(img.shape[1], c+2)])
    return img


def gauss_psf(window_sizes, radiuss):
    gauss_grid_window = []
    for window_size, radius in zip(window_sizes, radiuss):
        x_subpixel = np.arange(window_size[0]) + .5
        y_subpixel = np.arange(window_size[1]) + .5
        center_x = window_size[0] / 2.
        center_y = window_size[1] / 2.
        base_vals = np.ones((window_size[1], window_size[0], 2)) * np.array([center_x, center_y])
        gauss_psf_vals = np.stack(np.meshgrid(x_subpixel, y_subpixel), -1)
        gauss_psf_vals = np.exp(-(np.sum((gauss_psf_vals - base_vals)**2, axis=2))
                                /(2*(radius**2))) / (np.sqrt(np.pi) * radius)
        gauss_grid_window.append(np.array(gauss_psf_vals))
    return gauss_grid_window


def likelihood(crop_imgs, gauss_grid, bg_squared_sums, bg_means, window_size):
    crop_imgs = np.ascontiguousarray(crop_imgs)
    surface_window = window_size[0] * window_size[1]
    g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).reshape(window_size[0]*window_size[1], 1)
    g_squared_sum = np.sum(g_bar ** 2)
    i_hat = crop_imgs @ g_bar / g_squared_sum
    L = ((surface_window / 2.) * np.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    return L


def localization(imgs: np.ndarray, bgs, gauss_grids):
    shift = 1
    coords = [[] for _ in range(imgs.shape[0])]
    reg_pdfs = [[] for _ in range(imgs.shape[0])]
    bg_means = bgs[0][:, 0]
    extend = WINDOW_SIZES[-1][0] - 1 if WINDOW_SIZES[-1][0] % 2 == 1 else WINDOW_SIZES[-1][0]
    extended_imgs = np.zeros((imgs.shape[0], imgs.shape[1] + extend, imgs.shape[2] + extend)) + bg_means.reshape(-1, 1, 1)
    extended_imgs[:, int(extend/2):int(extend/2) + imgs.shape[1], int(extend/2):int(extend/2) + imgs.shape[2]] += (
            imgs - bg_means.reshape(-1, 1, 1))

    for step, (bg, gauss_grid, window_size, radius, threshold) in (
            enumerate(zip(bgs, gauss_grids, WINDOW_SIZES, RADIUS, THRESHOLDS))):
        print(f'{step} : {imgs.shape}')
        regress_imgs = []
        bg_regress = []
        start = timer()
        crop_imgs = image_cropping(extended_imgs, extend, window_size, shift=shift)
        crop_imgs = np.array(crop_imgs).reshape(imgs.shape[1] * imgs.shape[2], imgs.shape[0],
                                                window_size[0] * window_size[1])
        print(f'cropping : {timer() - start}')
        crop_imgs = np.moveaxis(crop_imgs, 0, 1)
        bg_squared_sums = window_size[0] * window_size[1] * bg_means**2
        start = timer()
        c = likelihood(crop_imgs, gauss_grid, bg_squared_sums, bg_means, window_size)
        print(f'likelihood : {timer() - start}')

        h_maps = c.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2])
        #h_map = h_map * img / np.max(h_map * img)
        indices = region_max_filter(h_maps, window_size, threshold)
        if len(indices) != 0:
            start = timer()
            for n, r, c in indices:
                regress_imgs.append(crop_imgs[n][imgs.shape[2] * r + c])
                bg_regress.append(bg[n])
            pdfs, xs, ys = image_regression(regress_imgs, bg_regress, window_size)
            print(f'regression : {timer() - start}')
            start = timer()
            for (n, r, c), dx, dy, pdf in zip(indices, xs, ys, pdfs):
                coords[n].append([r + dx, c + dy])
                reg_pdfs[n].append(pdf)
            new_imgs = subtract_pdf(extended_imgs, pdfs, indices, window_size, bg_means, extend)
            print(f'subtraction : {timer() - start}')
            extended_imgs = new_imgs
    return coords, reg_pdfs


@njit
def image_cropping(extended_imgs: np.ndarray, extend, window_size, shift):
    cropped_imgs = []
    start_row = int(extend/2 - (window_size[1]-1)/2)
    end_row = extended_imgs.shape[1] - window_size[1] - start_row
    start_col = int(extend/2 - (window_size[0]-1)/2)
    end_col = extended_imgs.shape[2]-window_size[0] - start_col
    for r in range(start_row, end_row+1, shift):
        for c in range(start_col, end_col+1, shift):
            cropped_imgs.append(extended_imgs[:, r:r + window_size[1], c:c + window_size[0]])
    return cropped_imgs


def cov_matrix(grid, qt):
    observations = qt * grid
    nbs = np.sum(qt, axis=1)
    obv_mean = (np.sum(observations, axis=1) / nbs).reshape(observations.shape[0], 1, -1)
    obv_mean = np.ones(observations.shape) * obv_mean
    a = np.sqrt(qt) * grid - (obv_mean * np.sqrt(qt))
    estimated_cov = (a.transpose(0, 2, 1) @ a) / list((nbs-1).reshape(nbs.shape[0], 1, 1))
    return estimated_cov


def quantification(imgs, window_size, amp):
    qt_imgs = (imgs * (10**amp)).astype(np.uint8).reshape(imgs.shape[0], -1, 1)
    x = np.arange(-(window_size[0]-1)/2, (window_size[0]+1)/2)
    y = np.arange(-(window_size[1]-1)/2, (window_size[1]+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size[0] * window_size[1], 2)
    return qt_imgs, grid


def bi_variate_normal_pdf(xy, cov, mu, normalization=True):
    a = np.ones((cov.shape[0], xy.shape[0], xy.shape[1])) * (xy - mu)
    if normalization:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2))
                / (2 * np.pi * np.sqrt(np.linalg.det(cov).reshape(-1, 1))))
    else:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2)))


def background(imgs, window_sizes):
    bgs = []
    bg_instensity = stats.mode(
        (imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) * 100).astype(np.uint8), axis=1, keepdims=False)[0] / 100
    for window_size in window_sizes:
        bg = np.ones((bg_instensity.shape[0], window_size[0] * window_size[1]))
        bg *= bg_instensity.reshape(-1, 1)
        bgs.append(bg)
    return bgs


def kl_divergence2(cov1, cov2):
    a = np.linalg.inv(cov2) * cov1
    return 1./2 * (a[:, 0, 0] + a[:, 1, 1] - 2 + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))


def kl_divergence(base, compares):
    a = np.mean((compares * np.log(compares / base)), axis=1)
    return a


def intensity_reg2(img, grid, cov):
    a = np.ones((cov.shape[0], grid.shape[0], grid.shape[1])) * grid
    b = np.exp((1. / 2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2))
    cov_dets = cov[:, 0, 0] + cov[:, 1, 1]
    Y = np.sum(img, axis=1)
    return (2. * np.pi * Y * np.sqrt(cov_dets) * np.sum(b, axis=1)).reshape(-1, 1) / grid.shape[0]


def intensity_reg(imgs, pdfs, center_i):
    bg_i = np.min(imgs, axis=1).reshape(-1, 1)
    intensity = ((imgs - bg_i) / pdfs)[:, center_i].reshape(-1, 1)
    return intensity, bg_i


def image_regression(imgs, bgs, window_size, amp=3):
    imgs = np.array(imgs)
    bgs = np.array(bgs)
    qt_imgs, grid = quantification(imgs, window_size, amp)
    coefs = guo_algorithm(imgs, bgs, p0=P0, window_size=window_size)
    variables = np.array(unpack_coefs(coefs)).T
    cov_mat = np.array([variables[:, 0], [0]*variables.shape[0], [0]*variables.shape[0], variables[:, 1]]).T.reshape(variables.shape[0], 2, 2)
    pdfs = bi_variate_normal_pdf(grid, cov_mat, mu=np.array([0, 0]), normalization=False)
    pdfs = variables[:, 4].reshape(-1, 1) * pdfs + bgs
    return pdfs, variables[:, 2], variables[:, 3]


@njit
def unpack_coefs(coefs):
    x_var = -1./(2 * coefs[:, 0])
    y_var = -1./(2 * coefs[:, 2])
    x0 = coefs[:, 1] * x_var
    y0 = coefs[:, 3] * y_var
    amp = np.exp(coefs[:, 4] + ((x0**2)/(2 * x_var)) + ((y0**2)/(2 * y_var)))
    return [x_var, y_var, x0, y0, amp]


@njit
def pack_vars(vars, len_img):
    coef0 = -1./(2 * vars[0])
    coef2 = -1. / (2 * vars[1])
    coef1 = vars[2] / vars[0]
    coef3 = vars[3] / vars[1]
    coef4 = np.log(vars[4]) - ((vars[2]**2)/(2 * vars[0])) - ((vars[3]**2)/(2 * vars[1]))
    return [[coef0, coef1, coef2, coef3, coef4] for _ in range(len_img)]


@njit
def matrix_decomp(matrix, q):
    ret_mat = []
    for x in range(0, len(matrix), q):
        ret_mat.append(matrix[x: min(x+q, len(matrix))])
    return ret_mat


def guo_algorithm(imgs, bgs, p0=None, window_size=(7, 7)):
    nb_imgs = imgs.shape[0]
    if p0 is None:
        p0 = [2., 2., 0., 0., 0.1]
    coef_vals = np.array(pack_vars(nbList(p0), nb_imgs))
    imgs = imgs.reshape(imgs.shape[0], window_size[0], window_size[1])
    ## background for each crop image needed rather than background intensity for whole image.
    imgs = np.maximum(np.zeros(imgs.shape), imgs - bgs.reshape(-1, window_size[0], window_size[1])) + 1e-2
    yk_2 = imgs.astype(np.float64)
    x_grid = (np.array([list(np.arange(-int(window_size[0]/2), int((window_size[0]/2) + 1), 1))] * window_size[1])
              .reshape(-1, window_size[0], window_size[1]))
    y_grid = (np.array([[y] * window_size[0] for y in range(int(window_size[1]/2), -int((window_size[1]/2) + 1), -1)])
              .reshape(-1, window_size[0], window_size[1]))
    for k in range(0, 5):
        if k != 0:
            yk_2 = np.exp(coef_vals[:, 0].reshape(-1, 1, 1) * x_grid**2 + coef_vals[:, 1].reshape(-1, 1, 1) * x_grid +
                          coef_vals[:, 2].reshape(-1, 1, 1) * y_grid**2 + coef_vals[:, 3].reshape(-1, 1, 1) * y_grid +
                          coef_vals[:, 4].reshape(-1, 1, 1))
        yk_2 *= yk_2
        coef1 = yk_2 * x_grid**4
        coef2 = yk_2 * x_grid**3
        coef3 = yk_2 * y_grid**2 * x_grid**2
        coef4 = yk_2 * y_grid * x_grid**2
        coef5 = yk_2 * x_grid**2
        coef6 = yk_2 * y_grid**2 * x_grid
        coef7 = yk_2 * y_grid * x_grid
        coef8 = yk_2 * x_grid
        coef9 = yk_2 * y_grid**4
        coef10 = yk_2 * y_grid**3
        coef11 = yk_2 * y_grid**2
        coef12 = yk_2 * y_grid
        coef_matrix = np.sum(
            np.array(
                [[coef1, coef2, coef3, coef4, coef5],
                 [coef2, coef5, coef6, coef7, coef8],
                 [coef3, coef6, coef9, coef10, coef11],
                 [coef4, coef7, coef10, coef11, coef12],
                 [coef5, coef8, coef11, coef12, yk_2]]
            ), axis=(3, 4)).transpose(2, 0, 1)
        ans1 = x_grid ** 2 * yk_2 * np.log(imgs)
        ans2 = x_grid * yk_2 * np.log(imgs)
        ans3 = y_grid ** 2 * yk_2 * np.log(imgs)
        ans4 = y_grid * yk_2 * np.log(imgs)
        ans5 = yk_2 * np.log(imgs)
        ans_matrix = np.sum(
            np.array(
                [[ans1], [ans2], [ans3], [ans4], [ans5]]
            ), axis=(3, 4), dtype=np.float64).transpose(2, 0, 1)
        coef_matrix = matrix_decomp(coef_matrix, GAUSS_SEIDEL_DECOMP)
        ans_matrix = matrix_decomp(ans_matrix, GAUSS_SEIDEL_DECOMP)
        decomp_coef_vals = matrix_decomp(coef_vals, GAUSS_SEIDEL_DECOMP)
        x_matrix = []
        for (a_mats, b_mats, coef_val) in zip(coef_matrix, ans_matrix, decomp_coef_vals):
            a_mat = np.zeros((a_mats.shape[0] * a_mats.shape[1], a_mats.shape[0] * a_mats.shape[2]))
            for x, vals in zip(range(0, a_mat.shape[0], 5), a_mats):
                a_mat[x:x+5, x:x+5] = vals
            b_mat = b_mats.flatten().reshape(-1, 1)
            #x_matrix.extend(np.linalg.lstsq(a_mat, b_mat, rcond=None)[0])
            x_matrix.extend(gauss_seidel(a_mat, b_mat, p0=coef_val.ravel(), iter=200))
        x_matrix = np.array(x_matrix).reshape(-1, 5)
        if np.allclose(coef_vals, x_matrix, rtol=1e-7):
            break
        coef_vals = x_matrix
    return coef_vals


@njit
def gauss_seidel(a, b, p0, iter=1000, tol=1e-8):
    x = p0.ravel()
    for it_count in range(1, iter):
        x_new = np.zeros(x.shape, dtype=np.float64)
        for i in range(a.shape[0]):
            s1 = np.dot(a[i, :i], x_new[:i])
            s2 = np.dot(a[i, i + 1:], x[i + 1:])
            x_new[i] = ((b[i] - s1 - s2) / a[i, i])[0]
        if np.allclose(x, x_new, rtol=tol):
            break
        x = x_new
    return x


ans = []
reg_probas = []
gauss_grids = gauss_psf(WINDOW_SIZES, RADIUS)
for div_q in range(0, len(images), DIV_Q):
    print(f'{div_q} epoch')
    bgs = background(images[div_q:div_q+DIV_Q], window_sizes=WINDOW_SIZES)
    a1, a2 = localization(images[div_q:div_q+DIV_Q], bgs, gauss_grids)
    ans.extend(a1)
    reg_probas.extend(a2)

for img_n, (coord_list, reg_pdf) in enumerate(zip(ans, reg_probas)):
    print([len(coord_list), len(coord_list[0])], [len(reg_pdf), len(reg_pdf[0])])
    circle_img = (np.array([images[img_n].copy(), images[img_n].copy(), images[img_n].copy()])).transpose(1, 2, 0)
    for coord in coord_list:
        circle_img[int(coord[0])][int(coord[1])][0] = 255
        circle_img[int(coord[0])][int(coord[1])][1] = 0
        circle_img[int(coord[0])][int(coord[1])][2] = 0
    plt.figure(figsize=(14, 7))
    plt.imshow(np.hstack(((np.array([images[img_n].copy(), images[img_n].copy(), images[img_n].copy()])).transpose(1, 2, 0), circle_img)))
    plt.show()
