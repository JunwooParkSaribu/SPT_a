import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numba import njit

from ImageModule import read_tif

#images = read_tif('RealData/20220217_aa4_cel8_no_ir.tif')
#images = read_tif('SimulData/receptor_7_low.tif')
images = read_tif('tif_trxyt/receptor_7_low.tif')
print(images[0].shape)


def gauss_psf(cropped_img, window_size):
    cropped_img = cropped_img.reshape(window_size)
    radius = 1.1
    x_subpixel = np.arange(cropped_img.shape[1]) + .5
    y_subpixel = np.arange(cropped_img.shape[0]) + .5
    center_x = cropped_img.shape[1] / 2.
    center_y = cropped_img.shape[0] / 2.
    base_vals = np.ones((cropped_img.shape[0], cropped_img.shape[1], 2)) * np.array([center_x, center_y])
    gauss_psf_vals = np.stack(np.meshgrid(x_subpixel, y_subpixel), -1)
    gauss_psf_vals = np.exp(-(np.sum((gauss_psf_vals - base_vals)**2, axis=2))/(2*(radius**2))) / (np.sqrt(np.pi) * radius)
    return gauss_psf_vals


def background_likelihood(img: np.ndarray, window_size=(7, 7)):
    shift = 1
    surface_window = window_size[0] * window_size[1]
    crop_imgs, xy_coords = image_cropping(img, window_size, shift=shift)
    bg_means = np.sum(crop_imgs, axis=1) / surface_window
    bg_squared_sum = np.sum(crop_imgs ** 2, axis=1)
    gauss_grid = gauss_psf(crop_imgs[0], window_size)
    g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).flatten()
    g_squared_sum = np.sum(g_bar ** 2)
    i_hat = crop_imgs @ g_bar / g_squared_sum
    c = (surface_window / 2.) * np.log(1 - (i_hat**2 * g_squared_sum) / (bg_squared_sum - (surface_window * bg_means)))
    for i, (val, xy) in enumerate(zip(c, xy_coords)):
        if val > 0.5:
            print(xy, val)
            plt.figure()
            plt.imshow(crop_imgs[i].reshape(window_size))
            plt.show()
    print(c.shape)
    pass


def image_cropping(img: np.ndarray, window_size=(7, 7), shift=1, bg_intensity=None):
    if bg_intensity is None:
        bg_intensity = np.mean(img)
    extend = window_size[0] + 1 if window_size[0] % 2 == 1 else window_size[0]
    extended_img = np.zeros((img.shape[0] + extend, img.shape[1] + extend)) + bg_intensity
    extended_img[int(extend/2):int(extend/2) + img.shape[0], int(extend/2):int(extend/2) + img.shape[1]] += (
            img - bg_intensity)

    img_height = len(img)
    img_width = len(img[0])
    cropped_imgs = []
    cropped_xy = []
    for j in range(0, img_height-window_size[1]+1, shift):
        for i in range(0, img_width-window_size[0]+1, shift):
            cropped_imgs.append(img[j:j + window_size[1], i:i + window_size[0]].flatten())
            cropped_xy.append([i - int(extend/2), j - int(extend/2)])
    return np.array(cropped_imgs), np.array(cropped_xy)


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


def bi_variate_normal_pdf(xy, cov, mu=None):
    if mu is None:
        mu = np.array([0, 0])
    else:
        mu = np.array(mu)
    a = np.ones((cov.shape[0], xy.shape[0], xy.shape[1])) * (xy - mu)
    return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2))
            / (2 * np.pi * np.sqrt(np.linalg.det(cov).reshape(-1, 1))))


def background(imgs, window_size=(7, 7), amp=3):
    bg_instensity = stats.mode(imgs.flatten(), keepdims=False)[0]
    bg = np.ones(window_size[0] * window_size[1]) * bg_instensity
    qt_imgs, grid = quantification(bg, window_size, amp)
    qt_imgs = qt_imgs.reshape(1, -1, 1)
    covariance_mat = cov_matrix(grid, qt=qt_imgs)
    return covariance_mat, bg


def kl_divergence2(cov1, cov2):
    a = np.linalg.inv(cov2) * cov1
    return 1./2 * (a[:, 0, 0] + a[:, 1, 1] - 2 + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))


def kl_divergence(base, compares):
    #base = base / np.sum(base)
    #compares = compares / np.sum(compares, axis=1).reshape(-1, 1)
    a = np.sum((compares * np.log(compares / base)), axis=1)
    return a


def ab(img: np.ndarray, bg_cov, bg, window_size=(7, 7), amp=3):
    shift = 1
    surface_window = window_size[0] * window_size[1]
    center_i = int((surface_window - 1) / 2)
    crop_imgs, xy_coords = image_cropping(img, window_size, shift=shift, bg_intensity=bg[0])
    qt_imgs, grid = quantification(crop_imgs, window_size, amp)
    covariance_mat = cov_matrix(grid, qt=qt_imgs)
    pdfs = bi_variate_normal_pdf(grid, covariance_mat)
    alphas = (crop_imgs[:, center_i] / pdfs[:, center_i]).reshape(-1, 1) + 1e-7

    pdfs *= alphas
    kls = kl_divergence(bg, pdfs)
    for img, cov, pdf, kl in zip(crop_imgs, covariance_mat, pdfs, kls):
        if kl > 15:
            plt.figure()
            plt.imshow(img.reshape(window_size), cmap='gray', vmin=0, vmax=1.)
            print(np.sum((img - pdf)**2))
            print(kl)
            print(np.linalg.eig(cov),'\n', cov)
            plt.show()


#background_likelihood(images[0], window_size=(7, 7))

"""
x = np.arange(-(7 - 1) / 2, (7 + 1) / 2)
y = np.arange(-(7 - 1) / 2, (7 + 1) / 2)
xv, yv = np.meshgrid(x, y, sparse=True)
grid = np.stack(np.meshgrid(xv, yv), -1).reshape(7 * 7, 2)
grid = list(grid)
grid.extend([[0, 0]] * 999)
print(grid)
grid = np.array(grid)
print(grid.shape)
print(np.cov(grid.T), np.linalg.eig(np.cov(grid.T)))
kk = grid - np.mean(grid, axis=0)
print(kk)
print(kk.T @ kk / (len(grid)-1))
exit(1)
"""

#images = np.zeros(images.shape) + 0.01
#images[0][3][3] = 1.0
bg_covariance_mat, bg = background(images)
ab(images[0], bg_covariance_mat, bg)
