import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from ImageModule import read_tif

#images = read_tif('RealData/20220217_aa4_cel8_no_ir.tif')
#images = read_tif('SimulData/receptor_7_low.tif')
#print(images[0].shape)


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


def image_cropping(img: np.ndarray, window_size=(7, 7), shift=1):
    extend = window_size[0] + 1 if window_size[0] % 2 == 1 else window_size[0]
    mean_val_original_img = np.mean(img)
    extended_img = np.zeros((img.shape[0] + extend, img.shape[1] + extend)) + mean_val_original_img
    extended_img[int(extend/2):int(extend/2) + img.shape[0], int(extend/2):int(extend/2) + img.shape[1]] += (
            img - mean_val_original_img)
    img_height = len(extended_img)
    img_width = len(extended_img[0])
    cropped_imgs = []
    cropped_xy = []
    for j in range(0, img_height-window_size[1]+1, shift):
        for i in range(0, img_width-window_size[0]+1, shift):
            cropped_imgs.append(extended_img[j:j + window_size[1], i:i + window_size[0]].flatten())
            cropped_xy.append([i - int(extend/2), j - int(extend/2)])
    return np.array(cropped_imgs), np.array(cropped_xy)


def cov_matrix(observations):
    obv_mean = np.mean(observations, axis=0)
    a = observations - obv_mean
    estimated_cov = a.T @ a / len(observations)
    #estimated_cov = a.T @ a / (len(observations) - 1)
    print(estimated_cov)


#background_likelihood(images[0], window_size=(7, 7))

kk = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1],
               [-2, 2], [-1, 2], [1, 2], [2, 2], [-2, 1], [2, 1], [-2, -1], [2, -1],
               [-2, -2], [-1, -2], [1, -2], [2, -2],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1],
               [-1, 1], [1, 1], [-1, -1], [1, -1]])
cov_matrix(kk)