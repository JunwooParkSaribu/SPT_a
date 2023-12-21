import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from ImageModule import read_tif

images = read_tif('SimulData/receptor_7_low.tif')
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
    surface_window = window_size[0] * window_size[1]
    crop_imgs = image_cropping(img, window_size)
    bg_means = np.sum(crop_imgs, axis=1) / surface_window
    squared_sum = np.sum(crop_imgs ** 2, axis=1)
    bg_var = (squared_sum / surface_window) - (bg_means ** 2)
    gauss_grid = gauss_psf(crop_imgs[0], window_size)
    g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).flatten()
    print(g_bar.shape)
    squared_g = np.sum(g_bar ** 2)
    i_hat = crop_imgs @ g_bar / squared_g
    c = (surface_window / 2.) * np.log(1 - (i_hat**2 * squared_g) / (squared_sum - (surface_window * bg_means)))
    susms = 0
    piv = int(512 / 1. + 1)
    print(int(512 / 1. + 1))
    for i, val in enumerate(list((i_hat**2 * squared_g) / (squared_sum - (surface_window * bg_means)))):
        if val >= 0.5:
            print(i, val)
            print(i//piv * 7, i%piv * 7, val)
            susms += 1
            plt.figure()
            plt.imshow(crop_imgs[i].reshape(window_size))
            plt.show()
    print(susms)

    print(c.shape)
    pass


def image_cropping(img: np.ndarray, window_size=(7, 7)):
    img_height = len(img)
    img_width = len(img[0])
    cropped_imgs = []
    for j in range(0, img_height, 1):
        if j + window_size[1] > img_height:
            j -= (j + window_size[1] - img_height)
        for i in range(0, img_width, 1):
            if i + window_size[0] > img_width:
                i -= (i + window_size[0] - img_width)
            cropped_imgs.append(img[i:i + window_size[0], j:j + window_size[1]].flatten())
    return np.array(cropped_imgs)


background_likelihood(images[0], window_size=(7, 7))