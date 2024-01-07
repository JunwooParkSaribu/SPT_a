import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numba import njit
from scipy.optimize import curve_fit

from ImageModule import read_tif

#images = read_tif('RealData/20220217_aa4_cel8_no_ir.tif')
#images = read_tif('SimulData/receptor_7_low.tif')
images = read_tif('tif_trxyt/receptor_7_mid.tif')
#images = read_tif("C:/Users/jwoo/Desktop/U2OS-H2B-Halo_0.25%50ms_field1.tif")
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

    img_height = len(extended_img)
    img_width = len(extended_img[0])
    cropped_imgs = []
    cropped_xy = []
    for j in range(0, img_height-window_size[1]+1, shift):
        for i in range(0, img_width-window_size[0]+1, shift):
            cropped_imgs.append(extended_img[j:j + window_size[1], i:i + window_size[0]].flatten())
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


def bi_variate_normal_pdf(xy, cov, mu=None, normalization=True):
    if mu is None:
        mu = np.array([0, 0])
    else:
        mu = np.array(mu)
    a = np.ones((cov.shape[0], xy.shape[0], xy.shape[1])) * (xy - mu)
    if normalization:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2))
                / (2 * np.pi * np.sqrt(np.linalg.det(cov).reshape(-1, 1))))
    else:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2)))


def background(imgs, window_size=(7, 7), amp=3):
    bg_instensity = stats.mode(
        (imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) * 100).astype(np.uint8), axis=1, keepdims=False)[0] / 100
    bgs = np.ones((bg_instensity.shape[0], window_size[0] * window_size[1]))
    bgs *= bg_instensity.reshape(-1, 1)
    #qt_imgs, grid = quantification(bg, window_size, amp)
    #qt_imgs = qt_imgs.reshape(1, -1, 1)
    #covariance_mat = cov_matrix(grid, qt=qt_imgs)
    return bgs


def kl_divergence2(cov1, cov2):
    a = np.linalg.inv(cov2) * cov1
    return 1./2 * (a[:, 0, 0] + a[:, 1, 1] - 2 + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))


def kl_divergence(base, compares):
    #base = base / np.sum(base)
    #compares = compares / np.sum(compares, axis=1).reshape(-1, 1)
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


def ab(img: np.ndarray, bg, window_size=(7, 7), amp=3):
    shift = 1
    surface_window = window_size[0] * window_size[1]
    center_i = int((surface_window - 1) / 2)
    crop_imgs, xy_coords = image_cropping(img, window_size, shift=shift, bg_intensity=bg[0])
    qt_imgs, grid = quantification(crop_imgs, window_size, amp)
    covariance_mat = cov_matrix(grid, qt=qt_imgs)
    pdfs = bi_variate_normal_pdf(grid, covariance_mat)
    intensity, bg_i = intensity_reg(crop_imgs, pdfs, center_i)
    alphas2 = (crop_imgs[:, center_i] / pdfs[:, center_i]).reshape(-1, 1)
    pdfs1 = pdfs * intensity + bg_i
    pdfs2 = pdfs * alphas2
    kls1 = kl_divergence(bg, pdfs1)
    kls2 = kl_divergence(bg, pdfs2)
    for img, xy, cov, pdf1, pdf2, kl1, kl2 in zip(crop_imgs, xy_coords, covariance_mat, pdfs1, pdfs2, kls1, kls2):
        if kl1 > 0.1:
            coefs = guo_algorithm(img, bg, window_size=window_size)
            x_var, y_var, x0, y0, amp = unpack_coefs(coefs)
            print(x_var, y_var, x0, y0, amp)
            pdfs = bi_variate_normal_pdf(grid, np.array([[[x_var, 0], [0, y_var]]], dtype=np.float64), normalization=False)
            pdfs = amp * pdfs + bg
            print(kl_divergence(bg, pdfs))
            plt.figure()
            plt.imshow((img - bg).reshape(window_size), cmap='gray',
                       vmin=0, vmax=1)
            plt.figure()
            plt.imshow(np.hstack((img.reshape(window_size), pdfs[0].reshape(window_size))), cmap='gray',
                       vmin=0, vmax=1)
            plt.show()
            continue
            exit(1)
            print('xy = ', xy)
            print('diff1=', np.sum((img - pdf1) ** 2), ' diff2=',np.sum((img - pdf2) ** 2))
            print('kl1=', kl1, ' kl2=', kl2)
            print('eigvals=', np.linalg.eig(cov)[0], '\n', 'eigvecs=', list(np.linalg.eig(cov)[1]), '\n', cov)
            print(np.mean(img), np.std(img))
            plt.figure()
            plt.imshow(np.hstack((img.reshape(window_size), pdf1.reshape(window_size))), cmap='gray', vmin=0, vmax=1.)
            plt.vlines(x=window_size[0]-.5, ymin=0, ymax=window_size[1]-1, colors='red')
            plt.show()


def unpack_coefs(coefs):
    x_var = -1./(2 * coefs[0])
    y_var = -1./(2 * coefs[2])
    x0 = coefs[1] * x_var
    y0 = coefs[3] * y_var
    amp = np.exp(coefs[4] + ((x0**2)/(2 * x_var)) + ((y0**2)/(2 * y_var)))
    return np.array([x_var, y_var, x0, y0, amp])

def pack_vars(vars):
    coef0 = -1./(2 * vars[0])
    coef2 = -1. / (2 * vars[1])
    coef1 = vars[2] / vars[0]
    coef3 = vars[3] / vars[1]
    coef4 = np.log(vars[4]) - ((vars[2]**2)/(2 * vars[0])) - ((vars[3]**2)/(2 * vars[1]))
    return np.array([coef0, coef1, coef2, coef3, coef4])


def guo_algorithm(img, bg, bound=None, window_size=(7, 7)):
    if bound is None:
        bound = [[-1e5, 1e5], [-1e5, 1e5], [-0.5, 0.5], [-0.5, 0.5], [0, 1e5]]
    coef_vals = pack_vars([10, 10, 0, 0, 0.5])
    img = img.reshape(window_size)
    img = np.maximum(np.zeros(img.shape), img - bg.reshape(img.shape)) + 1e-2
    yk_2 = img.astype(np.float64)
    x_grid = np.array([list(np.arange(-int(window_size[0]/2), int((window_size[0]/2) + 1), 1))] * window_size[1])
    y_grid = np.array([[y] * window_size[0] for y in range(int(window_size[1]/2), -int((window_size[1]/2) + 1), -1)])
    for k in range(0, 20):
        if k != 0:
            x_var, y_var, x0, y0, amp = unpack_coefs(coef_vals)
            yk_2 = np.exp(coef_vals[0] * x_grid**2 + coef_vals[1] * x_grid +
                          coef_vals[2] * y_grid**2 + coef_vals[3] * y_grid + coef_vals[4])
            #if abs(x0) >= 1 or abs(y0) >= 1:
            #    return coef_vals
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
            ), axis=(2, 3))
        ans1 = x_grid ** 2 * yk_2 * np.log(img)
        ans2 = x_grid * yk_2 * np.log(img)
        ans3 = y_grid ** 2 * yk_2 * np.log(img)
        ans4 = y_grid * yk_2 * np.log(img)
        ans5 = yk_2 * np.log(img)
        ans_matrix = np.sum(
            np.array(
                [[ans1], [ans2], [ans3], [ans4], [ans5]]
            ), axis=(2, 3))
        akz = np.linalg.lstsq(coef_matrix, ans_matrix, rcond=None)[0]
        print('@', unpack_coefs(akz))
        coef_vals = gauss_seidel(coef_matrix, ans_matrix, p0=coef_vals, bound=bound, iter=200)
        print('#', unpack_coefs(coef_vals))
        #for val, bd in zip(unpack_coefs(coef_vals), bound):
        #    if val < bd[0] or val > bd[1]:
        #        return coef_vals
    return coef_vals


def gauss_seidel(a, b, p0, bound, iter=1000, tol=1e-8):
    x = np.array(p0)
    for it_count in range(1, iter):
        x_new = np.zeros_like(x, dtype=np.float_)
        print(f"Iteration {it_count}: {unpack_coefs(x)}")
        for i in range(a.shape[0]):
            s1 = np.dot(a[i, :i], x_new[:i])
            s2 = np.dot(a[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / a[i, i]
        if np.allclose(x, x_new, rtol=tol):
            break
        for val, bd in zip(unpack_coefs(x_new), bound):
            if val < bd[0] or val > bd[1]:
                return x_new
        x = x_new

    print(f"Solution: {x}")
    error = np.dot(a, x) - b
    print(f"Error: {error}")
    return x

#background_likelihood(images[0], window_size=(7, 7))
#images = np.zeros(images.shape) + 0.01
#images[0][3][3] = 1.0
bgs = background(images, window_size=(15, 15))
ab(images[0], bgs[0], window_size=(15, 15))
