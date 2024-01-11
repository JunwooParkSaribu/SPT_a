import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from numba import njit
from scipy.optimize import curve_fit

from ImageModule import read_tif

images = read_tif('RealData/20220217_aa4_cel8_no_ir.tif')
#images = read_tif('SimulData/receptor_7_low.tif')
#images = read_tif('tif_trxyt/receptor_7_mid.tif')
#images = read_tif('tif_trxyt/U2OS-H2B-Halo_0.25%50ms_field1.tif')
#images = read_tif("C:/Users/jwoo/Desktop/U2OS-H2B-Halo_0.25%50ms_field1.tif")
print(images[0].shape)


def gauss_psf(window_size, radius):
    x_subpixel = np.arange(window_size[0]) + .5
    y_subpixel = np.arange(window_size[1]) + .5
    center_x = window_size[0] / 2.
    center_y = window_size[1] / 2.
    base_vals = np.ones((window_size[1], window_size[0], 2)) * np.array([center_x, center_y])
    gauss_psf_vals = np.stack(np.meshgrid(x_subpixel, y_subpixel), -1)
    gauss_psf_vals = np.exp(-(np.sum((gauss_psf_vals - base_vals)**2, axis=2))/(2*(radius**2))) / (np.sqrt(np.pi) * radius)
    return gauss_psf_vals


def background_likelihood2(img: np.ndarray, bg, window_size=(7, 7)):
    shift = 1
    surface_window = window_size[0] * window_size[1]
    crop_imgs, xy_coords = image_cropping(img, window_size, shift=shift)
    bg_means = np.sum(crop_imgs, axis=1) / surface_window
    bg_squared_sum = np.sum(crop_imgs ** 2, axis=1)
    cs = []
    for radius in [1, 5, 9, 15]:
        gauss_grid = gauss_psf(window_size)
        g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).flatten()
        g_squared_sum = np.sum(g_bar ** 2)
        i_hat = crop_imgs @ g_bar / g_squared_sum
        c = (surface_window / 2.) * np.log(1 - (i_hat**2 * g_squared_sum) / (bg_squared_sum - (surface_window * bg_means)))
        cs.append(c)
    for i, (val0, val1, val2, val3, xy) in enumerate(zip(cs[0], cs[1], cs[2], cs[3], xy_coords)):
        if val0 > 0.5 or val1 > 0.5 or val2 > 0.5 or val3 > 0.5:
            print(xy, val0, val1, val2, val3)
            plt.figure()
            plt.imshow(crop_imgs[i].reshape(window_size), vmin=0, vmax=1., cmap='gray')
            plt.show()
    print(c.shape)
    pass


def background_likelihood(img: np.ndarray, bg, window_sizes):
    cs = []
    h_map = np.zeros(img.shape)
    shift = 1
    bg_mean = bg[0][0][0]
    xy_s = []
    my_imgs = []
    for window_size, radius in zip(window_sizes, [1.1, 3, 5, 7]):
        surface_window = window_size[0] * window_size[1]
        crop_imgs, xy_coords = image_cropping(img, window_size, shift=shift)
        my_imgs.append(crop_imgs)
        xy_s.append(xy_coords)
        #bg_means = np.sum(crop_imgs, axis=1) / surface_window
        #bg_squared_sum = np.sum(crop_imgs ** 2, axis=1)
        bg_squared_sum = np.sum(window_size[0] * window_size[1] * bg_mean**2)
        bg_variance = (1 / surface_window) * bg_squared_sum - bg_mean**2
        #c = (surface_window / 2.) * np.log(target_variance**2 / bg_variance**2)
        #plt.figure()
        #plt.hist(c, bins=np.arange(np.min(c), np.max(c)+1, 10))
        #print(c.shape, c[:5])
        #plt.show()

        gauss_grid = gauss_psf(window_size, radius)
        g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).flatten()
        g_squared_sum = np.sum(g_bar ** 2)
        i_hat = crop_imgs @ g_bar / g_squared_sum
        c = (surface_window / 2.) * np.log(1 - (i_hat**2 * g_squared_sum) / (bg_squared_sum - (surface_window * bg_mean)))
        cs.append(c)

    for i, (val0, val1, val2, val3) in enumerate(zip(cs[0], cs[1], cs[2], cs[3])):
        print(i % img.shape[1], i // img.shape[1])
        h_map[i // img.shape[1]][i % img.shape[1]] = np.max([val0, val1, val2, val3])
        """
        if val0 > 0.5 or val1 > 1 or val2 > 1 or val3 > 1:
            print(val0, val1, val2, val3)
            max_argg = np.argmax([val0, val1, val2, val3])
            print('xy: ', xy_s[max_argg][i])
            plt.figure()
            plt.imshow(my_imgs[max_argg][i].reshape(window_sizes[max_argg]), vmin=0, vmax=1., cmap='gray')
            plt.show()
        """
    plt.figure()
    plt.imshow(h_map)
    plt.show()
    pass


def image_cropping(img: np.ndarray, window_size=(7, 7), shift=1, bg_intensity=None):
    if bg_intensity is None:
        bg_intensity = np.mean(img)
    extend = window_size[0] - 1 if window_size[0] % 2 == 1 else window_size[0]
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
    coefs = guo_algorithm(crop_imgs, bg, window_size=window_size)
    variables = unpack_coefs(coefs)
    cov_mat = np.array([variables[:, 0], [0]*variables.shape[0], [0]*variables.shape[0], variables[:, 1]]).T.reshape(variables.shape[0], 2, 2)
    pdfs3 = bi_variate_normal_pdf(grid, cov_mat, normalization=False)
    pdfs3 = variables[:, 4].reshape(-1, 1) * pdfs3 + bg

    covariance_mat = cov_matrix(grid, qt=qt_imgs)
    pdfs = bi_variate_normal_pdf(grid, covariance_mat)
    intensity, bg_i = intensity_reg(crop_imgs, pdfs, center_i)
    alphas2 = (crop_imgs[:, center_i] / pdfs[:, center_i]).reshape(-1, 1)
    pdfs1 = pdfs * intensity + bg_i
    pdfs2 = pdfs * alphas2
    kls1 = kl_divergence(bg, pdfs1)
    kls2 = kl_divergence(bg, pdfs2)
    kls3 = kl_divergence(bg, pdfs3)
    for img, xy, vv, pdf1, pdf2, pdf3, kl1, kl2, kl3 in zip(crop_imgs, xy_coords, variables, pdfs1, pdfs2, pdfs3, kls1, kls2, kls3):
        #if kl1 > 0.03:
            print('xy = ', xy)
            print('diff1=', np.sum((img - pdf1) ** 2), ' diff2=', np.sum((img - pdf2) ** 2), ' diff3=', np.sum((img - pdf3) ** 2))
            print('kl1=', kl1, ' kl2=', kl2, ' kl3=', kl3)
            print('variables=', vv)
            plt.figure()
            plt.imshow(np.hstack((img.reshape(window_size), pdf3.reshape(window_size))), cmap='gray', vmin=0, vmax=1.)
            plt.vlines(x=window_size[0]-.5, ymin=0, ymax=window_size[1]-1, colors='red')
            plt.show()


def unpack_coefs(coefs):
    x_var = -1./(2 * coefs[:, 0])
    y_var = -1./(2 * coefs[:, 2])
    x0 = coefs[:, 1] * x_var
    y0 = coefs[:, 3] * y_var
    amp = np.exp(coefs[:, 4] + ((x0**2)/(2 * x_var)) + ((y0**2)/(2 * y_var)))
    return np.array([x_var, y_var, x0, y0, amp]).T


def pack_vars(vars, len_img):
    coef0 = -1./(2 * vars[0])
    coef2 = -1. / (2 * vars[1])
    coef1 = vars[2] / vars[0]
    coef3 = vars[3] / vars[1]
    coef4 = np.log(vars[4]) - ((vars[2]**2)/(2 * vars[0])) - ((vars[3]**2)/(2 * vars[1]))
    return np.array([[coef0, coef1, coef2, coef3, coef4] for _ in range(len_img)])


def matrix_decomp(matrix, q):
    ret_mat = []
    for x in range(0, len(matrix), q):
        ret_mat.append(matrix[x: min(x+q, len(matrix))])
    return ret_mat


def guo_algorithm(imgs, bg, p0=None, window_size=(7, 7)):
    nb_imgs = imgs.shape[0]
    if p0 is not None:
        coef_vals = pack_vars(p0, nb_imgs)
    else:
        coef_vals = pack_vars([2, 2, 0, 0, 0.1], nb_imgs)

    imgs = imgs.reshape(imgs.shape[0], window_size[0], window_size[1])
    ## background for each crop image needed rather than background intensity for whole image.
    imgs = np.maximum(np.zeros(imgs.shape), imgs - bg.reshape(-1, window_size[0], window_size[1])) + 1e-2
    yk_2 = imgs.astype(np.float64)
    x_grid = (np.array([list(np.arange(-int(window_size[0]/2), int((window_size[0]/2) + 1), 1))] * window_size[1])
              .reshape(-1, window_size[0], window_size[1]))
    y_grid = (np.array([[y] * window_size[0] for y in range(int(window_size[1]/2), -int((window_size[1]/2) + 1), -1)])
              .reshape(-1, window_size[0], window_size[1]))
    for k in range(0, 5):
        print(k)
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
        coef_matrix = matrix_decomp(coef_matrix, 5)
        ans_matrix = matrix_decomp(ans_matrix, 5)
        decomp_coef_vals = matrix_decomp(coef_vals, 5)
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

    #print(f"Solution: {x}")
    #error = np.dot(a, x) - b
    #print(f"Error: {error}")
    return x


diff_bgs = []
for window_size in [(5, 5), (7, 7), (11, 11), (15, 15)]:
    bgs = background(images, window_size=window_size)
    diff_bgs.append(bgs)


background_likelihood(images[0], diff_bgs, window_sizes=[(5, 5), (7, 7), (11, 11), (15, 15)])
#ab(images[0], bgs[0], window_size=(9, 9))
