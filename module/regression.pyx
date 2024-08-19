#cython: infer_types=True
#cython: cdivision=True
from libc.math cimport sqrt, M_PI, log
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef pack_vars(double[::1] vars, int len_img):
    cdef double a,b,c,d,e,f
    cdef Py_ssize_t i
    var_stack = np.zeros([len_img, 6], dtype=np.double)

    a = -1. / (2 * vars[0] * (1 - vars[4]*vars[4]))
    b = vars[1] / ((1 - vars[4] * vars[4]) * vars[0]) - (vars[4] * vars[3]) / ((1 - vars[4] * vars[4]) * sqrt(vars[0]) * sqrt(vars[2]))
    c = -1. / (2 * vars[2] * (1 - vars[4]*vars[4]))
    d = vars[3] / ((1 - vars[4] * vars[4]) * vars[2]) - (vars[4] * vars[1]) / ((1 - vars[4] * vars[4]) * sqrt(vars[0]) * sqrt(vars[2]))
    e = vars[4] / ((1 - vars[4] * vars[4]) * sqrt(vars[0]) * sqrt(vars[2]))
    f = (-(vars[1] * vars[1])/(2 * (1 - vars[4] * vars[4]) * vars[0]) - (vars[3] * vars[3])/(2 * (1 - vars[4] * vars[4]) * vars[2]) + (vars[4] * vars[1] * vars[3]) / ((1 - vars[4] * vars[4]) * sqrt(vars[0]) * sqrt(vars[2])) + 
        log(1 / (2 * M_PI * sqrt(vars[0]) * sqrt(vars[2]) * (sqrt(1 - vars[4] * vars[4])))))

    for i in range(len_img):
        var_stack[i][0] = a
        var_stack[i][1] = b
        var_stack[i][2] = c
        var_stack[i][3] = d
        var_stack[i][4] = e
        var_stack[i][5] = f
    return var_stack


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef matrix_decomp(matrix, q):
    ret_mat = []
    for x in range(0, len(matrix), q):
        ret_mat.append(matrix[x: min(x+q, len(matrix))])
    return ret_mat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef unpack_coefs(coefs, window_size):
    err_indices = []
    x_mu = []
    y_mu = []
    rho = coefs[:, 4] * np.sqrt(1/(4 * -abs(coefs[:, 0]) * -abs(coefs[:, 2])))
    k = 1 - rho**2
    x_var = abs(1/(-2 * coefs[:, 0] * k))
    y_var = abs(1/(-2 * coefs[:, 2] * k))
    for err_indice, (xvar_check, yvar_check, r) in enumerate(zip(x_var, y_var, rho)):
        if xvar_check < 0 or yvar_check < 0 or xvar_check > 3 * window_size[0] or yvar_check > 3 * window_size[1] or r < -1 or r > 1 or np.isnan(r):
            #print(r, np.isnan(r))
            err_indices.append(err_indice)

    for i, (b, d) in enumerate(zip(coefs[:, 1], coefs[:, 3])):
        if i in err_indices:
            x_mu.extend([0])
            y_mu.extend([0])
        else:
            coef_mat = np.array([[-rho[i] * np.sqrt(y_var[i]) / np.sqrt(x_var[i]), 1.],
                                 [1., -rho[i] * np.sqrt(x_var[i]) / np.sqrt(y_var[i])]])
            ans_mat = np.array([[d * k[i] * y_var[i]], [b * k[i] * x_var[i]]])
            x_, y_, = np.linalg.lstsq(coef_mat, ans_mat, rcond=None)[0]
            x_mu.extend(x_)
            y_mu.extend(y_)
    x_mu = np.array(x_mu)
    y_mu = np.array(y_mu)
    amp = np.exp(coefs[:, 5] + (x_mu**2 / (2 * k * x_var)) + (y_mu**2 / (2 * k * y_var)) - (rho * x_mu * y_mu / (k * np.sqrt(x_var) * np.sqrt(y_var))))
    return [x_var, x_mu, y_var, y_mu, rho, amp], err_indices


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef guo_algorithm(imgs:np.ndarray, bgs, double[::1] p0, 
                    xgrid, ygrid, window_size=(7, 7), repeat=7, decomp_n=1):
    cdef int k, nb_imgs
    cdef double [:,::1] imgs_view, bgs_view
    cdef double [::1] local_backgrounds

    k = 0
    nb_imgs = imgs.shape[0]
    coef_vals = pack_vars(p0, nb_imgs)
    img_view = imgs
    bgs_view = bgs
    ## background for each crop image needed rather than background intensity for whole image.
    #imgs = np.maximum(np.zeros(imgs.shape), imgs - bgs.reshape(-1, window_size[0], window_size[1])) + 1e-2
    local_backgrounds = element_wise_subtraction_2d(img_view, bgs_view)
    element_wise_maximum_2d(img_view, local_backgrounds)
    yk_2 = imgs.copy()

    while k < repeat:
        if k != 0:
            yk_2 = np.exp(coef_vals[:, 0].reshape(-1, 1) * xgrid**2 + coef_vals[:, 1].reshape(-1, 1) * xgrid +
                          coef_vals[:, 2].reshape(-1, 1) * ygrid**2 + coef_vals[:, 3].reshape(-1, 1) * ygrid +
                          coef_vals[:, 4].reshape(-1, 1) * xgrid * ygrid + coef_vals[:, 5].reshape(-1, 1))
        yk_2 = yk_2 * yk_2
        coef1 = yk_2 * xgrid * xgrid * xgrid * xgrid
        coef2 = yk_2 * xgrid * xgrid * xgrid
        coef3 = yk_2 * xgrid**2 * ygrid**2
        coef4 = yk_2 * xgrid**2 * ygrid
        coef5 = yk_2 * xgrid * xgrid * xgrid * ygrid
        coef6 = yk_2 * xgrid**2
        coef7 = yk_2 * xgrid * ygrid**2
        coef8 = yk_2 * xgrid * ygrid
        coef9 = yk_2 * xgrid
        coef10 = yk_2 * ygrid * ygrid * ygrid * ygrid
        coef11 = yk_2 * ygrid * ygrid * ygrid
        coef12 = yk_2 * xgrid * ygrid * ygrid * ygrid
        coef13 = yk_2 * ygrid**2
        coef14 = yk_2 * ygrid
        coef15 = yk_2
        coef_matrix = np.sum(
            np.array(
                [[coef1, coef2, coef3, coef4, coef5, coef6],
                 [coef2, coef6, coef7, coef8, coef4, coef9],
                 [coef3, coef7, coef10, coef11, coef12, coef13],
                 [coef4, coef8, coef11, coef13, coef7, coef14],
                 [coef5, coef4, coef12, coef7, coef3, coef8],
                 [coef6, coef9, coef13, coef14, coef8, coef15]]
            ), axis=(3)).transpose(2, 0, 1)
        ans1 = xgrid ** 2 * yk_2 * np.log(imgs)
        ans2 = xgrid * yk_2 * np.log(imgs)
        ans3 = ygrid ** 2 * yk_2 * np.log(imgs)
        ans4 = ygrid * yk_2 * np.log(imgs)
        ans5 = xgrid * ygrid * yk_2 * np.log(imgs)
        ans6 = yk_2 * np.log(imgs)
        ans_matrix = np.sum(
            np.array(
                [[ans1], [ans2], [ans3], [ans4], [ans5], [ans6]]
            ), axis=(3), dtype=np.float64).transpose(2, 0, 1)
        coef_matrix = matrix_decomp(coef_matrix, decomp_n)
        ans_matrix = matrix_decomp(ans_matrix, decomp_n)
        decomp_coef_vals = matrix_decomp(coef_vals, decomp_n)
        x_matrix = []
        for (a_mats, b_mats, coef_val) in zip(coef_matrix, ans_matrix, decomp_coef_vals):
            a_mat = np.zeros((a_mats.shape[0] * a_mats.shape[1], a_mats.shape[0] * a_mats.shape[2]))
            for x, vals in zip(range(0, a_mat.shape[0], 6), a_mats):
                a_mat[x:x+6, x:x+6] = vals
            b_mat = b_mats.flatten().reshape(-1, 1)
            x_matrix.extend(np.linalg.lstsq(a_mat, b_mat, rcond=None)[0])
        x_matrix = np.array(x_matrix).reshape(-1, 6)
        if np.allclose(coef_vals, x_matrix, rtol=1e-7):
            break
        coef_vals = x_matrix
        k += 1
    return coef_vals


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef element_wise_maximum_2d(double [:,::1] array1, double[:] local_bgs):
    cdef int nb_imgs, img_size
    cdef Py_ssize_t i, j
    nb_imgs = array1.shape[0]
    img_size = array1.shape[1]
    
    for i in range(nb_imgs):
        for j in range(img_size):
            array1[i][j] = array1[i][j] - local_bgs[i] + 1e-2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef element_wise_subtraction_2d(double [:,::1] array1, double [:,::1] array2):
    assert(array1.shape[0] == array2.shape[0] and array1.shape[1] == array2.shape[1])
    cdef int nb_imgs, img_size
    cdef double local_bg
    cdef Py_ssize_t i, j
    nb_imgs = array1.shape[0]
    img_size = array1.shape[1]
    local_bgs = np.empty(nb_imgs, dtype=np.double)
    cdef double[:] local_bgs_view = local_bgs

    for i in range(nb_imgs):
        local_bg = 9999.0
        for j in range(img_size):
            array1[i][j] = array1[i][j] - array2[i][j]
            local_bg = min(local_bg, array1[i][j])
        local_bgs_view[i] = local_bg
    return local_bgs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef matrix_pow_2d(double [:,::1] array1, int power):
    cdef int row_size, col_size
    cdef Py_ssize_t i, j, k
    cdef double tmp
    row_size = array1.shape[0]
    col_size = array1.shape[1]
    assert(power >= 0)
    if power == 0:
        for i in range(row_size):
            for j in range(col_size):
                array1[i][j] = 1.0 
    elif power >= 2:
        power -= 1
        for i in range(row_size):
            for j in range(col_size):
                    tmp = array1[i][j]
                    for k in range(power):
                        array1[i][j] = array1[i][j] * tmp
