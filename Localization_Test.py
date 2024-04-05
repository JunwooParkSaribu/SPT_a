import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import tifffile
import concurrent.futures
from FileIO import write_localization, read_parameters, check_video_ext
from numba.typed import List as nbList
from ImageModule import draw_cross
from timeit import default_timer as timer


def region_max_filter2(maps, window_size, thresholds):
    indices = []
    r_start_index = int(window_size[1] // 2)
    col_start_index = int(window_size[0] // 2)
    for _ in range(2):
        args_map = maps > thresholds.reshape(-1, 1, 1)
        maps = maps * args_map
        img_n, row, col = np.where(args_map == True)
        for n, r, c in zip(img_n, row, col):
            if maps[n][r][c] == np.max(maps[n, max(0, r-r_start_index):min(maps.shape[1]+1, r+r_start_index+1),
                                       max(0, c-col_start_index):min(maps.shape[2]+1, c+col_start_index+1)]) and maps[n][r][c] != 0:
                indices.append([n, r, c])
                maps[n, max(0, r - r_start_index):min(maps.shape[1] + 1, r + r_start_index + 1),
                max(0, c - col_start_index):min(maps.shape[2] + 1, c + col_start_index + 1)] = 0
    return indices


def region_max_filter(maps, window_sizes, thresholds):
    indices = []
    nb_imgs = maps.shape[1]
    infos = [[] for _ in range(nb_imgs)]
    for i, (hmap, window_size) in enumerate(zip(maps, window_sizes)):
        args_map = hmap > thresholds[:, i].reshape(-1, 1, 1)
        maps[i] = hmap * args_map
        r_start_index = int(window_size[1] // 2)
        col_start_index = int(window_size[0] // 2)
        img_n, row, col = np.where(args_map == True)
        for n, r, c in zip(img_n, row, col):
            if maps[i][n][r][c] == np.max(
                    maps[i, n, max(0, r - r_start_index):min(maps[i].shape[1] + 1, r + r_start_index + 1),
                    max(0, c - col_start_index):min(maps[i].shape[2] + 1, c + col_start_index + 1)]):
                infos[n].append([i, r, c, hmap[n][r][c]])
    maps = np.moveaxis(maps, 0, 1)
    for img_n, info in enumerate(infos):
        mask = np.zeros((maps.shape[2], maps.shape[3])).astype(np.uint8)
        if len(info) > 0:
            info = np.array(info)
            info = info[np.argsort(info[:, 3])[::-1]]
            for mol_info in info:
                extend = int((window_sizes[int(mol_info[0])][0] - 1) / 2)
                row_min = int(max(0, mol_info[1] - extend))
                row_max = int(min(mask.shape[0] - 1, mol_info[1] + extend))
                col_min = int(max(0, mol_info[2] - extend))
                col_max = int(min(mask.shape[1] - 1, mol_info[2] + extend))
                if mask[int(mol_info[1])][int(mol_info[2])] != 1:
                    indices.append([img_n, int(mol_info[1]), int(mol_info[2]), int(window_sizes[int(mol_info[0])][0])])
                    mask[row_min:row_max, col_min:col_max] = 1
    return np.array(indices)


def subtract_pdf(ext_imgs, pdfs, indices, window_size, bg_means, extend):
    for pdf, (n, r, c) in zip(pdfs, indices):
        bg = (np.ones(pdf.shape) * bg_means[n]).reshape(window_size)
        pdf = np.ascontiguousarray(pdf).reshape(window_size)
        row_indice = np.array([r - int((window_size[1]-1)/2), r + int((window_size[1]-1)/2)]) + int(extend/2)
        col_indice = np.array([c - int((window_size[0]-1)/2), c + int((window_size[0]-1)/2)]) + int(extend/2)
        ext_imgs[n][row_indice[0]:row_indice[1]+1, col_indice[0]:col_indice[1]+1] -= pdf
        ext_imgs[n][row_indice[0]:row_indice[1] + 1, col_indice[0]:col_indice[1] + 1] = (
            np.maximum(ext_imgs[n][row_indice[0]:row_indice[1]+1, col_indice[0]:col_indice[1]+1], bg))
        ext_imgs[n] = boundary_smoothing(ext_imgs[n], row_indice, col_indice)
    return ext_imgs


def boundary_smoothing(img, row_indice, col_indice):
    center_xy = []
    repeat_n = 2
    borders = [x for x in range(50)]
    erase_space = 2
    for border in borders:
        row_min = max(0, row_indice[0]+border)
        row_max = min(img.shape[0]-1, row_indice[1]-border)
        col_min = max(0, col_indice[0]+border)
        col_max = min(img.shape[1]-1, col_indice[1]-border)
        for col in range(col_min, col_max+1):
            center_xy.append([row_min, col])
        for row in range(row_min, row_max+1):
            center_xy.append([row, col_max])
        for col in range(col_max, col_min-1, -1):
            center_xy.append([row_max, col])
        for row in range(row_max, row_min-1, -1):
            center_xy.append([row, col_min])
    for _ in range(repeat_n):
        for r, c in center_xy:
            img[r][c] = np.mean(img[max(0, r-erase_space):min(img.shape[0], r+erase_space+1), max(0, c-erase_space):min(img.shape[1], c+erase_space+1)])
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
    #crop_imgs = np.ascontiguousarray(crop_imgs)
    surface_window = window_size[0] * window_size[1]
    #gauss_grid = gauss_grid / np.sum(gauss_grid)
    g_bar = (gauss_grid - (np.sum(gauss_grid) / surface_window)).reshape(window_size[0]*window_size[1], 1)
    g_squared_sum = np.sum(g_bar ** 2)

    before_time = timer()
    i_hat = (crop_imgs - bg_means.reshape(crop_imgs.shape[0], 1, 1))
    before_time = timer()
    i_hat = i_hat @ g_bar / g_squared_sum
    i_hat = np.maximum(np.zeros(i_hat.shape), i_hat)
    L = ((surface_window / 2.) * np.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    return L


def add_block_noise(imgs, extend):
    gap = int(extend/2)
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


def indice_filtering(indices, window_sizes, img_shape, extend):
    masks = np.zeros(img_shape)
    win_masks = [[[[] for _ in range(img_shape[2])] for _ in range(img_shape[1])] for _ in range(img_shape[0])]
    for indexx, wins in zip(indices[::-1], window_sizes[::-1]):
        for index in indexx:
            masks[index[0], index[1], index[2]] += 1
            win_masks[index[0]][index[1]][index[2]].append(wins[0])
    ret_indices = check_masks_overlaps(masks, win_masks, extend, window_sizes)

    """
    regions = [[] for _ in range(len(indices))]
    max_window = window_sizes[-1]
    for i, indexx in enumerate(indices):
        for index in indexx:
            r = [max(0, index[1] - int((max_window[1]-1)/2)), min(img_shape[1]-1, index[1] + int((max_window[1]-1)/2)),
            max(0, index[2] - int((max_window[0]-1)/2)), min(img_shape[2]-1, index[2] + int((max_window[0]-1)/2))]
            regions[i].append(r)
        #win_mask[index[0]][index[1]][index[2]].append(window_sizes[-1][0])
    ret_indices = []
    for indexx, regs in zip(indices[::-1], regions[::-1]):
        for index, reg in zip(indexx, regs):
            #if np.sum(mask[index[0], reg[0]:reg[1]+1, reg[2]:reg[3]+1]) > len(indices) - 1:
            al = []
            ms = {ws[0]: [] for ws in window_sizes}
            rs, cs = np.where(mask[index[0], reg[0]:reg[1]+1, reg[2]:reg[3]+1] >= 1)
            for r, c in zip(rs, cs):
                for win_size_val in win_mask[index[0]][r + reg[0]][c + reg[2]]:
                    mask[index[0], reg[0]:reg[1]+1, reg[2]:reg[3]+1] -= 1
                    ms[int(win_size_val)].append([index[0], r + reg[0] + extend, c + reg[2] + extend, int(win_size_val)])
            for ws in ms:
                if len(ms[ws]) > 0:
                    al.append(ms[ws])
            if len(al) > 0:
                ret_indices.append(al)

    print(ret_indices)
    """
    return ret_indices


def check_masks_overlaps(masks, window_masks, extend, window_sizes):
    nb_window_sizes = len(window_sizes)
    w_size_dict = {ws[0]: i for i, ws in enumerate(window_sizes)}
    all_groups = []
    for img_n, (mask, window_mask) in enumerate(zip(masks, window_masks)):
        #plt.figure()
        #plt.imshow(mask)
        #plt.show()
        groups = []
        overlay_mask = np.zeros_like(mask, dtype=np.uint8)
        rs, cs = np.where(mask >= 1)
        if len(rs) == 0:
            continue

        coords = np.vstack((rs, cs)).T
        while 1:
            group = [[] for _ in range(nb_window_sizes)]
            selected_args = []
            piv_coord = coords[0]

            for window_size in window_mask[piv_coord[0]][piv_coord[1]]:
                group[w_size_dict[window_size]].append([img_n, piv_coord[0] + extend, piv_coord[1] + extend, window_size])
                row_min = max(0, piv_coord[0] - (window_size // 2))
                row_max = min(mask.shape[0] - 1, piv_coord[0] + (window_size // 2))
                col_min = max(0, piv_coord[1] - (window_size // 2))
                col_max = min(mask.shape[1] - 1, piv_coord[1] + (window_size // 2))
                overlay_mask[row_min: row_max+1, col_min: col_max+1] += 1

            explorer_coords = coords[1:].copy()
            while 1:
                group_added = 0
                for selected_index, coord in enumerate(explorer_coords):
                    if selected_index in selected_args:
                        continue
                    if overlay_mask[coord[0]][coord[1]] >= 1:
                        selected_args.append(selected_index)
                        for window_size in window_mask[coord[0]][coord[1]]:
                            row_min = max(0, coord[0] - (window_size // 2))
                            row_max = min(mask.shape[0] - 1, coord[0] + (window_size // 2))
                            col_min = max(0, coord[1] - (window_size // 2))
                            col_max = min(mask.shape[1] - 1, coord[1] + (window_size // 2))
                            overlay_mask[row_min: row_max + 1, col_min: col_max + 1] += 1
                            group[w_size_dict[window_size]].append([img_n, coord[0] + extend, coord[1] + extend, window_size])
                            group_added += 1

                if group_added == 0:
                    groups.append(group)
                    break
            selected_args = np.array(selected_args) + 1
            selected_args = np.append(selected_args, 0).astype(np.uint32)
            coords = np.delete(coords, selected_args, axis=0)
            if len(coords) == 0:
                break
        all_groups.extend(groups)
    return all_groups


def localization(imgs: np.ndarray, bgs, f_gauss_grids, b_gauss_grids, *args):
    index = 0
    bin_winsizes = args[0]
    bin_radius = args[1]
    bin_thresholds = args[2]
    multi_winsizes = args[3]
    multi_radius = args[4]
    multi_thresholds = args[5]
    deflation_loop_backward = args[10]
    extend = multi_winsizes[-1][0]*4

    forward_linkage = {i: ws[0] for i, ws in enumerate(bin_winsizes)}
    backward_linkage = {ws[0]: i for i, ws in enumerate(multi_winsizes)}
    coords = [[] for _ in range(imgs.shape[0])]
    reg_pdfs = [[] for _ in range(imgs.shape[0])]
    reg_infos = [[] for _ in range(imgs.shape[0])]
    bg_means = bgs[multi_winsizes[0][0]][:, 0]
    extended_imgs = np.zeros((imgs.shape[0], imgs.shape[1] + extend, imgs.shape[2] + extend))
    extended_imgs[:, int(extend/2):int(extend/2) + imgs.shape[1], int(extend/2):int(extend/2) + imgs.shape[2]] += imgs
    extended_imgs = add_block_noise(extended_imgs, extend)

    while 1:
        window_sizes = bin_winsizes[index:]
        thresholds = bin_thresholds[:, index:]
        radiuss = bin_radius[index:]
        g_grids = f_gauss_grids[index:]
        all_crop_imgs = {ws[0]: None for ws in multi_winsizes}
        win_s_dict = {}
        for ws in window_sizes:
            win_s_dict[ws[0]] = []

        if index == bin_thresholds.shape[1] - 1:
            for df_loop in range(deflation_loop_backward):
                h_maps = []
                for step, (g_grid, window_size, radius) in (
                        enumerate(zip(b_gauss_grids, multi_winsizes, multi_radius))):
                    crop_imgs = image_cropping(extended_imgs, extend, window_size, shift=1)
                    crop_imgs = np.array(crop_imgs).reshape(imgs.shape[1] * imgs.shape[2], imgs.shape[0],
                                                            window_size[0] * window_size[1])
                    crop_imgs = np.moveaxis(crop_imgs, 0, 1)
                    all_crop_imgs[window_size[0]] = crop_imgs
                    bg_squared_sums = window_size[0] * window_size[1] * bg_means ** 2
                    c = likelihood(crop_imgs.copy(), g_grid, bg_squared_sums, bg_means, window_size)
                    h_maps.append(c.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2]))
                h_maps = np.array(h_maps)

                """
                plt.figure()
                plt.imshow(extended_imgs[0])
                for hm in h_maps:
                    plt.figure()
                    plt.imshow(hm[0], vmin=0., vmax=1.)
                plt.show()
                """

                back_indices = [[] for _ in range(multi_thresholds.shape[1])]
                for backward_index in range(multi_thresholds.shape[1]-1, -1, -1):
                    back_indices[backward_index] = region_max_filter2(h_maps[backward_index], multi_winsizes[backward_index],
                                                                      multi_thresholds[:, backward_index])
                reregress_indice = indice_filtering(back_indices, multi_winsizes, imgs.shape, int(extend/2))
                regress_imgs_copy = extended_imgs.copy()
                for regress_comp_set in reregress_indice:
                    loss_vals = []
                    selected_dt = []
                    for win_s_set in regress_comp_set:
                        regress_imgs = []
                        bg_regress = []
                        for regress_index in win_s_set:
                            ws = regress_index[3]
                            regress_imgs.append(regress_imgs_copy[regress_index[0],
                                                regress_index[1] - int((regress_index[3] - 1) / 2):regress_index[1] + int(
                                                    (regress_index[3] - 1) / 2) + 1,
                                                regress_index[2] - int((regress_index[3] - 1) / 2):regress_index[2] + int(
                                                    (regress_index[3] - 1) / 2) + 1])
                            bg_regress.append(bgs[ws][regress_index[0]])
                        regress_imgs = np.array(regress_imgs)

                        if len(regress_imgs) > 0:
                            pdfs, xs, ys, x_vars, y_vars, amps, rhos = image_regression(regress_imgs, bg_regress,
                                                                                        (ws, ws), p0=args[6], decomp_n=args[8])
                            penalty = 0
                            for x_var, y_var, rho, dx, dy in zip(x_vars, y_vars, rhos, xs, ys):
                                if x_var < 0 or y_var < 0 or x_var > 3*ws or y_var > 3*ws or rho > 1 or rho < -1:
                                    penalty += 1e6
                            regressed_imgs = []
                            for regress_index, dx, dy in zip(win_s_set, xs, ys):
                                reged_img = regress_imgs_copy[regress_index[0],
                                            regress_index[1] - int((ws - 1) / 2) + int(np.round(dy)):
                                            regress_index[1] + int((ws - 1) / 2) + int(np.round(dy)) + 1,
                                            regress_index[2] - int((ws - 1) / 2) + int(np.round(dx)):
                                            regress_index[2] + int((ws - 1) / 2) + int(np.round(dx)) + 1]
                                if reged_img.shape == (ws, ws):
                                    regressed_imgs.append(reged_img)
                            regressed_imgs = np.array(regressed_imgs)
                            selected_dt.append([pdfs, xs, ys, x_vars, y_vars, rhos, amps])
                            if regressed_imgs.shape != (pdfs.reshape(regress_imgs.shape)).shape:
                                ## x_var or y_var is (-)
                                loss_vals.append(penalty)
                            else:
                                loss = np.mean(np.sort(np.mean(np.log(abs(regressed_imgs - pdfs.reshape(regress_imgs.shape))), axis=0).
                                                       flatten())[::-1][:multi_winsizes[0][0] * multi_winsizes[0][1]]) + penalty
                                #loss = np.mean(abs(regressed_imgs - pdfs.reshape(regress_imgs.shape))**2) * penalty
                                #loss = np.mean(abs(kl_divergence(pdfs, regressed_imgs.reshape(pdfs.shape)))) * penalty
                                loss_vals.append(loss)
                        else:
                            selected_dt.append([0, 0, 0, 0, 0, 0, 0])
                            loss_vals.append(1e3)

                    #print('------------------')
                    #print(loss_vals)
                    #print(regress_comp_set, ',  extend:', extend/2)

                    if np.sum(np.array(loss_vals) < 0.) >= 1:
                        selec_arg = np.argmin(loss_vals)
                        pdfs, xs, ys, x_vars, y_vars, rhos, amps = selected_dt[selec_arg]
                        infos = np.array(regress_comp_set[selec_arg])
                        ns = infos[:, 0]
                        rs = infos[:, 1]
                        cs = infos[:, 2]
                        ws = infos[:, 3][0]
                        for n, r, c, dx, dy, pdf, x_var, y_var, rho, amp in zip(ns, rs, cs, xs, ys, pdfs, x_vars, y_vars, rhos, amps):
                            r -= int(extend/2)
                            c -= int(extend/2)
                            if r+dy <= -1 or r+dy >= imgs.shape[1] or c+dx <= -1 or c+dx >= imgs.shape[2]:
                                continue
                            row_coord = max(0, min(r+dy, imgs.shape[1]-1))
                            col_coord = max(0, min(c+dx, imgs.shape[2]-1))
                            coords[n].append([row_coord, col_coord])
                            reg_pdfs[n].append(pdf)
                            reg_infos[n].append([x_var, y_var, rho, amp])
                        if df_loop < deflation_loop_backward - 1:
                            del_indices = np.round(np.array([ns, rs+ys, cs+xs])).astype(int).T
                            extended_imgs = subtract_pdf(extended_imgs, pdfs, del_indices, (ws, ws), bg_means, extend=0)
            return coords, reg_pdfs, reg_infos

        else:
            h_maps = []
            for step, (g_grid, window_size, radius) in (
                    enumerate(zip(g_grids, window_sizes, radiuss))):
                #print(f'{step} : {imgs.shape}')
                before_time = timer()
                h_map = np.zeros_like(imgs)
                crop_imgs = np.array(image_cropping(extended_imgs, extend, window_size, shift=args[7]))
                crop_imgs = crop_imgs.reshape(crop_imgs.shape[0], crop_imgs.shape[1],
                                                        window_size[0] * window_size[1])
                #print(f'{step}{": 1 calcul":<35}:{(timer() - before_time):.2f}s')

                crop_imgs = np.moveaxis(crop_imgs, 0, 1)
                all_crop_imgs[window_size[0]] = crop_imgs
                bg_squared_sums = window_size[0] * window_size[1] * bg_means**2

                before_time = timer()
                c = likelihood(crop_imgs, g_grid, bg_squared_sums, bg_means, window_size)
                #print(f'{step}{": 3 calcul":<35}:{(timer() - before_time):.2f}s')
                before_time = timer()
                h_map = mapping(h_map, c, args[7])
                h_maps.append(h_map)
                #print(f'{step}{": hmap calcul":<35}:{(timer() - before_time):.2f}s')
            h_maps = np.array(h_maps)
            """
            plt.figure()
            plt.imshow(extended_imgs[0])
            for hm in h_maps:
                plt.figure()
                plt.imshow(hm[0], vmin=0., vmax=1.)
            plt.show()
            """
            indices = region_max_filter(h_maps, window_sizes, thresholds)
            if len(indices) != 0:
                for n, r, c, ws in indices:
                    #win_s_dict[ws].append([all_crop_imgs[ws][n][imgs.shape[2] * r + c], bgs[ws][n], n, r, c])
                    win_s_dict[ws].append([all_crop_imgs[ws][n]
                                           [int((r//args[7]) * (imgs.shape[2]//args[7]) + (c//args[7]))],
                                           bgs[ws][n], n, r, c])
                ws = window_sizes[0][0]
                if len(win_s_dict[ws]) != 0:
                    err_indice = []
                    regress_imgs = []
                    bg_regress = []
                    ns = []
                    rs = []
                    cs = []
                    for i1, i2, i3, i4, i5 in win_s_dict[ws]:
                        regress_imgs.append(i1)
                        bg_regress.append(i2)
                        ns.append(i3)
                        rs.append(i4)
                        cs.append(i5)
                    pdfs, xs, ys, x_vars, y_vars, amps, rhos = image_regression(regress_imgs, bg_regress,
                                                                                (ws, ws), p0=args[6], decomp_n=args[8])
                    for err_i, (x_var, y_var, rho) in enumerate(zip(x_vars, y_vars, rhos)):
                        if x_var < 0 or y_var < 0 or x_var > 3*ws or y_var > 3*ws or rho > 1 or rho < -1:
                            err_indice.append(err_i)
                    if len(err_indice) == len(pdfs):
                        index += 1
                    else:
                        pdfs = np.delete(pdfs, err_indice, 0)
                        xs = np.delete(xs, err_indice, 0)
                        ys = np.delete(ys, err_indice, 0)
                        x_vars = np.delete(x_vars, err_indice, 0)
                        y_vars = np.delete(y_vars, err_indice, 0)
                        ns = np.delete(ns, err_indice, 0)
                        rs = np.delete(rs, err_indice, 0)
                        cs = np.delete(cs, err_indice, 0)
                        for n, r, c, dx, dy, pdf, x_var, y_var, rho, amp in zip(ns, rs, cs, xs, ys, pdfs, x_vars, y_vars, rhos, amps):
                            if r+dy <= -1 or r+dy >= imgs.shape[1] or c+dx <= -1 or c+dx >= imgs.shape[2]:
                                continue
                            row_coord = max(0, min(r+dy, imgs.shape[1]-1))
                            col_coord = max(0, min(c+dx, imgs.shape[2]-1))
                            coords[n].append([row_coord, col_coord])
                            reg_pdfs[n].append(pdf)
                            reg_infos[n].append([x_var, y_var, rho, amp])
                        del_indices = np.round(np.array([ns, rs+ys, cs+xs])).astype(int).T
                        extended_imgs_copy = extended_imgs.copy()
                        extended_imgs = subtract_pdf(extended_imgs, pdfs, del_indices, (ws, ws), bg_means, extend)
                        if np.sum(extended_imgs_copy) == np.sum(extended_imgs):
                            index += 1
            if len(indices) == 0 or forward_linkage[index] not in indices[:, 3]:
                index += 1


def image_cropping(extended_imgs: np.ndarray, extend, window_size, shift):
    cropped_imgs = []
    start_row = int(extend/2 - (window_size[1]-1)/2)
    end_row = extended_imgs.shape[1] - window_size[1] - start_row
    start_col = int(extend/2 - (window_size[0]-1)/2)
    end_col = extended_imgs.shape[2] - window_size[0] - start_col
    for r in range(start_row, end_row+1, shift):
        for c in range(start_col, end_col+1, shift):
            cropped_imgs.append(extended_imgs[:, r:r + window_size[1], c:c + window_size[0]])
    return cropped_imgs


def mapping(img1, img2, shift):
    if shift == 1:
        return img2.reshape(img1.shape[0], img1.shape[1], img1.shape[2])
    index = 0
    for row in range(0, img1.shape[1], shift):
        for col in range(0, img1.shape[2], shift):
            img1[:, row, col] = img2[:, index, 0]
            index += 1
    return img1


def empiric_cov_matrix(grid, qt):
    observations = qt * grid
    nbs = np.sum(qt, axis=1)
    obv_mean = (np.sum(observations, axis=1) / nbs).reshape(observations.shape[0], 1, -1)
    obv_mean = np.ones(observations.shape) * obv_mean
    a = np.sqrt(qt) * grid - (obv_mean * np.sqrt(qt))
    estimated_cov = (a.transpose(0, 2, 1) @ a) / list((nbs-1).reshape(nbs.shape[0], 1, 1))
    return estimated_cov


def cov_matrix(grid, qt, x_means, y_means):
    grids = []
    for n in range(x_means.shape[0]):
        grids.append(grid)
    grids = np.array(grids)
    cov = np.sum(
        (qt.reshape(x_means.shape[0], -1) *
         (grids[:, :, 0] - x_means.reshape(-1, 1)) *
         (grids[:, :, 1] - y_means.reshape(-1, 1))), axis=1)
    return cov


def quantification(imgs, window_size, amp):
    if amp == 0:
        qt_imgs = imgs.reshape(imgs.shape[0], -1, 1)
    else:
        qt_imgs = (imgs * (10**amp)).astype(np.uint32).reshape(imgs.shape[0], -1, 1)
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


def kl_divergence2(cov1, cov2):
    a = np.linalg.inv(cov2) * cov1
    return 1./2 * (a[:, 0, 0] + a[:, 1, 1] - 2 + np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))


def kl_divergence(base, compares):
    base = abs(base)
    compares = abs(compares)
    base += 1e-8
    compares += 1e-8
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


def image_regression(imgs, bgs, window_size, p0, decomp_n, amp=0, repeat=5):
    imgs = np.array(imgs)
    bgs = np.array(bgs)
    qt_imgs, grid = quantification(imgs, window_size, amp)
    coefs = guo_algorithm(imgs, bgs, p0=p0, window_size=window_size, repeat=repeat, decomp_n=decomp_n)
    variables, err_indices = unpack_coefs(coefs, window_size)
    if len(err_indices) > 0:
        coefs = guo_algorithm(imgs, bgs, p0=p0, window_size=window_size, repeat=repeat+1, decomp_n=decomp_n)
        variables, err_indices = unpack_coefs(coefs, window_size)
    variables = np.array(variables).T
    cov_mat = np.array([variables[:, 0], variables[:, 4] * np.sqrt(variables[:, 0]) * np.sqrt(variables[:, 2]),
                        variables[:, 4] * np.sqrt(variables[:, 0] * np.sqrt(variables[:, 2])), variables[:, 2]]
                       ).T.reshape(variables.shape[0], 2, 2)
    pdfs = bi_variate_normal_pdf(grid, cov_mat, mu=np.array([0, 0]), normalization=False)
    pdfs = variables[:, 5].reshape(-1, 1) * pdfs + bgs
    for err_i in err_indices:
        variables[err_i][0] = -100
        variables[err_i][2] = -100
    return pdfs, variables[:, 1], variables[:, 3], variables[:, 0], variables[:, 2], variables[:, 5], variables[:, 4]


def matrix_decomp(matrix, q):
    ret_mat = []
    for x in range(0, len(matrix), q):
        ret_mat.append(matrix[x: min(x+q, len(matrix))])
    return ret_mat


def unpack_coefs(coefs, window_size):
    err_indices = []
    x_mu = []
    y_mu = []
    rho = coefs[:, 4] * np.sqrt(1/(4 * -abs(coefs[:, 0]) * -abs(coefs[:, 2])))
    k = 1 - rho**2
    x_var = abs(1/(-2 * coefs[:, 0] * k))
    y_var = abs(1/(-2 * coefs[:, 2] * k))
    for err_indice, (xvar_check, yvar_check, r) in enumerate(zip(x_var, y_var, rho)):
        if xvar_check < 0 or yvar_check < 0 or xvar_check > 3 * window_size[0] or yvar_check > 3 * window_size[1] or r < -1 or r > 1:
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


def pack_vars(vars, len_img):
    a = -1./(2 * vars[0] * (1 - vars[4]**2))
    b = vars[1] / ((1 - vars[4]**2) * vars[0]) - (vars[4] * vars[3]) / ((1 - vars[4]**2) * np.sqrt(vars[0]) * np.sqrt(vars[2]))
    c = -1. / (2 * vars[2] * (1 - vars[4]**2))
    d = vars[3] / ((1 - vars[4]**2) * vars[2]) - (vars[4] * vars[1]) / ((1 - vars[4]**2) * np.sqrt(vars[0]) * np.sqrt(vars[2]))
    e = vars[4] / ((1 - vars[4]**2) * np.sqrt(vars[0]) * np.sqrt(vars[2]))
    f = (-(vars[1]**2)/(2*(1-vars[4]**2)*vars[0]) - (vars[3]**2)/(2*(1-vars[4]**2)*vars[2]) + (vars[4]*vars[1]*vars[3])/((1-vars[4]**2)*np.sqrt(vars[0]) * np.sqrt(vars[2])) +
         np.log(1/(2*np.pi*np.sqrt(vars[0]) * np.sqrt(vars[2])*(np.sqrt(1-vars[4]**2)))))
    return [[a, b, c, d, e, f] for _ in range(len_img)]


def guo_algorithm(imgs, bgs, p0=None, window_size=(7, 7), repeat=5, decomp_n=2):
    k = 0
    nb_imgs = imgs.shape[0]
    if p0 is None:
        p0 = [1.5, 0., 1.5, 0., 0., 0.5]  # x_var, x_mu, y_var, y_mu, rho, amp
    coef_vals = np.array(pack_vars(nbList(p0), nb_imgs))
    imgs = imgs.reshape(imgs.shape[0], window_size[0], window_size[1])
    ## background for each crop image needed rather than background intensity for whole image.
    imgs = np.maximum(np.zeros(imgs.shape), imgs - bgs.reshape(-1, window_size[0], window_size[1])) + 1e-2
    yk_2 = imgs.astype(np.float64).copy()
    x_grid = (np.array([list(np.arange(-int(window_size[0]/2), int((window_size[0]/2) + 1), 1))] * window_size[1])
              .reshape(-1, window_size[0], window_size[1]))
    y_grid = (np.array([[y] * window_size[0] for y in range(-int(window_size[1]/2), int((window_size[1]/2) + 1), 1)])
              .reshape(-1, window_size[0], window_size[1]))
    while k < repeat:
        if k != 0:
            yk_2 = np.exp(coef_vals[:, 0].reshape(-1, 1, 1) * x_grid**2 + coef_vals[:, 1].reshape(-1, 1, 1) * x_grid +
                          coef_vals[:, 2].reshape(-1, 1, 1) * y_grid**2 + coef_vals[:, 3].reshape(-1, 1, 1) * y_grid +
                          coef_vals[:, 4].reshape(-1, 1, 1) * x_grid * y_grid + coef_vals[:, 5].reshape(-1, 1, 1))
        yk_2 *= yk_2
        coef1 = yk_2 * x_grid**4
        coef2 = yk_2 * x_grid**3
        coef3 = yk_2 * x_grid**2 * y_grid**2
        coef4 = yk_2 * x_grid**2 * y_grid
        coef5 = yk_2 * x_grid**3 * y_grid
        coef6 = yk_2 * x_grid**2
        coef7 = yk_2 * x_grid * y_grid**2
        coef8 = yk_2 * x_grid * y_grid
        coef9 = yk_2 * x_grid
        coef10 = yk_2 * y_grid**4
        coef11 = yk_2 * y_grid**3
        coef12 = yk_2 * x_grid * y_grid**3
        coef13 = yk_2 * y_grid**2
        coef14 = yk_2 * y_grid
        coef15 = yk_2
        coef_matrix = np.sum(
            np.array(
                [[coef1, coef2, coef3, coef4, coef5, coef6],
                 [coef2, coef6, coef7, coef8, coef4, coef9],
                 [coef3, coef7, coef10, coef11, coef12, coef13],
                 [coef4, coef8, coef11, coef13, coef7, coef14],
                 [coef5, coef4, coef12, coef7, coef3, coef8],
                 [coef6, coef9, coef13, coef14, coef8, coef15]]
            ), axis=(3, 4)).transpose(2, 0, 1)
        ans1 = x_grid ** 2 * yk_2 * np.log(imgs)
        ans2 = x_grid * yk_2 * np.log(imgs)
        ans3 = y_grid ** 2 * yk_2 * np.log(imgs)
        ans4 = y_grid * yk_2 * np.log(imgs)
        ans5 = x_grid * y_grid * yk_2 * np.log(imgs)
        ans6 = yk_2 * np.log(imgs)
        ans_matrix = np.sum(
            np.array(
                [[ans1], [ans2], [ans3], [ans4], [ans5], [ans6]]
            ), axis=(3, 4), dtype=np.float64).transpose(2, 0, 1)
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
            #x_matrix.extend(gauss_seidel(a_mat, b_mat, p0=coef_val.ravel(), iter=200))
        x_matrix = np.array(x_matrix).reshape(-1, 6)
        if np.allclose(coef_vals, x_matrix, rtol=1e-7):
            break
        coef_vals = x_matrix
        k += 1
    return coef_vals


def gauss_seidel(a, b, p0, iter=200, tol=1e-8):
    x = p0
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


def make_red_circles(original_imgs, circle_imgs, localized_xys):
    stacked_imgs = nbList()
    for img_n, coords in enumerate(localized_xys):
        xy_cum = []
        for center_coord in coords:
            x, y = int(round(center_coord[0])), int(round(center_coord[1]))
            if (x, y) in xy_cum:
                circle_imgs[img_n] = draw_cross(circle_imgs[img_n], x, y, (0, 0, 1))
            else:
                circle_imgs[img_n] = draw_cross(circle_imgs[img_n], x, y, (1, 0, 0))
            xy_cum.append((x, y))
        stacked_imgs.append(np.hstack((original_imgs[img_n], circle_imgs[img_n])))
    return stacked_imgs


def visualilzation(output_dir, images, localized_xys):
    orignal_imgs_3ch = np.array([images.copy(), images.copy(), images.copy()])
    orignal_imgs_3ch = np.ascontiguousarray(np.moveaxis(orignal_imgs_3ch, 0, 3))
    circle_imgs = orignal_imgs_3ch.copy()
    stacked_img = np.array(make_red_circles(orignal_imgs_3ch, circle_imgs, localized_xys))
    tifffile.imwrite(f'{output_dir}_locvideo.tif', data=(stacked_img * 255).astype(np.uint8), imagej=True)


def background(imgs, window_sizes, alpha):
    bins = 0.01
    bgs = {}
    bg_means = []
    bg_stds = []
    bg_intensities = (imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) * 100).astype(np.uint8) / 100
    max_itensities = np.max(imgs, axis=(1, 2))
    mean_intensities = np.mean(imgs, axis=(1, 2))
    for i in range(len(bg_intensities)):
        args = np.arange(len(bg_intensities[i]))
        post_mask_args = args.copy()
        for _ in range(3):
            it_hist, bin_width = np.histogram(bg_intensities[i][post_mask_args],
                                              bins=np.arange(0, np.max(bg_intensities[i][post_mask_args]) + bins, bins))
            mask_sums_mode = (np.argmax(it_hist) * bins + (bins / 2))
            mask_std = np.std(bg_intensities[i][post_mask_args])
            post_mask_args = np.array([arg for arg, val in zip(args, bg_intensities[i]) if
                                       (mask_sums_mode - 3. * mask_std) < val < (mask_sums_mode + 3. * mask_std)])
        it_data = bg_intensities[i][post_mask_args]
        bg_means.append(np.mean(it_data))
        bg_stds.append(np.std(it_data))
    bg_means = np.array(bg_means)
    bg_stds = np.array(bg_stds)

    #for xxx in range(imgs.shape[0]):
    #    print(f'{xxx}: {bg_means[xxx]}, {bg_stds[xxx]}, {max_itensities[xxx]}, {mean_intensities[xxx]}')

    for window_size in window_sizes:
        bg = np.ones((bg_intensities.shape[0], window_size[0] * window_size[1]))
        bg *= bg_means.reshape(-1, 1)
        bgs[window_size[0]] = bg
    return bgs, bg_stds / bg_means / alpha


def intensity_distribution(images, reg_pdfs, xy_coords, reg_infos, sigma=3.5):
    new_pdfs = []
    new_coords = []
    new_infos = []
    for img_n, (pdfs, xy_coord, infos) in enumerate(zip(reg_pdfs, xy_coords, reg_infos)):
        if len(pdfs) < 1:
            continue
        new_pdf_tmp = pdfs.copy()
        new_xy_coord_tmp = xy_coord.copy()
        new_reg_tmp = infos.copy()
        max_pdf_vals = []

        for pdf, xy, info in zip(pdfs, xy_coord, infos):
            #max_pdf_vals.append(pdf[int((pdf.shape[0] - 1)/2)])  # - bgs[int(np.sqrt(pdf.shape[0]))][0][0]
            max_pdf_vals.append(np.max(images[img_n,
                                       max(0, int(np.round(xy[0])) - 1): min(images[img_n].shape[0], int(np.round(xy[0])) + 2),
                                       max(0, int(np.round(xy[1])) - 1): min(images[img_n].shape[1], int(np.round(xy[1])) + 2)]))
            #max_pdf_vals.append(info[3])  # - bgs[int(np.sqrt(pdf.shape[0]))][0][0]

        max_pdf_vals = np.array(max_pdf_vals)
        bin_edgs = np.arange(0, np.max(max_pdf_vals) + 0.05, 0.025)
        max_pdf_vals_hist = np.histogram(max_pdf_vals, bins=bin_edgs)
        mode_sigma = (bin_edgs[:-1] + 0.025)[np.argmax(max_pdf_vals_hist[0])] + sigma * np.std(max_pdf_vals)

        """
        print(mode_sigma)
        plt.figure()
        plt.plot(max_pdf_vals_hist[1][:-1], max_pdf_vals_hist[0])
        plt.show()
        """

        for i, max_pdf_val in enumerate(max_pdf_vals):
            if max_pdf_val > mode_sigma:
                new_pdf_tmp.append(pdfs[i])
                new_xy_coord_tmp.append(xy_coord[i])
                new_reg_tmp.append(infos[i])
        new_pdfs.append(new_pdf_tmp)
        new_coords.append(new_xy_coord_tmp)
        new_infos.append(new_reg_tmp)
    return new_pdfs, new_coords, new_infos


def params_gen(lowest, highest):
    assert type(lowest) is int and type(highest) is int
    if lowest % 2 == 0:
        lowest += 1
    if highest % 2 == 0:
        highest += 1
    mid = int((highest + lowest) / 2)
    if mid % 2 == 0: mid += 1
    if lowest == highest:
        bin_winsizes = [(lowest, lowest)]
        multi_winsizes = [(lowest, lowest)]
        bin_radius = [(lowest // 2) / 2.]
        multi_radius = [(lowest // 2) / 2.]
    elif mid == highest:
        bin_winsizes = [(lowest, lowest), (highest, highest)]
        multi_winsizes = [(lowest, lowest), (highest, highest)]
        bin_radius = [((r[0]//2) / 2.) for r in bin_winsizes]
        multi_radius = [((r[0]//2) / 2.) for r in multi_winsizes]
    else:
        bin_winsizes = [(lowest, lowest), (highest, highest)]
        multi_winsizes = [(lowest, lowest), (mid, mid), (highest, highest)]
        bin_radius = [((r[0]//2) / 2.) for r in bin_winsizes]
        multi_radius = [((r[0]//2) / 2.) for r in multi_winsizes]
    return bin_winsizes, bin_radius, multi_winsizes, multi_radius


def main_process(imgs, forward_gauss_grids, backward_gauss_grids, *args):
    args = list(args)
    before_time = timer()
    bgs, thresholds = background(imgs, window_sizes=args[3], alpha=args[9])
    if args[2] is None:
        args[2] = np.array([thresholds for _ in range(len(args[0]))]).T
    else:
        args[2] = np.ones((len(imgs), len(args[0]))) * args[2]
    if args[5] is None:
        args[5] = np.array([thresholds for _ in range(len(args[3]))]).T
    else:
        args[5] = np.ones((len(imgs), len(args[3]))) * args[5]
    print(f'{"background calcul":<35}:{(timer() - before_time):.2f}s')
    before_time = timer()
    xy_coord, pdf, info = localization(imgs, bgs, forward_gauss_grids, backward_gauss_grids, *args)
    print(f'{"localization calcul":<35}:{(timer() - before_time):.2f}s')
    return xy_coord, pdf, info


if __name__ == '__main__':
    #WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
    #WINDOWS_PATH = 'C:/Users/jwoo/Desktop'
    params = read_parameters('./config.txt')
    images = check_video_ext(params['localization']['VIDEO'])
    OUTPUT_DIR = params['localization']['OUTPUT_DIR']
    OUTPUT_LOC = f'{OUTPUT_DIR}/{params["localization"]["VIDEO"].split("/")[-1].split(".tif")[0]}'

    SIGMA = params['localization']['SIGMA']
    MIN_WIN = params['localization']['MIN_WIN']
    MAX_WIN = params['localization']['MAX_WIN']
    THRES_ALPHA = params['localization']['THRES_ALPHA']
    DEFLATION_LOOP_IN_BACKWARD = params['localization']['DEFLATION_LOOP_IN_BACKWARD']
    BINARY_THRESHOLDS = None
    MULTI_THRESHOLDS = None

    PARALLEL = params['localization']['PARALLEL']
    CORE = params['localization']['CORE']
    DIV_Q = params['localization']['DIV_Q']
    SHIFT = params['localization']['SHIFT']
    GAUSS_SEIDEL_DECOMP = params['localization']['GAUSS_SEIDEL_DECOMP']
    visualization = params['localization']['LOC_VISUALIZATION']
    P0 = [1.5, 0., 1.5, 0., 0., 0.5]

    xy_coords = []
    reg_pdfs = []
    reg_infos = []
    BINARY_COMP_WINSIZES, BINARY_RADIUS, MULTI_COMP_WINSIZES, MULTI_RADIUS = params_gen(MIN_WIN, MAX_WIN)
    forward_gauss_grids = gauss_psf(BINARY_COMP_WINSIZES, BINARY_RADIUS)
    backward_gauss_grids = gauss_psf(MULTI_COMP_WINSIZES, MULTI_RADIUS)

    start_time = timer()
    if PARALLEL:
        for div_q in range(0, len(images), CORE * DIV_Q):
            print(f'{div_q}/{len(images)} frame (parallelized)')
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executors = {i: None for i in range(CORE)}
                for cc in range(CORE):
                    if div_q + cc*DIV_Q < len(images):
                        future = executor.submit(main_process, images[div_q + cc*DIV_Q: div_q + cc*DIV_Q + DIV_Q],
                                                 forward_gauss_grids, backward_gauss_grids,
                                                 BINARY_COMP_WINSIZES, BINARY_RADIUS, BINARY_THRESHOLDS,
                                                 MULTI_COMP_WINSIZES, MULTI_RADIUS, MULTI_THRESHOLDS,
                                                 P0, SHIFT, GAUSS_SEIDEL_DECOMP, THRES_ALPHA, DEFLATION_LOOP_IN_BACKWARD)
                        executors[cc] = future
                for wait_ in executors:
                    if type(executors[wait_]) is concurrent.futures.Future:
                        xy_coord, pdf, info = executors[wait_].result()
                        xy_coords.extend(xy_coord)
                        reg_pdfs.extend(pdf)
                        reg_infos.extend(info)
    else:
        for div_q in range(0, len(images), DIV_Q):
            print(f'{div_q}/{len(images)} frame (non parallelized)')
            xy_coord, pdf, info = main_process(images[div_q:div_q+DIV_Q], forward_gauss_grids, backward_gauss_grids,
                                               BINARY_COMP_WINSIZES, BINARY_RADIUS, BINARY_THRESHOLDS,
                                               MULTI_COMP_WINSIZES, MULTI_RADIUS, MULTI_THRESHOLDS,
                                               P0, SHIFT, GAUSS_SEIDEL_DECOMP, THRES_ALPHA, DEFLATION_LOOP_IN_BACKWARD)
            xy_coords.extend(xy_coord)
            reg_pdfs.extend(pdf)
            reg_infos.extend(info)

    #reg_pdfs, xy_coords, reg_infos = intensity_distribution(images, reg_pdfs, xy_coords, reg_infos, sigma=SIGMA)
    write_localization(OUTPUT_LOC, xy_coords, reg_pdfs, reg_infos)

    if visualization:
        print(f'Visualizing localizations...')
        visualilzation(OUTPUT_LOC, images, xy_coords)
    print(f'{"Total time":<35}:{(timer() - start_time):.2f}s')
