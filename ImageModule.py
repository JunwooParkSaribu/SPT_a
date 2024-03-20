import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
from tifffile import TiffFile
from PIL import Image


def read_tif(filepath):
    normalized_imgs = []
    with TiffFile(filepath) as tif:
        imgs = tif.asarray()
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata

    nb_tif = imgs.shape[0]
    y_size = imgs.shape[1]
    x_size = imgs.shape[2]

    s_min = np.min(np.min(imgs, axis=(1, 2)))
    s_max = np.max(np.max(imgs, axis=(1, 2)))

    #modes = scipy.stats.mode(imgs.reshape(nb_tif, y_size*x_size), axis=1, keepdims=False)[0]
    for i, img in enumerate(imgs):
        img = (img - s_min) / (s_max - s_min)
        normalized_imgs.append(img)

    normalized_imgs = np.array(normalized_imgs, dtype=np.double)
    return normalized_imgs


def read_single_tif(filepath, ch3=True):
    with TiffFile(filepath) as tif:
        imgs = tif.asarray()
        if len(imgs.shape) >= 3:
            imgs = imgs[0]
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
        tag = tif.pages[0].tags

    y_size = imgs.shape[0]
    x_size = imgs.shape[1]
    s_mins = np.min(imgs)
    s_maxima = np.max(imgs)
    signal_maxima_avg = np.mean(s_maxima)
    zero_base = np.zeros((y_size, x_size), dtype=np.uint8)
    one_base = np.ones((y_size, x_size), dtype=np.uint8)
    #img = img - mode
    #img = np.maximum(img, zero_base)
    imgs = (imgs - s_mins) / (s_maxima - s_mins)
    #img = np.minimum(img, one_base)
    normalized_imgs = np.array(imgs * 255, dtype=np.uint8)
    if ch3 is False:
        return normalized_imgs
    img_3chs = np.array([np.zeros(normalized_imgs.shape), normalized_imgs, np.zeros(normalized_imgs.shape)]).astype(np.uint8)
    img_3chs = np.moveaxis(img_3chs, 0, 2)
    return img_3chs


def stack_tif(filename, normalized_imgs):
    tifffile.imwrite(filename, normalized_imgs)


def scatter_optimality(trajectory_list):
    plt.figure()
    scatter_x = []
    scatter_y = []
    scatter_color = []
    for traj in trajectory_list:
        if traj.get_optimality() is not None:
            scatter_x.append(traj.get_index())
            scatter_y.append(traj.get_optimality())
            scatter_color.append(traj.get_color())
    plt.scatter(scatter_x, scatter_y, c=scatter_color, s=5, alpha=0.7)
    plt.savefig('entropy_scatter.png')


def make_image(output, trajectory_list, cutoff=0, pixel_shape=(512, 512), amp=1, add_index=True, add_time=True):
    img = np.zeros((pixel_shape[0] * (10**amp), pixel_shape[1] * (10**amp), 3), dtype=np.uint8)
    for traj in trajectory_list:
        if traj.get_trajectory_length() >= cutoff:
            xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                           for x, y, _ in traj.get_positions()], np.int32)
            img_poly = cv2.polylines(img, [xx],
                                     isClosed=False,
                                     color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                            int(traj.get_color()[2] * 255)),
                                     thickness=1)
    if add_index:
        for traj in trajectory_list:
            if traj.get_trajectory_length() >= cutoff:
                xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                               for x, y, _ in traj.get_positions()], np.int32)
                cv2.putText(img, f'{  traj.get_index()}', org=xx[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                   int(traj.get_color()[2] * 255)))
    if add_time:
        for traj in trajectory_list:
            if traj.get_trajectory_length() >= cutoff:
                xx = np.array([[int(x * (10**amp)), int(y * (10**amp))]
                               for x, y, _ in traj.get_positions()], np.int32)
                cv2.putText(img, f'[{traj.get_times()[0]},{traj.get_times()[-1]}]',
                            org=[xx[0][0], xx[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                   int(traj.get_color()[2] * 255)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output, img)


def make_image_seqs2(*trajectory_lists, output_dir, time_steps, cutoff=0, original_shape=(512, 512),
                    target_shape=(512, 512), amp=0, add_index=True):
    """
    Use:
    make_image_seqs(gt_list, trajectory_list, output_dir=output_img, time_steps=time_steps, cutoff=1,
    original_shape=(images.shape[1], images.shape[2]), target_shape=(1536, 1536), add_index=True)
    """
    img_origin = np.zeros((target_shape[0] * (10**amp), target_shape[1] * (10**amp), 3), dtype=np.uint8)
    result_stack = []
    x_amp = img_origin.shape[0] / original_shape[0]
    y_amp = img_origin.shape[1] / original_shape[1]
    for frame in time_steps:
        img_stack = []
        for trajectory_list in trajectory_lists:
            img = img_origin.copy()
            for traj in trajectory_list:
                times = traj.get_times()
                if times[-1] < frame - 2:
                    continue
                indices = [i for i, time in enumerate(times) if time <= frame]
                if traj.get_trajectory_length() >= cutoff:
                    xy = np.array([[int(x * x_amp), int(y * y_amp)]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    font_scale = 0.1 * x_amp
                    img_poly = cv2.polylines(img, [xy],
                                             isClosed=False,
                                             color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                                    int(traj.get_color()[2] * 255)),
                                             thickness=1)
                    for x, y in xy:
                        cv2.circle(img, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    if len(indices) > 0:
                        cv2.putText(img, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                    org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                    color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                           int(traj.get_color()[2] * 255)))
                        if add_index:
                            cv2.putText(img, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                               int(traj.get_color()[2] * 255)))
            img[:, -1, :] = 255
            img_stack.append(img)
        hstacked_img = np.hstack(img_stack)
        result_stack.append(hstacked_img)
    result_stack = np.array(result_stack)
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)


def make_image_seqs(trajectory_list, output_dir, img_stacks, time_steps, cutoff=2,
                    add_index=True, local_img=None, gt_trajectory=None):
    if np.mean(img_stacks) < 0.35:
        bright_ = 1
    else:
        bright_ = 0

    if img_stacks.shape[1] * img_stacks.shape[2] < 512 * 512:
        upscailing_factor = 2  # int(512 / img_stacks.shape[1])
    else:
        upscailing_factor = 1
    result_stack = []
    for img, frame in zip(img_stacks, time_steps):
        img = cv2.resize(img, (img.shape[0]*upscailing_factor, img.shape[1]*upscailing_factor),
                         interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = np.array([img, img, img])
            img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        img_org = img.copy()
        if local_img is not None:
            local_img = img_org.copy()
            for traj in trajectory_list:
                times = traj.get_times()
                if frame in times:
                    indices = [i for i, time in enumerate(times) if time == frame]
                    xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    if local_img[xy[0][0], xy[0][1], 0] == 1 and local_img[xy[0][0], xy[0][1], 1] == 0 and local_img[xy[0][0], xy[0][1], 2] == 0:
                        local_img = draw_cross(local_img, xy[0][0], xy[0][1], (0, 0, 1))
                    else:
                        local_img = draw_cross(local_img, xy[0][0], xy[0][1], (1, 0, 0))
            local_img[:, -1, :] = 1

        if bright_:
            overlay = np.zeros(img.shape)
        else:
            overlay = np.ones(img.shape)
        for traj in trajectory_list:
            times = traj.get_times()
            if times[-1] < frame:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]
            if traj.get_trajectory_length() >= cutoff:
                xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                               for x, y, _ in traj.get_positions()[indices]], np.int32)
                font_scale = 0.1 * 2
                img_poly = cv2.polylines(overlay, [xy],
                                         isClosed=False,
                                         color=(traj.get_color()[0],
                                                traj.get_color()[1],
                                                traj.get_color()[2]),
                                         thickness=1)
                if len(indices) > 0:
                    if add_index:
                        cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                    org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(traj.get_color()[0],
                                           traj.get_color()[1],
                                           traj.get_color()[2]))
                        cv2.putText(overlay, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(traj.get_color()[0],
                                           traj.get_color()[1],
                                           traj.get_color()[2]))
        img_org[:, -1, :] = 1
        if bright_:
            overlay = img_org + overlay
        else:
            overlay = img_org * overlay
        overlay = np.minimum(np.ones_like(overlay), overlay)
        if local_img is not None:
            hstacked_img = np.hstack((local_img, overlay))
        else:
            hstacked_img = overlay

        if gt_trajectory is not None:
            overlay = img.copy()
            for traj in gt_trajectory:
                times = traj.get_times()
                if times[-1] < frame:
                    continue
                indices = [i for i, time in enumerate(times) if time <= frame]
                if traj.get_trajectory_length() >= cutoff:
                    xy = np.array([[int(np.around(x * upscailing_factor)), int(np.around(y * upscailing_factor))]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    font_scale = 0.1 * 2
                    img_poly = cv2.polylines(overlay, [xy],
                                             isClosed=False,
                                             color=(traj.get_color()[0],
                                                    traj.get_color()[1],
                                                    traj.get_color()[2]),
                                             thickness=1)
                    if len(indices) > 0:
                        if add_index:
                            cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                        org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(traj.get_color()[0],
                                               traj.get_color()[1],
                                               traj.get_color()[2]))
                            cv2.putText(overlay, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(traj.get_color()[0],
                                               traj.get_color()[1],
                                               traj.get_color()[2]))
            hstacked_img[:, -1, :] = 1
            hstacked_img = np.hstack((hstacked_img, overlay))
        result_stack.append(hstacked_img)
    result_stack = (np.array(result_stack) * 255).astype(np.uint8)
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)


def draw_cross(img, row, col, color):
    comb = [[row-2, col], [row-1, col], [row, col], [row+1, col], [row+2, col], [row, col-2], [row, col-1], [row, col+1], [row, col+2]]
    for r, c in comb:
        if 0 <= r < img.shape[0] and 0 <= c < img.shape[1]:
            for i, couleur in enumerate(color):
                if couleur >= 1:
                    img[r, c, i] = 1
                else:
                    img[r, c, i] = 0
    return img


def compare_two_localization_visual(output_dir, images, localized_xys_1, localized_xys_2):
    orignal_imgs_3ch = np.array([images.copy(), images.copy(), images.copy()])
    orignal_imgs_3ch = np.ascontiguousarray(np.moveaxis(orignal_imgs_3ch, 0, 3))
    original_imgs_3ch_2 = orignal_imgs_3ch.copy()
    stacked_imgs = []
    frames = np.sort(list(localized_xys_1.keys()))
    for img_n in frames:
        for center_coord in localized_xys_1[img_n]:
            if (center_coord[0] > orignal_imgs_3ch.shape[1] or center_coord[0] < 0
                    or center_coord[1] > orignal_imgs_3ch.shape[2] or center_coord[1] < 0):
                print("ERR")
                print(img_n, 'row:', center_coord[0], 'col:', center_coord[1])
            x, y = int(round(center_coord[1])), int(round(center_coord[0]))
            orignal_imgs_3ch[img_n-1][x][y][0] = 1
            orignal_imgs_3ch[img_n-1][x][y][1] = 0
            orignal_imgs_3ch[img_n-1][x][y][2] = 0

        for center_coord in localized_xys_2[img_n]:
            if (center_coord[0] > original_imgs_3ch_2.shape[1] or center_coord[0] < 0
                    or center_coord[1] > original_imgs_3ch_2.shape[2] or center_coord[1] < 0):
                print("ERR")
                print(img_n, 'row:', center_coord[0], 'col:', center_coord[1])
            x, y = int(round(center_coord[1])), int(round(center_coord[0]))
            original_imgs_3ch_2[img_n-1][x][y][0] = 1
            original_imgs_3ch_2[img_n-1][x][y][1] = 0
            original_imgs_3ch_2[img_n-1][x][y][2] = 0
        stacked_imgs.append(np.hstack((orignal_imgs_3ch[img_n-1], original_imgs_3ch_2[img_n-1])))
    stacked_imgs = np.array(stacked_imgs)
    tifffile.imwrite(f'{output_dir}/local_comparison.tif', data=(stacked_imgs * 255).astype(np.uint8), imagej=True)


"""
imgs = []
for i in range(100):
    if i < 10:
        i = '00'+str(i)
    else:
        i = '0' + str(i)
    f = f'/home/junwoo/SPT_a/simulated_data/MICROTUBULE/MICROTUBULE snr 4 density mid/MICROTUBULE snr 4 density mid t{i} z0.tif'
    imgs.append(read_single_tif(f, ch3=False))
stack_tif(filename=f'microtubule_4_mid.tif', normalized_imgs=imgs)
exit(1)
"""
