import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
from tifffile import TiffFile


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


def read_single_tif(filepath):
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


def make_image_seqs(trajectory_list, output_dir, img_stacks, time_steps, cutoff=2, add_index=True, gt_trajectory=None):
    alpha = 1.
    result_stack = []
    for img, frame in zip(img_stacks, time_steps):
        img = np.array([img, img, img])
        img = np.moveaxis(img, 0, 2)
        img = np.ascontiguousarray(img)
        img_org = img.copy()
        overlay = img.copy()
        for traj in trajectory_list:
            times = traj.get_times()
            if times[-1] < frame - 1:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]
            if traj.get_trajectory_length() >= cutoff:
                xy = np.array([[int(np.around(x)), int(np.around(y))]
                               for x, y, _ in traj.get_positions()[indices]], np.int32)
                font_scale = 0.1 * 2
                img_poly = cv2.polylines(overlay, [xy],
                                         isClosed=False,
                                         color=(int(traj.get_color()[0] * 255),
                                                int(traj.get_color()[1] * 255),
                                                int(traj.get_color()[2] * 255)),
                                         thickness=1)
                if len(indices) > 0:
                    if add_index:
                        cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                    org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                           int(traj.get_color()[2] * 255)))
                        cv2.putText(overlay, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                               int(traj.get_color()[2] * 255)))
        img_org[:, -1, :] = 1
        image_alpha = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        hstacked_img = np.hstack((img_org, image_alpha))

        if gt_trajectory is not None:
            overlay = img.copy()
            for traj in gt_trajectory:
                times = traj.get_times()
                if times[-1] < frame - 1:
                    continue
                indices = [i for i, time in enumerate(times) if time <= frame]
                if traj.get_trajectory_length() >= cutoff:
                    xy = np.array([[int(np.around(x)), int(np.around(y))]
                                   for x, y, _ in traj.get_positions()[indices]], np.int32)
                    font_scale = 0.1 * 2
                    img_poly = cv2.polylines(overlay, [xy],
                                             isClosed=False,
                                             color=(int(traj.get_color()[0] * 255),
                                                    int(traj.get_color()[1] * 255),
                                                    int(traj.get_color()[2] * 255)),
                                             thickness=1)
                    if len(indices) > 0:
                        if add_index:
                            cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                        org=[xy[0][0], xy[0][1] + 12], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                               int(traj.get_color()[2] * 255)))
                            cv2.putText(overlay, f'{traj.get_index()}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_scale,
                                        color=(int(traj.get_color()[0] * 255), int(traj.get_color()[1] * 255),
                                               int(traj.get_color()[2] * 255)))
            hstacked_img[:, -1, :] = 1
            image_alpha = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            hstacked_img = np.hstack((hstacked_img, image_alpha))
        result_stack.append(hstacked_img)
    result_stack = (np.array(result_stack) * 255).astype(np.uint8)
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)


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
    f = f'/home/junwoo/MT/simulated_data/VESICLE/VESICLE snr 7 density mid/VESICLE snr 7 density mid t{i} z0.tif'
    imgs.append(read_single_tif(f))
stack_tif(filename=f'vesicle_7_mid.tif', normalized_imgs=imgs)
exit(1)
"""


