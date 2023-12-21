import matplotlib.pyplot as plt
import numpy as np
import cv2
import tifffile
import scipy
from PIL import Image


def read_tif(filepath):
    normalized_imgs = []
    imgs = tifffile.imread(filepath).astype(np.int16)

    nb_tif = imgs.shape[0]
    y_size = imgs.shape[1]
    x_size = imgs.shape[2]

    s_mins = np.min(imgs, axis=(1, 2))
    s_maxima = np.max(imgs, axis=(1, 2))

    modes = scipy.stats.mode(imgs.reshape(nb_tif, y_size*x_size), axis=1, keepdims=False)[0]
    zero_base = np.zeros((y_size, x_size), dtype=np.uint8)
    one_base = np.ones((y_size, x_size), dtype=np.uint8)

    for i, (img, mode, s_min, s_max) in enumerate(zip(imgs, modes, s_mins, s_maxima)):
        img = (img - s_min) / (s_max - s_min)
        normalized_imgs.append(img * 255)

    normalized_imgs = np.array(normalized_imgs, dtype=np.uint8)
    return normalized_imgs


def read_single_tif(filepath):
    imgs = tifffile.imread(filepath).astype(np.int16)
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
    return normalized_imgs


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


def make_image_seqs(*trajectory_lists, output_dir, time_steps, cutoff=0, original_shape=(512, 512),
                    target_shape=(512, 512), amp=0, add_index=True):
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


