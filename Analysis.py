import itertools
import numpy as np
import scipy
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.stats import gmean
from scipy.spatial import KDTree
from numba import jit, njit, cuda, vectorize, int32, int64, float32, float64
from numba.typed import Dict, List
from numba.core import types
from ImageModule import read_tif, read_single_tif, make_image, make_image_seqs, stack_tif, compare_two_localization_visual
from TrajectoryObject import TrajectoryObj
from XmlModule import xml_to_object, write_xml, read_xml, andi_gt_to_xml
from FileIO import write_trajectory, read_trajectory, read_mosaic, read_localization
from timeit import default_timer as timer
from skimage.restoration import denoise_tv_chambolle


WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
WINDOWS_PATH = 'C:/Users/jwoo/Desktop'


def plot_diff_coefs(trajectory_list):
    FIGSIZE = (9, 8)
    time_interval = 1
    weights = [1, 2, 10]
    nrow = len(weights) + 1
    t_range = [0, 200]
    ncol = 4
    for traj in trajectory_list:
        if traj.get_index() == 18:
            denoised = [[[] for _ in range(ncol)] for _ in range(nrow)]
            diff_coefs = traj.get_diffusion_coefs(time_interval=time_interval, t_range=t_range)
            angles = traj.get_trajectory_angles(time_interval=time_interval, t_range=t_range)
            MSD = traj.get_msd(time_interval=time_interval, t_range=t_range)

            #print(np.mean(diff_coefs))
            #rescaled_range(MSD)
            #print(MSD[1])
            #print(4 * 0.9096942927464597 * (1 ** 1.499041665996179))
            #exit(1)
            alphas = MSD / (4 * np.insert(diff_coefs, 0, 1e-5))

            denoised[0][0] = diff_coefs
            denoised[0][1] = angles
            denoised[0][2] = MSD
            denoised[0][3] = alphas

            for r, weight in enumerate(weights):
                denoised[r+1][0] = denoise_tv_chambolle(diff_coefs, weight=weight)
                denoised[r+1][1] = denoise_tv_chambolle(angles, weight=weight)
                denoised[r+1][2] = denoise_tv_chambolle(MSD, weight=weight)

            fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE)
            for row in range(nrow):
                for col in range(ncol):
                    if len(denoised[row][col]) > 0:
                        axs[row, col].plot(np.arange(len(denoised[row][col])), denoised[row][col])
                        if col == 0 or col == 1:
                            axs[row, col].set_ylim([0, 10])
                        #if col == 3:
                        #    axs[row, col].set_ylim([0, 2])
                        axs[row, col].set_xlim([0, 200])
            plt.show()


def rescaled_range(data):
    rescaled_rng = []
    rescaled_nb_obvs = []
    data = np.array(data)
    for n in range(1, len(data) + 1):
        dt = data[:n]
        m = np.mean(dt)
        Y_t = dt - m
        Z_t = np.cumsum(dt)
        R_n = np.max(Z_t) - np.min(Z_t)
        S_n = np.sqrt((1/len(dt)) * np.sum(Y_t ** 2))
        if S_n <= 0:
            continue
        else:
            rescaled_rng.append(R_n / S_n)
            rescaled_nb_obvs.append(n)

    #X = np.arange(len(rescaled_rng))
    X = np.array([[x, 1] for x in np.log(np.array(rescaled_nb_obvs))])
    Y = np.array(np.log(rescaled_rng))
    W = np.linalg.inv((X.T @ X))@X.T@Y

    print(W)

    plt.figure()
    plt.plot(np.array(rescaled_nb_obvs), rescaled_rng)
    plt.plot(X[:, 0], Y)
    plt.plot(X[:, 0], X[:, 0] * W[0] + W[1])
    plt.show()


if __name__ == '__main__':
    #input_tif = f'{WINDOWS_PATH}/20220217_aa4_cel8_no_ir.tif'
    #input_tif = f'{WINDOWS_PATH}/videos_fov_0.tif'
    #input_tif = f'{WINDOWS_PATH}/single1.tif'
    #input_tif = f'{WINDOWS_PATH}/multi3.tif'
    #input_tif = f'{WINDOWS_PATH}/immobile_traps1.tif'
    #input_tif = f'{WINDOWS_PATH}/dimer1.tif'
    #input_tif = f'{WINDOWS_PATH}/confinement1.tif'
    input_tif = f'andi_data/multi3.tif'

    output_img_fname = f'{WINDOWS_PATH}/mymethod.tif'
    #input_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0_singlestate.csv'
    #input_trj_fname = f'{WINDOWS_PATH}/multi3.csv'
    input_trj_fname = f'andi_data/multi3.csv'
    #gt_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0.csv'
    gt_trj_fname = f'andi_data/trajs_fov_0_multi3.csv'

    #images = read_tif(input_tif)[1:]
    trajectories = read_trajectory(input_trj_fname)
    andi_gt_list = read_trajectory(gt_trj_fname, andi_gt=True)

    time_step_min = 9999
    time_step_max = -1
    for traj in trajectories:
        time_step_min = min(traj.get_times()[0], time_step_min)
        time_step_max = max(traj.get_times()[-1], time_step_max)
    time_steps = np.arange(time_step_min, time_step_max+1, 1, dtype=np.uint32)

    #make_image_seqs(trajectories, output_dir=output_img_fname, img_stacks=images, time_steps=time_steps, cutoff=2,
    #                add_index=True, local_img=True, gt_trajectory=andi_gt_list)

    plot_diff_coefs(andi_gt_list)
