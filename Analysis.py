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
    for traj in trajectory_list:
        diff_coefs = np.array(traj.get_diffusion_coefs())
        if len(diff_coefs) > 50:
            print(traj.get_index())
            tv_denoised_1 = denoise_tv_chambolle(diff_coefs, weight=1)
            tv_denoised_10 = denoise_tv_chambolle(diff_coefs, weight=10)

            plt.figure()
            plt.plot(np.arange(len(diff_coefs)), diff_coefs)
            plt.ylim([0, 10])
            plt.xlim([0, 200])

            plt.figure()
            plt.plot(np.arange(len(diff_coefs)), tv_denoised_1)
            plt.ylim([0, 10])
            plt.xlim([0, 200])

            plt.figure()
            plt.plot(np.arange(len(diff_coefs)), tv_denoised_10)
            plt.ylim([0, 10])
            plt.xlim([0, 200])
            plt.show()


if __name__ == '__main__':
    #input_tif = f'{WINDOWS_PATH}/20220217_aa4_cel8_no_ir.tif'
    #input_tif = f'{WINDOWS_PATH}/videos_fov_0.tif'
    #input_tif = f'{WINDOWS_PATH}/single1.tif'
    input_tif = f'{WINDOWS_PATH}/multi3.tif'
    #input_tif = f'{WINDOWS_PATH}/immobile_traps1.tif'
    #input_tif = f'{WINDOWS_PATH}/dimer1.tif'
    #input_tif = f'{WINDOWS_PATH}/confinement1.tif'

    output_img_fname = f'{WINDOWS_PATH}/mymethod.tif'
    #input_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0_singlestate.csv'
    input_trj_fname = f'{WINDOWS_PATH}/multi3.csv'
    gt_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0.csv'

    images = read_tif(input_tif)[1:]
    trajectories = read_trajectory(input_trj_fname)
    andi_gt_list = read_trajectory(gt_trj_fname, andi_gt=True)

    time_step_min = 9999
    time_step_max = -1
    for traj in trajectories:
        time_step_min = min(traj.get_times()[0], time_step_min)
        time_step_max = max(traj.get_times()[-1], time_step_max)
    time_steps = np.arange(time_step_min, time_step_max+1, 1, dtype=np.uint32)

    #make_image_seqs(trajectories, output_dir=output_img_fname, img_stacks=images, time_steps=time_steps, cutoff=2,
    #                add_index=True, local_img=True, gt_trajectory=None)

    plot_diff_coefs(trajectories)
