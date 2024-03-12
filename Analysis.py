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
#from Hurst import hurst
from scipy import stats, optimize
from hurst import compute_Hc, random_walk
from stochastic.processes.noise import FractionalGaussianNoise as FGN
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_challenge import label_continuous_to_list
import stochastic



WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
WINDOWS_PATH = 'C:/Users/jwoo/Desktop'


def power_fit(x, c, alpha):
    return c * (x ** alpha)


def disp_fbm(alpha: float,
             D: float,
             T: int,
             deltaT: int = 1):
    ''' Generates normalized Fractional Gaussian noise. This means that, in
    general:
    $$
    <x^2(t) > = 2Dt^{alpha}
    $$

    and in particular:
    $$
    <x^2(t = 1)> = 2D
    $$

    Parameters
    ----------
    alpha : float in [0,2]
        Anomalous exponent
    D : float
        Diffusion coefficient
    T : int
        Number of displacements to generate
    deltaT : int, optional
        Sampling time

    Returns
    -------
    numpy.array
        Array containing T displacements of given parameters
    '''

    # Generate displacements
    disp = FGN(hurst=alpha / 2).sample(n=T)
    # Normalization factor
    disp *= np.sqrt(T) ** (alpha)
    # Add D
    disp *= np.sqrt(2 * D * deltaT)

    return disp


def samples_GHE(serie, tau):
    return np.abs(serie[tau:] - serie[:-tau])


def KSGHE(serie, p0):
    scaling_range = [2**n for n in range(int(np.log2(len(serie)))-2)]
    sample_t0 = samples_GHE(serie, tau=scaling_range[0])
    f=lambda h: np.sum([stats.ks_2samp(sample_t0, samples_GHE(serie, tau=tau) / (tau**h)).statistic for tau in scaling_range[1:]])
    w = optimize.fmin(f, x0=p0, disp=False)
    return w[0]


def plot_diff_coefs(trajectory_list, *args, t_range=None):
    changepoints = args[0]
    alphas = args[1]
    Ds = args[2]
    state_num = args[3]
    if t_range is None:
        t_range = [0, 10000]
    FIGSIZE = (14, 9)
    time_interval = 1
    weights = [1, 2, 10]
    nrow = len(weights) + 1
    ncol = 4

    for traj in trajectory_list:
        if traj.get_index() == 0:
            denoised = [[[] for _ in range(ncol)] for _ in range(nrow)]
            diff_coefs = traj.get_diffusion_coefs(time_interval=time_interval, t_range=t_range)
            angles = traj.get_trajectory_angles(time_interval=time_interval, t_range=t_range)
            MSD = traj.get_msd(time_interval=time_interval, t_range=t_range)

            c, alpha = optimize.curve_fit(power_fit, np.arange(len(MSD)), MSD)[0]
            print(f'C:{c}, alpha:{alpha}')

            #diff_coefs = np.cumsum(diff_coefs)
            #angles = np.cumsum(angles)
            #MSD = np.cumsum(MSD)

            denoised[0][0] = diff_coefs
            denoised[0][1] = angles
            denoised[0][2] = MSD
            denoised[0][3] = []

            for r, weight in enumerate(weights):
                denoised[r+1][0] = denoise_tv_chambolle(diff_coefs, weight=weight)
                denoised[r+1][1] = denoise_tv_chambolle(angles, weight=weight)
                denoised[r+1][2] = denoise_tv_chambolle(MSD, weight=weight)

            fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE)
            fig.suptitle(f'alphas:   {np.round(alphas,2)}\n'
                         f'Ds:       {np.round(Ds, 2)}\n'
                         f'State_num:{state_num}')
            axs[0, 0].title.set_text('Diff_coef')
            axs[0, 1].title.set_text('Angles')
            axs[0, 2].title.set_text('MSD')
            axs[0, 3].title.set_text('empty')

            for row in range(nrow):
                for col in range(ncol):
                    if len(denoised[row][col]) > 0:
                        y_plot_max = 8
                        axs[row, col].plot(np.arange(len(denoised[row][col])), denoised[row][col])
                        axs[row, col].set_xlim(t_range)

                        if col == 0:
                            axs[row, col].set_ylim([0, np.max(diff_coefs) + 5])
                            if changepoints is not None:
                                axs[row, col].vlines(changepoints, ymin=0, ymax=np.max(diff_coefs) + 5,
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 1:
                            axs[row, col].set_ylim([0, np.max(angles) + 3])
                            if changepoints is not None:
                                axs[row, col].vlines(changepoints, ymin=0, ymax=np.max(angles) + 3,
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 2:
                            axs[row, col].plot(np.arange(len(MSD)), power_fit(np.arange(len(MSD)), c, alpha))
                            if changepoints is not None:
                                axs[row, col].vlines(changepoints, ymin=0, ymax=np.max(MSD),
                                                     colors='red',
                                                     alpha=0.5)
            plt.show()


def rescaled_range(data):
    rescaled_rngs = []
    rescaled_nb_obvs = []
    data = np.array(data)
    min_obs = 1

    chunks = []
    step = 2
    obs_size = min_obs
    obs_sizes = []
    while 1:
        tmp = []
        chunks = []
        dt_size = len(data) // obs_size
        if dt_size <= 1:
            break
        obs_sizes.append(dt_size)

        for i in range(0, len(data), dt_size):
            if i+dt_size <= len(data):
                chunk = data[i:i + dt_size]
                chunks.append(chunk)
        obs_size *= step
        chunks = np.array(chunks)

        for chunk in chunks:
            m = np.mean(chunk)
            Y_t = chunk - m
            Z_n = np.cumsum(Y_t)
            R_n = np.max(Z_n) - np.min(Z_n)
            S_n = np.sqrt(1/len(chunk) * np.sum((chunk - m)**2))
            tmp.append(R_n / S_n)
        rescaled_rngs.append(np.mean(tmp))

    rescaled_rngs = rescaled_rngs[::-1]
    obs_sizes = obs_sizes[::-1]
    X = np.array([[x, 1] for x in np.log(np.array(obs_sizes))])
    Y = np.array(np.log(rescaled_rngs))
    W = np.linalg.inv((X.T @ X))@X.T@Y
    return W[0]

    #plt.figure()
    #plt.plot(np.array(obs_sizes), rescaled_rngs)
    #plt.plot(X[:, 0], Y)
    #plt.plot(X[:, 0], X[:, 0] * W[0] + W[1])
    #plt.show()


if __name__ == '__main__':
    hurst_exp = 0.5
    t_range = [0, 200]
    D = 0.1
    L = 1.5 * 128
    """
    trajs_model1, labels_model1 = models_phenom().single_state(N=2,
                                                              L=False,
                                                              T=200,
                                                              Ds=[0.1, 0],  # Mean and variance
                                                              alphas=hurst_exp * 2
                                                              )
    """
    print('---------')
    trajs_model2, labels_model2 = models_phenom().multi_state(N=2,
                                                              L=L,
                                                              T=200,
                                                              alphas=[1.2, 0.7],  # Fixed alpha for each state
                                                              Ds=[[0.1, 0.0], [0.1, 0.0]],
                                                              # Mean and variance of each state
                                                              M=[[0.98, 0.02], [0.02, 0.98]]
                                                              )

    trajs_model = trajs_model2
    trajs_label = labels_model2
    print('Tajectory shapes, label shape: ', trajs_model.shape, trajs_label.shape)
    changepoints, alphas, Ds, state_num = label_continuous_to_list(trajs_label[:, 0, :])
    print(f'change points: {changepoints}')
    print(f'alphas: {alphas}')
    print(f'Ds: {Ds}')
    print(f'State_nums: {state_num}')

    #xs = disp_fbm(alpha=hurst_exp * 2, D=0.1, T=1024)[t_trange[0]:t_trange[1]]
    #ys = disp_fbm(alpha=hurst_exp * 2, D=0.1, T=1024)[t_trange[0]:t_trange[1]]
    #pos = np.array([xs, ys]).T

    xs = trajs_model[:, 0, 0]
    ys = trajs_model[:, 0, 1]
    pos = np.array([trajs_model[:, 0, 0], trajs_model[:, 0, 1]]).T
    traj1 = TrajectoryObj(index=0)
    for t, (x, y) in enumerate(pos):
        traj1.add_trajectory_position(t, x, y, 0.0)
    MSD = traj1.get_msd(time_interval=1, t_range=t_range)
    diff_coefs = traj1.get_diffusion_coefs(time_interval=1, t_range=t_range)

    print(f'DIFF_COEF_MEAN: {np.mean(diff_coefs)}')

    print('pip hurst compute_Hc')
    print(compute_Hc(xs, kind='random_walk')[0])
    print(compute_Hc(ys, kind='random_walk')[0])
    print(compute_Hc(MSD, kind='random_walk')[0])
    print("-----------------------------")

    print("\nKSGHD")
    print(f'X: {KSGHE(xs, hurst_exp)}')
    print(f'Y: {KSGHE(ys, hurst_exp)}')
    print(f'MSD: {KSGHE(MSD, hurst_exp)}')
    print("-----------------------------")

    print("\nMY RSRange")
    print(f'X: {rescaled_range(xs)}')
    print(f'Y: {rescaled_range(ys)}')
    print(f'MSD: {rescaled_range(MSD)}')
    print("-----------------------------")

    print("\nKSGHD with Mine")
    print(f'X: {KSGHE(xs, rescaled_range(xs))}')
    print(f'Y: {KSGHE(ys, rescaled_range(ys))}')
    print(f'MSD: {KSGHE(MSD, rescaled_range(MSD))}')
    print("-----------------------------")

    plot_diff_coefs([traj1],
                    changepoints, alphas, Ds, state_num,
                    t_range=t_range)
    exit(1)


    #input_tif = f'{WINDOWS_PATH}/20220217_aa4_cel8_no_ir.tif'
    #input_tif = f'{WINDOWS_PATH}/videos_fov_0.tif'
    #input_tif = f'{WINDOWS_PATH}/single1.tif'
    #input_tif = f'{WINDOWS_PATH}/multi3.tif'
    #input_tif = f'{WINDOWS_PATH}/immobile_traps1.tif'
    #input_tif = f'{WINDOWS_PATH}/dimer1.tif'
    #input_tif = f'{WINDOWS_PATH}/confinement1.tif'
    input_tif = f'{WINDOWS_PATH}/multi3.tif'

    output_img_fname = f'{WINDOWS_PATH}/mymethod.tif'
    input_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0_singlestate.csv'
    #input_trj_fname = f'{WINDOWS_PATH}/multi3.csv'
    #input_trj_fname = f'{WINDOWS_PATH}/multi3.csv'
    gt_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0.csv'
    #gt_trj_fname = f'{WINDOWS_PATH}/trajs_fov_0_multi3.csv'

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
