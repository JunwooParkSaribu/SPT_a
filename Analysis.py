import itertools
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
from andi_datasets.utils_trajectories import plot_trajs
import stochastic


WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
WINDOWS_PATH = 'C:/Users/jwoo/Desktop'

def uncumulate(xs:np.ndarray):
    assert xs.ndim == 1
    uncum_list = [0.]
    for i in range(1, len(xs)):
        uncum_list.append(xs[i] - xs[i-1])
    return np.array(uncum_list)


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

"""
xs = disp_fbm(alpha=0.001, D=0.1, T=100)
xs1 = disp_fbm(alpha=0.5, D=0.1, T=100)
xs2 = disp_fbm(alpha=1.0, D=0.1, T=100)
xs3 = disp_fbm(alpha=1.5, D=0.1, T=100)
xs4 = disp_fbm(alpha=1.999, D=0.1, T=100)
print(xs[:20])
print(np.mean(xs), np.mean(xs1), np.mean(xs2), np.mean(xs3), np.mean(xs4))
print(np.var(xs), np.var(xs1), np.var(xs2), np.var(xs3), np.var(xs4))
print(np.std(xs), np.std(xs1), np.std(xs2), np.std(xs3), np.std(xs4))
"""

L = 1.5*128  # boundary box size
N = 2  # number of trajectory
T = 32  # length of trajectory

trajs_model1, labels_model1 = models_phenom().multi_state(N=N,
                                                        L=L,
                                                        T=T,
                                                        alphas=[0.2, 0.2],  # Fixed alpha for each state
                                                        Ds=[[0.1, 0.0], [0.1, 0.0]],# Mean and variance of each state
                                                        M=[[1.0, 0.0], [0.0, 1.0]]  # Transition matrix
                                                       )

trajs_model2, labels_model2 = models_phenom().multi_state(N=N,
                                                        L=L,
                                                        T=T,
                                                        alphas=[0.5, 0.5],  # Fixed alpha for each state
                                                        Ds=[[0.1, 0.0], [0.1, 0.0]],# Mean and variance of each state
                                                        M=[[1.0, 0.0], [0.0, 1.0]]  # Transition matrix
                                                       )

trajs_model3, labels_model3 = models_phenom().multi_state(N=N,
                                                        L=L,
                                                        T=T,
                                                        alphas=[1.0, 1.0],  # Fixed alpha for each state
                                                        Ds=[[0.1, 0.0], [0.1, 0.0]],# Mean and variance of each state
                                                        M=[[1.0, 0.0], [0.0, 1.0]]  # Transition matrix
                                                       )

trajs_model4, labels_model4 = models_phenom().multi_state(N=N,
                                                        L=L,
                                                        T=T,
                                                        alphas=[1.5, 1.5],  # Fixed alpha for each state
                                                        Ds=[[0.1, 0.0], [0.1, 0.0]],# Mean and variance of each state
                                                        M=[[1.0, 0.0], [0.0, 1.0]]  # Transition matrix
                                                       )

trajs_model5, labels_model5 = models_phenom().multi_state(N=N,
                                                        L=L,
                                                        T=T,
                                                        alphas=[1.9, 1.9],  # Fixed alpha for each state
                                                        Ds=[[0.1, 0.0], [0.1, 0.0]],# Mean and variance of each state
                                                        M=[[1.0, 0.0], [0.0, 1.0]]  # Transition matrix
                                                       )


def count_positive_negative(x):
    count = 0

    sign = 1
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] > 0:
            new_sign = 1
        else:
            new_sign = -1

        if sign != new_sign:
            count += 1

        sign = new_sign
    return count

xs1 = trajs_model1[:, 0, 0]
ys1 = trajs_model1[:, 0, 1]
xs2 = trajs_model2[:, 0, 0]
ys2 = trajs_model2[:, 0, 1]
xs3 = trajs_model3[:, 0, 0]
ys3 = trajs_model3[:, 0, 1]
xs4 = trajs_model4[:, 0, 0]
ys4 = trajs_model4[:, 0, 1]
xs5 = trajs_model5[:, 0, 0]
ys5 = trajs_model5[:, 0, 1]

xs1 = xs1 - float(xs1[0])
xs2 = xs2 - float(xs2[0])
xs3 = xs3 - float(xs3[0])
xs4 = xs4 - float(xs4[0])
xs5 = xs5 - float(xs5[0])

"""
xs1 = uncumulate(xs1) / T**0.2
xs2 = uncumulate(xs2) / T**0.5
xs3 = uncumulate(xs3)/ T**1
xs4 = uncumulate(xs4)/ T**1.5
xs5 = uncumulate(xs5)/ T**1.9
"""

#xs1 = np.cumsum(xs1)
#xs2 = np.cumsum(xs2)

print(np.mean(xs1), np.std(xs1), np.var(xs1), count_positive_negative(xs1) / len(xs1))
print(np.mean(xs2), np.std(xs2), np.var(xs2), count_positive_negative(xs2) / len(xs2))
print(np.mean(xs3), np.std(xs3), np.var(xs3), count_positive_negative(xs3) / len(xs3))
print(np.mean(xs4), np.std(xs4), np.var(xs4), count_positive_negative(xs4) / len(xs4))
print(np.mean(xs5), np.std(xs5), np.var(xs5), count_positive_negative(xs5) / len(xs5))


plt.figure()
plt.plot(np.arange(len(xs1)), xs1, label='1')
plt.plot(np.arange(len(xs2)), xs2, label='2')
plt.plot(np.arange(len(xs3)), xs3, label='3')
plt.plot(np.arange(len(xs4)), xs4, label='4')
plt.plot(np.arange(len(xs5)), xs5, label='5')
plt.legend()

xs1 = denoise_tv_chambolle(xs1, weight=0.1)
xs2 = denoise_tv_chambolle(xs2, weight=0.1)
xs3 = denoise_tv_chambolle(xs3, weight=0.1)
xs4 = denoise_tv_chambolle(xs4, weight=0.1)
xs5 = denoise_tv_chambolle(xs5, weight=0.1)

print("---------------------------------------------------------------------------------")
print(np.mean(xs1), np.std(xs1), np.var(xs1), count_positive_negative(xs1) / len(xs1))
print(np.mean(xs2), np.std(xs2), np.var(xs2), count_positive_negative(xs2) / len(xs2))
print(np.mean(xs3), np.std(xs3), np.var(xs3), count_positive_negative(xs3) / len(xs3))
print(np.mean(xs4), np.std(xs4), np.var(xs4), count_positive_negative(xs4) / len(xs4))
print(np.mean(xs5), np.std(xs5), np.var(xs5), count_positive_negative(xs5) / len(xs5))

plt.figure()
plt.plot(np.arange(len(xs1)), xs1, label='1')
plt.plot(np.arange(len(xs2)), xs2, label='2')
plt.plot(np.arange(len(xs3)), xs3, label='3')
plt.plot(np.arange(len(xs4)), xs4, label='4')
plt.plot(np.arange(len(xs5)), xs5, label='5')
plt.legend()
plt.show()

plt.show()

exit(1)

disp_ = []

for i in range(len(xs) - 1):
    disp_.append(abs(xs[i+1] - xs[i]))
print(disp_)
print(2 * np.log(disp_)/ np.log(100))
plt.figure()
plt.plot(np.arange(len(disp_)), disp_, label='0')
plt.plot(np.arange(len(xs1)), xs1, label='1')
plt.plot(np.arange(len(xs2)), xs2, label='2')
plt.plot(np.arange(len(xs3)), xs3, label='3')
plt.plot(np.arange(len(xs4)), xs4, label='4')
plt.legend()
plt.show()
exit(1)


def samples_GHE(serie, tau):
    return np.abs(serie[tau:] - serie[:-tau])


def KSGHE(serie, p0):
    scaling_range = [2**n for n in range(int(np.log2(len(serie)))-2)]
    sample_t0 = samples_GHE(serie, tau=scaling_range[0])
    f = lambda h: np.sum([stats.ks_2samp(sample_t0, samples_GHE(serie, tau=tau) / (tau**h)).statistic for tau in scaling_range[1:]])
    w = optimize.fmin(f, x0=p0, disp=False)
    return w[0]


def plot_trajectory(trajectory_list):
    traj1 = trajectory_list[0]
    pos = traj1.get_positions()
    xs = pos[:, 0]
    ys = pos[:, 1]
    plt.figure()
    plt.plot(xs, ys, linewidth=0.5)
    plt.scatter(xs, ys, c=np.arange(T), cmap=plt.get_cmap("gist_rainbow"), alpha=0.7, s=9.0)
    plt.scatter(xs[changepoints], ys[changepoints], c='black', marker='+', s=120.0)
    plt_xlim = [np.min(xs) - 2, np.max(xs) + 2]
    plt_ylim = [np.min(ys) - 2, np.max(ys) + 2]
    if plt_xlim[-1] - plt_xlim[0] > plt_ylim[-1] - plt_ylim[0]:
        plt_ylim[0] = plt_ylim[0] - ((plt_xlim[-1] - plt_xlim[0]) - (plt_ylim[-1] - plt_ylim[0]))
    else:
        plt_xlim[0] = plt_xlim[0] - ((plt_ylim[-1] - plt_ylim[0]) - (plt_xlim[-1] - plt_xlim[0]))
    plt.xlim(plt_xlim)
    plt.ylim(plt_ylim)


def plot_diff_coefs(trajectory_list, *args, t_range=None):
    changepoints = np.array(args[0])
    alphas = args[1]
    Ds = args[2]
    state_num = args[3]
    if t_range is None:
        t_range = [0, 10000]
    FIGSIZE = (14, 9)
    time_interval = T_INTERVAL
    weights = [1, 2, 10]
    nrow = len(weights) + 1
    ncol = 5

    for traj in trajectory_list:
        if traj.get_index() == 0:
            denoised = [[[] for _ in range(ncol)] for _ in range(nrow)]
            diff_coefs = traj.get_inst_diffusion_coefs(time_interval=time_interval, t_range=t_range)
            angles = traj.get_trajectory_angles(time_interval=time_interval, t_range=t_range)
            MSD = traj.get_msd(time_interval=time_interval, t_range=t_range)
            density = traj.get_density(radius=np.mean(np.sort(diff_coefs)), t_range=t_range)
            x_pos = traj.get_positions()[:,0]

            std_angles = []
            for i in range(len(angles)):
                std_angles.append(np.mean(x_pos[max(0, i-5):min(len(angles), i+6)] - float(x_pos[max(0, i-5)])))
            std_angles = np.array(std_angles)

            c, alpha = optimize.curve_fit(power_fit, np.arange(len(MSD)), MSD)[0]
            print(f'C:{c}, alpha:{alpha}')

            #diff_coefs = np.cumsum(diff_coefs)
            #angles = np.cumsum(angles)
            #MSD = np.cumsum(MSD)

            denoised[0][0] = diff_coefs
            denoised[0][1] = angles
            denoised[0][2] = std_angles
            denoised[0][3] = MSD
            denoised[0][4] = density

            for r, weight in enumerate(weights):
                denoised[r+1][0] = denoise_tv_chambolle(diff_coefs, weight=weight)
                denoised[r+1][1] = denoise_tv_chambolle(angles, weight=weight)
                denoised[r+1][2] = denoise_tv_chambolle(std_angles, weight=weight)
                denoised[r+1][3] = denoise_tv_chambolle(MSD, weight=weight)
                denoised[r+1][4] = denoise_tv_chambolle(density, weight=weight)

            fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE)
            fig.suptitle(f'alphas:   {np.round(alphas,2)}\n'
                         f'Ds:       {np.round(Ds, 2)}\n'
                         f'State_num:{state_num}')
            axs[0, 0].title.set_text('Diff_coef')
            axs[0, 1].title.set_text('Angles')
            axs[0, 2].title.set_text('Std angles')
            axs[0, 3].title.set_text('MSD')
            axs[0, 4].title.set_text('Density')
            print(len(diff_coefs), len(angles), len(std_angles), len(MSD), len(density))

            for row in range(nrow):
                for col in range(ncol):
                    if len(denoised[row][col]) > 0:
                        axs[row, col].plot(np.arange(len(denoised[row][col])), denoised[row][col])
                        axs[row, col].set_xlim(t_range)
                        if col == 0:
                            axs[row, col].set_ylim([0, np.max(diff_coefs) + 2])
                            if len(changepoints) > 0:
                                axs[row, col].vlines(changepoints - (t_range[-1] - len(diff_coefs)), ymin=0, ymax=np.max(diff_coefs) + 2,
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 1:
                            axs[row, col].set_ylim([0, np.max(angles) + 3])
                            if len(changepoints) > 0:
                                axs[row, col].vlines(changepoints - (t_range[-1] - len(angles)), ymin=0, ymax=np.max(angles) + 3,
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 2:
                            axs[row, col].set_ylim([-(np.max(std_angles) + 2), np.max(std_angles) + 2])
                            if len(changepoints) > 0:
                                axs[row, col].vlines(changepoints - (t_range[-1] - len(std_angles)), ymin=0, ymax=np.max(std_angles) + 2,
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 3:
                            axs[row, col].plot(np.arange(len(MSD)), power_fit(np.arange(len(MSD)), c, alpha))
                            if len(changepoints) > 0:
                                axs[row, col].vlines(changepoints - (t_range[-1] - len(MSD)), ymin=0, ymax=np.max(MSD),
                                                     colors='red',
                                                     alpha=0.5)

                        if col == 4:
                            axs[row, col].set_ylim([0, np.max(density) + 2])
                            if len(changepoints) > 0:
                                axs[row, col].vlines(changepoints - (t_range[-1] - len(density)), ymin=0, ymax=np.max(density),
                                                     colors='red',
                                                     alpha=0.5)


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


def uncumulate(xs:np.ndarray):
    assert xs.ndim == 1
    uncum_list = [0.]
    for i in range(1, len(xs)):
        uncum_list.append(xs[i] - xs[i-1])
    return np.array(uncum_list)


if __name__ == '__main__':
    N = 1
    T = 200
    t_range = [0, 200]
    D = 0.1
    L = 1.5 * 128
    T_INTERVAL = 1
    RADIUS = 1

    #stochastic.random.seed(3)
    #np.random.seed(7)
    """
    trajs_model1, labels_model1 = models_phenom().single_state(N=N,
                                                              L=False,
                                                              T=200,
                                                              Ds=[0.1, 0],  # Mean and variance
                                                              alphas=hurst_exp * 2
                                                              )
    """
    print('---------')
    trajs_model2, labels_model2 = models_phenom().multi_state(N=N,
                                                              L=L,
                                                              T=T,
                                                              alphas=[1.2, 0.8],  # Fixed alpha for each state
                                                              Ds=[[0.5, 0.0], [0.5, 0.0]],
                                                              # Mean and variance of each state
                                                              M=[[0.98, 0.02], [0.02, 0.98]]
                                                              )

    trajs_model = trajs_model2
    trajs_label = labels_model2
    print('Tajectory shapes, label shape: ', trajs_model.shape, trajs_label.shape)
    changepoints, alphas, Ds, state_num = label_continuous_to_list(trajs_label[:, 0, :])
    changepoints = np.delete(changepoints, np.where(changepoints == T))
    print(f'change points: {changepoints}')
    print(f'alphas: {alphas}')
    print(f'Ds: {Ds}')
    print(f'State_nums: {state_num}')

    #xs = disp_fbm(alpha=hurst_exp * 2, D=0.1, T=1024)[t_trange[0]:t_trange[1]]
    #ys = disp_fbm(alpha=hurst_exp * 2, D=0.1, T=1024)[t_trange[0]:t_trange[1]]
    #pos = np.array([xs, ys]).T

    xs = trajs_model[:, 0, 0]
    ys = trajs_model[:, 0, 1]
    pos = np.array([xs, ys]).T
    traj1 = TrajectoryObj(index=0)
    for t, (x, y) in enumerate(pos):
        traj1.add_trajectory_position(t, x, y, 0.0)
    MSD = traj1.get_msd(time_interval=1, t_range=t_range)
    diff_coefs = traj1.get_inst_diffusion_coefs(time_interval=1, t_range=t_range)

    print(f'DIFF_COEF_MEAN: {np.mean(diff_coefs)}')

    print('pip hurst compute_Hc')
    print(compute_Hc(xs, kind='random_walk')[0] * 2)
    print(compute_Hc(ys, kind='random_walk')[0] * 2)
    print(compute_Hc(MSD, kind='random_walk')[0] * 2)
    print("-----------------------------")

    print("\nKSGHD")
    print(f'X: {KSGHE(xs, 0.5) * 2}')
    print(f'Y: {KSGHE(ys, 0.5) * 2}')
    print(f'MSD: {KSGHE(MSD, 0.5) * 2}')
    print("-----------------------------")

    print("\nMY RSRange")
    print(f'X: {rescaled_range(xs) * 2}')
    print(f'Y: {rescaled_range(ys) * 2}')
    print(f'MSD: {rescaled_range(MSD) * 2}')
    print("-----------------------------")

    print("\nKSGHD with Mine")
    print(f'X: {KSGHE(xs, rescaled_range(xs)) * 2}')
    print(f'Y: {KSGHE(ys, rescaled_range(ys)) * 2}')
    print(f'MSD: {KSGHE(MSD, rescaled_range(MSD)) * 2}')
    print("-----------------------------")


    plot_trajectory([traj1])
    plot_diff_coefs([traj1],
                    changepoints, alphas, Ds, state_num,
                    t_range=t_range)
    plt.show()
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
