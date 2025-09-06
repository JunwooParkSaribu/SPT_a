import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from FreeTrace.module.preprocessing import simple_preprocessing
from FreeTrace.module.data_load import read_csv, read_multiple_csv
from analysis_modeling.andi_datasets.models_phenom import models_phenom
from analysis_modeling.andi_datasets.datasets_phenom import datasets_phenom
from analysis_modeling.andi_datasets.utils_trajectories import plot_trajs
from analysis_modeling.andi_datasets.utils_challenge import label_continuous_to_list
from stochastic import random as strandom
from stochastic.processes.continuous import FractionalBrownianMotion
from scipy.stats import ks_1samp
from scipy.signal import find_peaks
from scipy import stats
from scipy import signal

traj_df = read_csv(file='outputs/sample0_traces.csv')
#trajectory_data = read_multiple_csv('outputs')

def cdf_cauchy(x, h, t=1, s=1):
    constant = 0.5
    val = np.arctan((2*(s**h)*x + (t**h)*(2-(2**(2*h)))) / ((t**h) * np.sqrt(2**(2*h+2) - 2**(4*h)))) / np.pi
    return val + constant

#res = ks_1samp(np.random.normal(0, 1, 100), cdf_cauchy, args=[0.5, 1, 1])
#plt.figure()
#plt.bar(np.linspace(-10, 10, 100), cdf_cauchy(np.linspace(-10, 10, 100), 0.5))
#plt.show()
#print(res)
#exit()
"""
Option settings for data analysis.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5  # minimum length of trajectory to consider for the analysis.
number_of_bins = 50
figure_resolution_in_dpi = 200
figure_font_size = 20


"""
Preprocessing generates 4 data.
@params: data folder path, pixel microns, frame rate, cutoff
@output: Analysis_data1(pd.DataFrame), Analysis_data2(pd.DataFrame), MSD(pd.DataFrame), TAMSD(pd.DataFrame)

Simple_preprocessing includes below steps.
1. Exclude the trajectory where the length is shorter than CUTOFF
2. Convert from pixel unit to micrometre unit with PIXELMICRONS and FRAMERATE
3. Generate 4 DataFrames.
"""
"""
analysis_data1, analysis_data2, analysis_data3, msd, tamsd = simple_preprocessing(data=trajectory_data,
                                                                                  pixelmicrons=PIXELMICRONS,
                                                                                  framerate=FRAMERATE,
                                                                                  cutoff=CUTOFF)

print(f'\nanalysis_data1:\n', analysis_data1)
print(f'\nanalysis_data2:\n', analysis_data2)
print(f'\nanalysis_data3:\n', analysis_data3)
print(f'\nMSD:\n', msd)
print(f'\nEnsemble-averaged TAMSD:\n', tamsd)
"""
#print(trajectory_data)
"""
def fbm_pdf(traj_df):
    traj_indices = traj_df['traj_idx'].unique()
    alphas = np.linspace(0, 2, 1002)[1:-1]

    for idx in traj_indices[::-1]:
        trajs = traj_df[traj_df['traj_idx'] == idx]
        xs = trajs['x'].to_numpy()
        ys = trajs['y'].to_numpy()
        frames = trajs['frame'].to_numpy()
        inc_xs = xs[1:] - xs[:-1]
        inc_ys = ys[1:] - ys[:-1]
        frame_inc = frames[1:] - frames[:-1]
        
        log_pdfs = []
        for alpha in alphas:
            log_pdf = predict_cauchy(inc_xs, inc_ys, frame_inc, alpha)
            log_pdfs.append(log_pdf)
        
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].bar(alphas, log_pdfs, width=0.01)
        axs[1].plot(xs, ys)
        axs[1].set_xlim([np.mean(xs) - 20, np.mean(xs) + 20])
        axs[1].set_ylim([np.mean(ys) - 20, np.mean(ys) + 20])
        plt.show()
    pass
"""

def predict_cauchy(x_inc, y_inc, frame_inc, alpha):
    log_pdf = []
    abnormal = False

    coord_ratios = []
    
    if abs(alpha - 1.0) < 1e-4:
        alpha += 1e-5


    for idx in range(len(x_inc) - 1):
        delta_t = frame_inc[idx+1]
        delta_s = frame_inc[idx]

        rho = math.pow(2, alpha - 1) -1
        std_ratio = math.sqrt((math.pow(2*delta_t, alpha) - 2*math.pow(delta_t, alpha)) / (math.pow(2*delta_s, alpha) - 2*math.pow(delta_s, alpha)))
        scale = math.sqrt(1 - math.pow(rho, 2)) * std_ratio

        coord_ratios = [x_inc[idx + 1] / x_inc[idx], y_inc[idx + 1] / y_inc[idx]]
        for coord_ratio in coord_ratios:
            density = 1.0/(math.pi * scale) * 1.0 / (( math.pow((coord_ratio - rho*std_ratio), 2) / (math.pow(scale, 2) * std_ratio) ) + (std_ratio))
            log_pdf.append(math.log(1 + density))
        

    return log_pdf


def pdf_cauchy(u, h, t=1, s=1):
    return 1/(np.pi * np.sqrt(1-(2**(2*h-1)-1)**2)) * 1 / ( (u - (2**(2*h-1) - 1)*(t/s)**h)**2 / ((1 - (2**(2*h-1)-1)**2)*np.sqrt((t/s)**(2*h))) + (np.sqrt((t/s)**(2*h))))


def cdf_cauchy(u, h, t=1, s=1):
    val = np.arctan( (2*(s**h)*u + (t**h)*(2 - 2**(2*h))) / ((t**h) * np.sqrt(2**(2*h+2) - 2**(4*h))) ) / np.pi + 0.5
    return val


def inv_cdf_cauchy(proba, h, t=1, s=1):
    denom = ((t**h) * np.sqrt(2**(2*h+2) - 2**(4*h)))
    a = (2*(s**h)) / denom
    return (np.tan(np.pi * proba - np.pi * 0.5) / a) - ((t**h)*(2 - 2**(2*h)) / 2*(s**h))


def cost_func(u, h):
    cost = 0
    conf_interval = 0.995

    lower_side = (1 - conf_interval) / 2.
    upper_side = 1 - lower_side
    lower_bound = inv_cdf_cauchy(lower_side, h)
    upper_bound = inv_cdf_cauchy(upper_side, h)

    if u > upper_bound:
        cost += (u - upper_bound) * pdf_cauchy(u, h)
    elif u < lower_bound:
        cost += (u - lower_bound) * pdf_cauchy(u, h)
    else:
        cost += pdf_cauchy(u, h)

    return cost



trajs_model, labels_model = models_phenom().multi_state(N=100,
                                                        L=None,
                                                        T=64,
                                                        alphas=[0.3, 1.0],
                                                        Ds=[[0.2, 0.0], [1.0, 0.0]],
                                                        M=[[0.995, 0.005], [0.005, 0.995]]
                                                       )

"""
trajs_model, labels_model = models_phenom().single_state(N=100,
                                                         L=None,
                                                         T=32,
                                                         alphas=0.3,
                                                         Ds=[0.01, 0],
                                                         dim=2
                                                         )
"""


alphas = np.linspace(0, 2, 1002)[1:-1]

"""
traj_indices = traj_df['traj_idx'].unique()
for idx in traj_indices[::-1]:
    trajs = traj_df[traj_df['traj_idx'] == idx]
    xs = trajs['x'].to_numpy()
    ys = trajs['y'].to_numpy()
    frames = trajs['frame'].to_numpy()



"""

aa = []
bb = []
np.set_printoptions(suppress=True)
for traj_idx in range(trajs_model.shape[1]):
    xs = trajs_model[:, traj_idx, 0]
    ys = trajs_model[:, traj_idx, 1]
    frames = np.arange(len(xs)) + 1
    print(xs, ys, labels_model[:, traj_idx, 0])
    change_points = abs(labels_model[:, traj_idx, 0][1:] - labels_model[:, traj_idx, 0][:-1]) > 0.1
    change_points_indices = [idx for idx, bool_check in enumerate(change_points) if bool_check == True]
    print(change_points_indices)
    #change_indices = [for idx, val in enumerate(labels_model[:, traj_idx, 0])]
######################################################

    inc_xs = xs[1:] - xs[:-1]
    inc_ys = ys[1:] - ys[:-1]
    frame_inc = frames[1:] - frames[:-1]



    #inc_xs = np.insert(inc_xs, int(len(inc_xs)/2), 5)
    #inc_ys = np.insert(inc_ys, int(len(inc_ys)/2), 1)
    #frame_inc = np.ones(len(inc_xs))
    


    peaks_raw, _ = find_peaks(inc_xs, distance=155)
    peakind = signal.find_peaks_cwt(inc_xs, np.arange(1,2))
    std_signal = np.array([np.std(inc_xs[max(0, idx-2):idx+3]) for idx in range(len(inc_xs))])
    peaks_std, _ = find_peaks(std_signal, distance=155)

    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(inc_xs)
    axs[0].plot(peaks_raw, inc_xs[peaks_raw], "x")
    axs[0].vlines(change_points_indices, ymin=-2, ymax=2, color='red')
    axs[1].plot([np.mean(inc_xs[:idx+1]) for idx in range(len(inc_xs))])
    axs[1].plot([np.std(inc_xs[max(0, idx-2):idx+3]) for idx in range(len(inc_xs))])
    axs[1].plot(peaks_std, std_signal[peaks_std], "x")
    axs[1].set_ylim([-6, 6])
    

    plt.show()

""" 
    ratio_concat = np.concatenate(((inc_xs[1:] / inc_xs[:-1]), (inc_ys[1:] / inc_ys[:-1])))
    print(inc_xs, inc_ys)
    print()
    mycheck = ratio_concat.reshape(2, -1)
    mycheck = mycheck[:,1:] / mycheck[:,:-1]
    first_ratio = ratio_concat.reshape(2, -1)
    mycheck = mycheck.reshape(-1)
    first_ratio = first_ratio.reshape(-1)

    aa = np.concatenate((aa, first_ratio))
    bb = np.concatenate((bb, mycheck))


    
    #log_pdfs = []
    ks_dist = []
    pvals = []
    all_costs = []
    check_vals = np.empty((len(inc_xs)-1, len(alphas), 2))
    for inc_index in range(len(inc_xs) - 1):
        #log_pdfs = []
        #print(one_incx, one_incy)
        costs = []
        for alpha_idx, alpha in enumerate(alphas):
            cost = 0
            log_pdf = predict_cauchy([inc_xs[inc_index], inc_xs[inc_index + 1]], [inc_ys[inc_index], inc_ys[inc_index + 1]], frame_inc, alpha)
            cost += cost_func(inc_xs[inc_index + 1] / inc_xs[inc_index], alpha/2.)
            cost += cost_func(inc_ys[inc_index + 1] / inc_ys[inc_index], alpha/2.)
            #log_pdfs.append(log_pdf)
            x_pdf = log_pdf[0]
            y_pdf = log_pdf[1]
            res = ks_1samp(ratio_concat, cdf_cauchy, args=(alpha/2, 1.0, 1.0) )
            ks_dist.append(res[0])
            pvals.append(res[1])
            #print(alpha, " : " , [inc_xs[inc_index], inc_xs[inc_index + 1]], [inc_ys[inc_index], inc_ys[inc_index + 1]], np.mean(log_pdfs))
            check_vals[inc_index, alpha_idx, 0] = x_pdf
            check_vals[inc_index, alpha_idx, 1] = y_pdf
            costs.append(cost)
        print(costs)
        all_costs.append(costs)
        exit()
    #print(check_vals)
    check_vals = np.array(check_vals)
    print(check_vals.shape)
    fig, axs = plt.subplots(nrows=len(inc_xs)-1, ncols=1)
    for ax_idx, ax in enumerate(axs):
        ax.plot(alphas, check_vals[ax_idx, :, 0], c='red')
        ax.plot(alphas, check_vals[ax_idx, :, 1], c='blue')
        ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.show()
    exit()
    
    fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(10, 6))
    axs[0].bar(alphas, log_pdfs, width=0.01)
    axs[1].plot(xs, ys)
    axs[1].set_xlim([np.mean(xs) - 10, np.mean(xs) + 10])
    axs[1].set_ylim([np.mean(ys) - 10, np.mean(ys) + 10])
    axs[2].hist(ratio_concat, bins = np.linspace(-25, 25, 25))
    #empirical_cdf = np.cumsum(np.arange(len(ratio_concat)))
    #empirical_cdf = empirical_cdf / np.max(empirical_cdf)
    #print(empirical_cdf)
    #axs[3].hist(empirical_cdf, bins = np.linspace(-5, 5, 100))
    #axs[3].bar(np.sort(ratio_concat), empirical_cdf)
    #axs[3].set_xlim([-5, 5])
    axs[3].bar(alphas, ks_dist, width=0.01)
    axs[3].set_ylim([0.0, 1.0])
    axs[4].bar(alphas, pvals, width=0.01)
    axs[4].set_ylim([0.0, 1.0])



    ks_dist_fake = []
    pvals_fake = []
    ratio_concat_fake = np.concatenate((ratio_concat, [40]*2))
    for alpha in alphas:
        res = ks_1samp(ratio_concat_fake, cdf_cauchy, args=(alpha / 2.0, 1, 1))
        ks_dist_fake.append(res[0])
        pvals_fake.append(res[1])
    axs[5].bar(alphas, ks_dist_fake, width=0.01)
    axs[5].set_ylim([0.0, 1.0])
    axs[6].bar(alphas, pvals_fake, width=0.01)
    axs[6].set_ylim([0.0, 1.0])


    plt.show()

   

#fbm_pdf(trajectory_data)


plt.figure()
#plt.scatter([1]*len(aa), aa)
#plt.scatter([2]*len(bb), bb)
plt.boxplot(aa)
plt.show()
"""