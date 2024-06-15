import os
import numpy as np
import json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import gc


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

reg_model_nums = [5, 8, 12, 16, 32, 48, 64, 128, 144]
reg_models = {n: tf.keras.models.load_model(f'./models/alpha_reg_models/reg_model_{n}.keras') for n in reg_model_nums}
reg_k_model = tf.keras.models.load_model(f'./models/alpha_reg_models/reg_k_model.keras')


MAX_DENSITY_NB = 25
EXT_WIDTH = 100
WIN_WIDTHS = np.arange(20, 50, 2)
param_grid = {
    "n_components": range(2, 5),
}
CLUSTER_IMAGE = False


N_FOVS = np.arange(0, 30).astype(int)
N_EXPS = np.arange(0, 13).astype(int)
TRACKS = [2, 1]
submit_number = 8


def gmm_bic_score(estimator, X):
    return -estimator.bic(X)


def uncumulate(xs:np.ndarray):
    assert xs.ndim == 1
    uncum_list = [0.]
    for i in range(1, len(xs)):
        uncum_list.append(xs[i] - xs[i-1])
    return np.array(uncum_list)


def radius_list(xs:np.ndarray, ys:np.ndarray):
    assert xs.ndim == 1 and ys.ndim == 1
    rad_list = [0.]
    disp_list = []
    for i in range(1, len(xs)):
        rad_list.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
        disp_list.append(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
    return np.array(rad_list) / np.mean(disp_list) / len(xs)


def make_signal(x_pos, y_pos, win_widths):
    all_vals = []
    for win_width in win_widths:
        if win_width >= len(x_pos):
            continue
        vals = []
        for checkpoint in range(win_width // 2, len(x_pos) - win_width // 2):
            xs = x_pos[checkpoint - int(win_width / 2): checkpoint + int(win_width / 2)]
            ys = y_pos[checkpoint - int(win_width / 2): checkpoint + int(win_width / 2)]

            xs1 = xs[1: int(len(xs) / 2) + 1] - float(xs[1: int(len(xs) / 2) + 1][0])
            xs2 = xs[int(len(xs) / 2):] - float(xs[int(len(xs) / 2):][0])
            ys1 = ys[1: int(len(ys) / 2) + 1] - float(ys[1: int(len(ys) / 2) + 1][0])
            ys2 = ys[int(len(ys) / 2):] - float(ys[int(len(ys) / 2):][0])

            cum_xs1 = abs(np.cumsum(abs(xs1)))
            cum_xs2 = abs(np.cumsum(abs(xs2)))
            cum_ys1 = abs(np.cumsum(abs(ys1)))
            cum_ys2 = abs(np.cumsum(abs(ys2)))

            xs_max_val = max(np.max(abs(cum_xs1)), np.max(abs(cum_xs2)))
            cum_xs1 = cum_xs1 / xs_max_val
            cum_xs2 = cum_xs2 / xs_max_val

            ys_max_val = max(np.max(abs(cum_ys1)), np.max(abs(cum_ys2)))
            cum_ys1 = cum_ys1 / ys_max_val
            cum_ys2 = cum_ys2 / ys_max_val

            vals.append((abs(cum_xs1[-1] - cum_xs2[-1] + cum_ys1[-1] - cum_ys2[-1]))
                        + (max(np.std(xs1), np.std(xs2)) - min(np.std(xs1), np.std(xs2)))
                        + (max(np.std(ys1), np.std(ys2)) - min(np.std(ys1), np.std(ys2))))

        vals = np.concatenate((np.ones(int(win_width / 2)) * 0, vals))
        vals = np.concatenate((vals, np.ones(int(win_width / 2)) * 0))
        vals = np.array(vals)
        all_vals.append(vals)

    all_vals = np.array(all_vals) + 1e-5
    return all_vals


def slice_data(signal_seq, jump_d, ext_width, shift_width):
    slice_d = []
    indice = []
    for i in range(ext_width, signal_seq.shape[1] - ext_width, jump_d):
        crop = signal_seq[:, i - shift_width//2: i + shift_width//2]
        slice_d.append(crop)
        indice.append(i)
    return np.array(slice_d), np.array(indice) - ext_width


def climb_mountain(signal, cp, seuil=5):
    while True:
        vals = [signal[x] if 0<=x<signal.shape[0] else -1 for x in range(cp-seuil, cp+1+seuil)]
        if len(vals) == 0:
            return -1
        new_cp = cp + np.argmax(vals) - seuil
        if new_cp == cp:
            return new_cp
        else:
            cp = new_cp


def position_extension(x, y, ext_width):
    datas = []
    for data in [x, y]:
        delta_prev_data = -uncumulate(data[:min(data.shape[0], ext_width)])[1:]
        delta_prev_data[0] += float(data[0])
        prev_data = np.cumsum(delta_prev_data)[::-1]

        delta_next_data = -uncumulate(data[data.shape[0] - min(data.shape[0], ext_width):][::-1])[1:]
        delta_next_data[0] += float(data[-1])
        next_data = np.cumsum(delta_next_data)

        ext_data = np.concatenate((prev_data, data))
        ext_data = np.concatenate((ext_data, next_data))
        datas.append(ext_data)
    return np.array(datas), delta_prev_data.shape[0], delta_next_data.shape[0]


def sigmoid(x, beta=3):
    x = np.minimum(np.ones_like(x)*0.999, x)
    x = np.maximum(np.ones_like(x)*0.001, x)
    return 1 / (1 + (x / (1-x))**(-beta))


def density_estimation(x, y, max_nb):
    densities = []
    dist_amp = 2.0
    local_mean_window_size = 5

    for i in range(x.shape[0]):
        density1 = 0
        density2 = 0

        slice_x = x[max(0, i - max_nb // 2):i].copy()
        slice_y = y[max(0, i - max_nb // 2):i].copy()

        if len(slice_x) > 0:
            mean_dist = np.sqrt(uncumulate(slice_x) ** 2 + uncumulate(slice_y) ** 2).mean()
            mean_dist *= dist_amp

            slice_x -= slice_x[len(slice_x) // 2]
            slice_y -= slice_y[len(slice_y) // 2]
            for s_x, s_y in zip(slice_x, slice_y):
                if np.sqrt(s_x ** 2 + s_y ** 2) < mean_dist:
                    density1 += 1

        slice_x = x[i:min(x.shape[0], i + max_nb // 2)].copy()
        slice_y = y[i:min(x.shape[0], i + max_nb // 2)].copy()

        if len(slice_x) > 0:
            mean_dist = np.sqrt(uncumulate(slice_x) ** 2 + uncumulate(slice_y) ** 2).mean()
            mean_dist *= dist_amp

            slice_x -= slice_x[len(slice_x) // 2]
            slice_y -= slice_y[len(slice_y) // 2]
            for s_x, s_y in zip(slice_x, slice_y):
                if np.sqrt(s_x ** 2 + s_y ** 2) < mean_dist:
                    density2 += 1
        densities.append(max(density1, density2))

    # local_mean
    new_densities = []
    for i in range(len(densities)):
        new_densities.append(np.mean(densities[max(0, i - local_mean_window_size // 2):
                                               min(len(densities), i + local_mean_window_size // 2 + 1)]))
    densities = new_densities
    return np.array(densities)


def signal_from_extended_data(x, y, win_widths, ext_width, jump_d, shift_width):
    assert ext_width > shift_width
    datas, shape_ext1, shape_ext2 = position_extension(x, y, ext_width)
    signal = make_signal(datas[0], datas[1], win_widths)

    density = density_estimation(datas[0], datas[1],
                                 max_nb=MAX_DENSITY_NB * 2)

    denoised_den = denoise_tv_chambolle(density, weight=3, eps=0.0002, max_num_iter=100, channel_axis=None)
    denoised_den = sigmoid(denoised_den / MAX_DENSITY_NB)
    # denoised_den = 1

    signal = signal[:, ] * denoised_den
    sliced_signals, slice_indice = slice_data(signal, jump_d, shape_ext1, shift_width)
    return sliced_signals


def local_roughness(signal, window_size):
    uc_signal = uncumulate(signal)
    uc_signal /= abs(uc_signal)
    counts = []
    for i in range(window_size//2, len(uc_signal) - window_size//2):
        count = 0
        cur_state = 1
        for j in range(i-window_size//2, i+window_size//2):
            new_state = uc_signal[j]
            if new_state != cur_state:
                count += 1
            cur_state = new_state
        counts.append(count)
    counts = np.concatenate(([counts[0]] * (window_size//2), counts))
    counts = np.concatenate((counts, [counts[-1]] * (window_size//2)))
    return counts


def slice_normalize(slices):
    val = np.mean(np.sum(slices, axis=(2)).T, axis=0)
    val = val - np.min(val)
    val = val / np.max(val)
    return val


def displacements(x, y):
    disps = []
    for i in range(len(x) - 1):
        x_seg = x[i + 1] - x[i]
        y_seg = y[i + 1] - y[i]
        disp = np.sqrt(x_seg ** 2 + y_seg ** 2)
        disps.append([np.log10(disp)])
    return np.array(disps).mean()


def model_selection(length):
    reg_model_num = -1
    if length < 8:
        reg_model_num = 5
    elif length < 12:
        reg_model_num = 8
    elif length < 16:
        reg_model_num = 12
    elif length < 32:
        reg_model_num = 16
    elif length < 48:
        reg_model_num = 32
    elif length < 64:
        reg_model_num = 48
    elif length < 128:
        reg_model_num = 64
    elif length < 144:
        reg_model_num = 128
    else:
        reg_model_num = 144
    return reg_model_num


def cvt_2_signal(x, y):
    rad_list = radius_list(x, y)
    x = x / (np.std(x))
    x = np.cumsum(abs(uncumulate(x))) / len(x)
    y = y / (np.std(y))
    y = np.cumsum(abs(uncumulate(y))) / len(y)
    return np.vstack((x, rad_list)).T, np.vstack((y, rad_list)).T


def partition_trajectory(x, y, cps):
    if len(cps) == 0:
        return [x], [y]
    new_x = []
    new_y = []
    for i in range(1, len(cps)):
        new_x.append(x[cps[i-1]:cps[i]])
        new_y.append(y[cps[i-1]:cps[i]])
    return new_x, new_y


def sort_by_signal(signal, cps):
    sort_indice = np.argsort(signal[cps])
    indice_tuple = [(i, i+1) for i in sort_indice]
    return indice_tuple, sort_indice


def recoupe_trajectory(x, y, model_num, jump=1):
    couped_x = []
    couped_y = []
    for i in range(0, len(x), jump):
        tmp1 = x[i: i+model_num]
        tmp2 = y[i: i+model_num]
        if len(tmp1) == model_num:
            couped_x.append(tmp1)
            couped_y.append(tmp2)
    return np.array(couped_x), np.array(couped_y)


def exhaustive_cps_search(x, y, win_widths, ext_width, search_seuil=0.25, cluster=None):
    if len(x) < np.min(reg_model_nums):
        return np.array([0, len(x)]), np.array([1.0]), np.array([0.1]), np.array([len(x)])

    if cluster is not None and len(cluster.means_) == 1:
        start_cps = []
        slice_norm_signal = np.zeros_like(x.shape)
    else:
        if len(x) + 2 * (len(x) - 1) >= win_widths[0]:
            sliced_signals = signal_from_extended_data(x, y, win_widths, ext_width, 1, 10)
            slice_norm_signal = slice_normalize(sliced_signals)

            det_cps = []
            for det_cp in np.where(slice_norm_signal > search_seuil)[0]:
                det_cps.append(climb_mountain(slice_norm_signal, det_cp, seuil=5))
            det_cps = np.unique(det_cps)
            start_cps = list(det_cps.copy())
        else:
            start_cps = []
            slice_norm_signal = np.zeros_like(x.shape)

    start_cps.append(0)
    start_cps.append(len(x))
    start_cps = np.sort(start_cps)
    cps_copy = [0]
    for i in range(1, len(start_cps) - 1):
        if start_cps[i] - start_cps[i - 1] > np.min(reg_model_nums) and start_cps[i + 1] - start_cps[i] > np.min(reg_model_nums):
            cps_copy.append(start_cps[i])
    start_cps = cps_copy
    start_cps.append(len(x))

    while True:
        filtered_cps = []
        alpha_preds = []
        k_preds = []

        part_xs, part_ys = partition_trajectory(x, y, start_cps)
        for p_x, p_y in zip(part_xs, part_ys):
            input_signals = []

            model_num = model_selection(len(p_x))
            model = reg_models[model_num]

            re_couped_x, re_couped_y = recoupe_trajectory(p_x, p_y, model_num)
            for r_x, r_y in zip(re_couped_x, re_couped_y):
                input_signal1, input_signal2 = cvt_2_signal(r_x, r_y)
                input_signals.append(input_signal1)
                input_signals.append(input_signal2)

            input_signals = np.array(input_signals).reshape(-1, model_num, 1, 2)
            pred_alpha = model.predict(input_signals, verbose=0).flatten()

            if len(pred_alpha) > 4:
                pred_alpha = np.sort(pred_alpha)[int(0.25 * len(pred_alpha)): int(0.75 * len(pred_alpha))].mean()
            else:
                pred_alpha = np.mean(pred_alpha)

            k_preds.append(reg_k_model.predict(np.array([displacements(p_x, p_y)]), verbose=0)[0][0])
            alpha_preds.append(pred_alpha)

        delete_cps = -1
        if cluster is not None:
            sorted_indice_tuple, sorted_indice = sort_by_signal(slice_norm_signal, start_cps[1:-1])
            for (l, r), i in zip(sorted_indice_tuple, sorted_indice):
                i += 1
                diff_alpha = abs(alpha_preds[l] - alpha_preds[r])

                cluster_pred_label = cluster.predict([[alpha_preds[l], k_preds[l]], [alpha_preds[r], k_preds[r]]])
                cluster_pred_proba = cluster.predict_proba([[alpha_preds[l], k_preds[l]], [alpha_preds[r], k_preds[r]]])

                prev_cluster_pred_label = cluster_pred_label[0]
                after_cluster_pred_label = cluster_pred_label[1]
                prev_cluster_pred_probas = cluster_pred_proba[0]
                after_cluster_pred_probas = cluster_pred_proba[1]

                left_length = start_cps[i] - start_cps[i - 1]
                right_length = start_cps[i + 1] - start_cps[i]

                del_conditions = [1e-6, 1e-4, 0.10, 0.20]
                len_conds = [-1, -1]

                for cond_k, length in enumerate([left_length, right_length]):
                    if length < 16:
                        len_conds[cond_k] = 0
                    elif length < 32:
                        len_conds[cond_k] = 1
                    elif length < 64:
                        len_conds[cond_k] = 2
                    else:
                        len_conds[cond_k] = 3

                if after_cluster_pred_probas[prev_cluster_pred_label] > del_conditions[len_conds[1]] or prev_cluster_pred_probas[after_cluster_pred_label] > del_conditions[len_conds[0]]:
                    delete_cps = start_cps[i]

        if delete_cps == -1:
            filtered_cps = start_cps
            break
        else:
            start_cps.remove(delete_cps)

    seg_lengths = uncumulate(np.array(filtered_cps))[1:]
    alpha_preds = np.array(alpha_preds)
    filtered_cps = np.array(filtered_cps)
    k_preds = np.array(k_preds)
    states = []

    for k, alpha in zip(k_preds, alpha_preds):
        if k < -2 and alpha < 1.0:
            states.append(0)
        elif alpha < 1.0:
            states.append(1)
        elif alpha > 1.9:
            states.append(3)
        else:
            states.append(2)
    return filtered_cps, alpha_preds, k_preds, states, seg_lengths


def load_priors(path, track, exp):
    try:
        loaded = np.load(f'{path}/priors_{track}_{exp}.npz')
        all_alphas = loaded['alphas']
        all_seg_lengths = loaded['seg_lengths']
        all_ks = loaded['all_ks'].flatten()
    except Exception as e:
        print(f'No priors for exp:{exp}')
        return -1, -1, -1
    return all_alphas, all_ks, all_seg_lengths


def make_priors(datapath, savepath):
    for track in TRACKS:
        for exp in N_EXPS:
            if not os.path.exists(f'{savepath}/priors_{track}_{exp}.npz'):
                print(f'Prior building: Track: {track}, Exp: {exp}')
                all_seg_lengths = []
                all_alphas = []
                all_ks = []

                for fov in N_FOVS:
                    if track == 2:
                        df = pd.read_csv(datapath + f'track_{track}/exp_{exp}/trajs_fov_{fov}.csv')
                    else:
                        df = pd.read_csv(datapath + f'track_{track}/exp_{exp}/videos_fov_{fov}_track.csv')
                    traj_idx = np.sort(df.traj_idx.unique())
                    for idx in traj_idx:
                        x = np.array(df[df.traj_idx == idx])[:, 2]
                        y = np.array(df[df.traj_idx == idx])[:, 3]

                        cps, alphas, ks, state, seg_lengths = exhaustive_cps_search(x, y,
                                                                                    WIN_WIDTHS,
                                                                                    EXT_WIDTH,
                                                                                    search_seuil=0.25,
                                                                                    cluster=None)
                        all_alphas.extend(alphas)
                        all_seg_lengths.extend(seg_lengths)
                        all_ks.extend(ks)

                all_alphas = np.array(all_alphas)
                all_seg_lengths = np.array(all_seg_lengths)
                all_ks = np.array(all_ks)
                np.savez(f'{savepath}/priors_{track}_{exp}.npz', alphas=all_alphas, seg_lengths=all_seg_lengths, all_ks=all_ks)
                del all_alphas
                del all_seg_lengths
                del all_ks
                del cps
                del alphas
                del ks
                del state
                del seg_lengths

            else:
                print(f'prior_ok: Track: {track}, Exp: {exp}')


def main():    # Define the number of experiments and number of FOVS
    print(f'Submit number: {submit_number}')
    print('--- Main processing ---')
    alpha_range = np.linspace(-2.2, 4.2, 300)
    k_range = np.linspace(-6.2, 6.2, 600)

    public_data_path = f'public_data_validation_v1/'  # make sure the folder has this name or change it
    path_results = f'result_validation_{submit_number}/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    for track in TRACKS:

        # Create the folder of the track if it does not exists
        path_track = path_results + f'track_{track}/'
        if not os.path.exists(path_track):
            os.makedirs(path_track)

        for exp in N_EXPS:
            # Create the folder of the experiment if it does not exits
            path_exp = path_track + f'exp_{exp}/'
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)
            """
            file_name = path_exp + 'ensemble_labels.txt'

            with open(file_name, 'a') as f:
                # Save the model (random) and the number of states (2 in this case)
                model_name = np.random.choice(datasets_phenom().avail_models_name, size = 1)[0]
                f.write(f'model: {model_name}; num_state: {2} \n')

                # Create some dummy data for 2 states. This means 2 columns
                # and 5 rows
                data = np.random.rand(5, 2)

                data[-1,:] /= data[-1,:].sum()

                # Save the data in the corresponding ensemble file
                np.savetxt(f, data, delimiter = ';')
            """

    make_priors(public_data_path, path_results)
    with open(f'{path_results}nb_states_prior.json') as f:
        states_priors = []
        states_json = json.load(f)
        for track_n in TRACKS:
            states_priors.append(np.array(states_json[f'track{track_n}'])[N_EXPS])

    for track, st_arr in zip(TRACKS, states_priors):
        path_track = path_results + f'track_{track}/'

        for exp, nb_state in zip(N_EXPS, st_arr):
            path_exp = path_track + f'exp_{exp}/'
            print(f'Predicting:  Track: {track}, Exp: {exp}')
            all_alphas, all_ks, all_seg_lengths = load_priors(path_results, track, exp)
            all_alphas = all_alphas[np.argwhere((all_seg_lengths > 32) & (all_seg_lengths < 512)).flatten()]
            all_ks = all_ks[np.argwhere((all_seg_lengths > 32) & (all_seg_lengths < 512)).flatten()]

            if nb_state == -1:
                grid_search = GridSearchCV(
                    GaussianMixture(max_iter=1000, n_init=50, covariance_type='tied'), param_grid=param_grid,
                    scoring=gmm_bic_score
                )
                grid_search.fit(np.vstack((all_alphas, all_ks)).T)
                cluster_df = pd.DataFrame(grid_search.cv_results_)[
                    ["param_n_components", "mean_test_score"]
                ]
                cluster_df["mean_test_score"] = -cluster_df["mean_test_score"]
                cluster_df = cluster_df.rename(
                    columns={
                        "param_n_components": "Number of components",
                        "mean_test_score": "BIC score",
                    }
                )
                opt_nb_component = np.argmin(cluster_df["BIC score"]) + param_grid['n_components'][0]
                print(f'Estimated nb clusters: {opt_nb_component}')
            else:
                opt_nb_component = nb_state
                print(f'Fixed nb clusters: {opt_nb_component}')

            cluster = GaussianMixture(n_components=opt_nb_component, max_iter=2000, n_init=30,
                                      covariance_type='tied').fit(np.vstack((all_alphas, all_ks)).T)

            if CLUSTER_IMAGE:
                H, xedges, yedges = np.histogram2d(all_alphas, all_ks, bins=[alpha_range, k_range])
                plt.figure(f'track{track}_exp{exp}_clusters')
                plt.imshow(H.T, extent=(alpha_range[0], alpha_range[-1], k_range[0], k_range[-1]), origin='lower', alpha=1.0)
                for i, cluster_mean in enumerate(cluster.means_):
                    plt.scatter(cluster_mean[0], cluster_mean[1], marker='+')
                plt.savefig(f'./cluster_images/track{track}_exp{exp}_clusters.png')

            print('Cluster centers: ', cluster.means_)
            print("--------------------------------")

            for fov in N_FOVS:
                # We read the corresponding csv file from the public data and extract the indices of the trajectories:
                if track == 2:
                    df = pd.read_csv(public_data_path + f'track_{track}/exp_{exp}/trajs_fov_{fov}.csv')
                else:
                    df = pd.read_csv(public_data_path + f'track_{track}/exp_{exp}/videos_fov_{fov}_track.csv')
                traj_idx = np.sort(df.traj_idx.unique())
                submission_file = path_exp + f'fov_{fov}.txt'
                outputs = ''

                # Loop over each index
                for idx in traj_idx:
                    # Get the lenght of the trajectory
                    x = np.array(df[df.traj_idx == idx])[:, 2]
                    y = np.array(df[df.traj_idx == idx])[:, 3]

                    cps, alphas, ks, states, seg_lengths = exhaustive_cps_search(x, y, WIN_WIDTHS,
                                                                                 EXT_WIDTH,
                                                                                 search_seuil=0.25,
                                                                                 cluster=cluster)

                    prediction_traj = [idx.astype(int)]
                    for k, alpha, state, cp in zip(ks, alphas, states, cps[1:]):
                        prediction_traj.append(10 ** k)
                        prediction_traj.append(alpha)
                        prediction_traj.append(state)
                        prediction_traj.append(cp)

                    outputs += ','.join(map(str, prediction_traj))
                    outputs += '\n'

                with open(submission_file, 'w') as f:
                    f.write(outputs)

            del seg_lengths
            del states
            del cps
            del alphas
            del ks
            del prediction_traj
            del all_alphas
            del all_ks
            del all_seg_lengths
            gc.collect()


if __name__ == "__main__":
    main()
