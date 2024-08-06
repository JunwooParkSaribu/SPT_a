import numpy as np
from TrajectoryObject import TrajectoryObj
from numba.typed import Dict
from numba.core import types
from ImageModule import read_tif


def read_trajectory(file: str, andi_gt=False) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx', 'csv']

    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        localizations = {}
        tmp = {}
        try:
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            for line in lines:
                temp = line.split('\t')
                x_pos = float(temp[1].strip())
                y_pos = float(temp[2].strip())
                z_pos = 0.
                time_step = float(temp[3].strip())
                if time_step in tmp:
                    tmp[time_step].append([x_pos, y_pos, z_pos])
                else:
                    tmp[time_step] = [[x_pos, y_pos, z_pos]]

            time_steps = np.sort(np.array(list(tmp.keys())))
            first_frame, last_frame = time_steps[0], time_steps[-1]
            steps = np.arange(int(np.round(first_frame * 100)), int(np.round(last_frame * 100)) + 1)
            for step in steps:
                if step/100 in tmp:
                    localizations[step] = tmp[step/100]
                else:
                    localizations[step] = []
            return localizations
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
    else:
        try:
            trajectory_list = []
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            nb_traj = 0
            old_index = -999
            for line in lines[1:]:
                temp = line.split(',')
                index = int(float(temp[0].strip()))
                frame = int(float(temp[1].strip()))
                x_pos = float(temp[2].strip())
                y_pos = float(temp[3].strip())
                if andi_gt:
                    x_pos = float(temp[3].strip())
                    y_pos = float(temp[2].strip())
                if len(temp) > 4:
                    z_pos = float(temp[4].strip())
                else:
                    z_pos = 0.0

                if index != old_index:
                    nb_traj += 1
                    trajectory_list.append(TrajectoryObj(index=index, max_pause=5))
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame, x_pos, y_pos, z_pos)
                else:
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame, x_pos, y_pos, z_pos)
                old_index = index
            return trajectory_list
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)


def write_trajectory(file: str, trajectory_list: list):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = 'traj_idx,frame,x,y,z\n'
            for trajectory_obj in trajectory_list:
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{trajectory_obj.get_index()},{time-1},{xpos},{ypos},{zpos}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_trxyt(file: str, trajectory_list: list, pixel_microns=1.0, frame_rate=1.0):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = ''
            for index, trajectory_obj in enumerate(trajectory_list):
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{index}\t{xpos * pixel_microns:.5f}\t{ypos * pixel_microns:.5f}\t{time * frame_rate:.3f}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_andi2_label(model_labels, filename: str):
    try:
        with open(filename, 'w', encoding="utf-8") as f:
            input = f''
            for traj_idx in range(model_labels.shape[1]):
                input += f'{traj_idx},'
                changepoints, alphas, Ds, state_nums = label_continuous_to_list(model_labels[:, traj_idx, :])
                for cp, alpha, D, state_num in zip(changepoints, alphas, Ds, state_nums):
                    input += f'{D},{alpha},{state_num},{cp},'
                input += f'\n'
            f.write(input)
    except Exception as e:
        print(f"Unexpected error: {e}")


def read_mosaic(file: str) -> dict:
    filetypes = ['csv']
    localizations = {}
    tmp = {}
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    try:
        with open(file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')[1:]
        for line in lines:
            temp = line.split(',')
            x_pos = float(temp[3].strip())
            y_pos = float(temp[4].strip())
            time_step = (float(temp[2].strip()) + 1)/100.
            if time_step in tmp:
                tmp[time_step].append([x_pos, y_pos, 0.])
            else:
                tmp[time_step] = [[x_pos, y_pos, 0.]]

        time_steps = np.sort(np.array(list(tmp.keys())))
        first_frame, last_frame = time_steps[0], time_steps[-1]
        steps = np.arange(int(first_frame * 100), int(last_frame * 100) + 1)
        for step in steps:
            if step/100 in tmp:
                localizations[step] = tmp[step/100]
            else:
                localizations[step] = []
        return localizations
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_localization(output_dir, coords, all_pdfs, infos):
    lines = f'frame,x,y,z,xvar,yvar,rho,norm_cst,intensity,window_size\n'
    for frame, (coord, pdfs, info) in enumerate(zip(coords, all_pdfs, infos)):
        for pos, (x_var, y_var, rho, amp), pdf in zip(coord, info, pdfs):
            window_size = int(np.sqrt(len(pdf)))
            peak_val = pdf[int((len(pdf) - 1) / 2)]
            lines += f'{frame + 1}'
            if len(pos) == 3:
                lines += f',{pos[1]},{pos[0]},{pos[2]}'
            elif len(pos) == 2:
                lines += f',{pos[1]},{pos[0]},0.0'
            elif len(pos) == 1:
                lines += f',{pos[0]},0.0,0.0'
            else:
                print(f'Localization writing Err')
                raise Exception
            lines += f',{x_var},{y_var},{rho},{amp},{peak_val},{window_size}'
            lines += f'\n'

    with open(f'{output_dir}_loc.csv', 'w') as f:
        f.write(lines)


def read_localization(input_file, video=None):
    locals = {}
    locals_info = {}
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split('\n')[0].split(',')
                if int(line[0]) not in locals:
                    locals[int(line[0])] = []
                    locals_info[int(line[0])] = []
                pos_line = []
                info_line = []
                for dt in line[1:4]:
                    pos_line.append(np.round(float(dt), 7))
                for dt in line[4:]:
                    info_line.append(np.round(float(dt), 7))
                locals[int(line[0])].append(pos_line)
                locals_info[int(line[0])].append(info_line)
        if video is None:
            max_t = np.max(list(locals.keys()))
        else:
            max_t = len(video)
        for t in np.arange(1, max_t+1):
            if t not in locals:
                locals[t] = [[]]
                locals_info[t] = [[]]

        numba_locals = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:, :],
        )
        numba_locals_info = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:, :],
        )

        for t in locals.keys():
            numba_locals[t] = np.array(locals[t])
            numba_locals_info[t] = np.array(locals_info[t])
    except Exception as e:
        print(f'Err msg: {e}')
        print('here')
        exit(1)
    return numba_locals, numba_locals_info


def read_andi2_trajectory_label(input_file, index=None):
    trajectory = {}
    if type(input_file) is str:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\n')[0].split(',')
                index = int(float(line[0]))
                line = line[1:]

                diff_coefs = []
                alphas = []
                state_nums = []
                cps = [0]
                cp_state = [False]
                turn_on = False
                for i, item in enumerate(line):
                    if i % 4 == 0:
                        turn_on = False
                        diff_coef = float(item)
                    elif i % 4 == 1:
                        alpha = float(item)
                    elif i % 4 == 2:
                        state_num = int(float(item))
                    elif i % 4 == 3:
                        cp = int(float(item))
                        turn_on = True
                    if turn_on:
                        cp_range = cp - cps[-1]

                        diff_coefs.extend([diff_coef] * cp_range)
                        alphas.extend([alpha] * cp_range)
                        state_nums.extend([state_num] * cp_range)
                        cps.extend([cp] * cp_range)
                        cp_state.extend([0] * (cp_range-1))
                        if i != len(line) - 1:
                            cp_state.append(1)
                        trajectory[index] = [np.array(diff_coefs), np.array(alphas), np.array(state_nums), np.array(cp_state)]
    else:
        trajectory = {}
        diff_coefs = []
        alphas = []
        state_nums = []
        cps = [0]
        cp_state = [False]
        turn_on = False
        if index is None:
            index = 0

        for traj_length, label_list in enumerate(np.array(input_file).T):
            for i, label in enumerate(label_list):
                if i % 4 == 0:
                    turn_on = False
                    diff_coef = float(label)
                elif i % 4 == 1:
                    alpha = float(label)
                elif i % 4 == 2:
                    state_num = int(float(label))
                elif i % 4 == 3:
                    cp = int(float(label))
                    turn_on = True
                if turn_on:
                    cp_range = cp - cps[-1]

                    diff_coefs.extend([diff_coef] * cp_range)
                    alphas.extend([alpha] * cp_range)
                    state_nums.extend([state_num] * cp_range)
                    cps.extend([cp] * cp_range)
                    cp_state.extend([0] * (cp_range-1))
                    if traj_length != len(input_file[0]) - 1:
                        cp_state.append(1)
                    trajectory[index] = [np.array(diff_coefs), np.array(alphas), np.array(state_nums), np.array(cp_state)]
    return trajectory


def read_parameters(param_file):
    params = {'localization': {}, 'tracking': {}}
    try:
        with open(param_file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')

        for line in lines:
            if 'video' in line.lower():
                params['localization']['VIDEO'] = line.strip().split('=')[1]
            if 'output_dir' in line.lower():
                params['localization']['OUTPUT_DIR'] = line.strip().split('=')[1]
            if 'sigma' in line.lower():
                params['localization']['SIGMA'] = float(eval(line.strip().split('=')[1]))
            if 'min_win' in line.lower():
                params['localization']['MIN_WIN'] = int(eval(line.strip().split('=')[1]))
            if 'max_win' in line.lower():
                params['localization']['MAX_WIN'] = int(eval(line.strip().split('=')[1]))
            if 'threshold_alpha' in line.lower():
                params['localization']['THRES_ALPHA'] = float(eval(line.strip().split('=')[1]))
            if 'deflation_loop_in_backward' in line.lower():
                params['localization']['DEFLATION_LOOP_IN_BACKWARD'] = int(eval(line.strip().split('=')[1]))
            if 'loc_parallel' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['localization']['PARALLEL'] = True
                else:
                    params['localization']['PARALLEL'] = False

            if 'core' in line.lower():
                params['localization']['CORE'] = int(eval(line.strip().split('=')[1]))
            if 'div_q' in line.lower():
                params['localization']['DIV_Q'] = int(eval(line.strip().split('=')[1]))
            if 'shift' in line.lower():
                params['localization']['SHIFT'] = int(eval(line.strip().split('=')[1]))
            if 'gauss_seidel_decomp' in line.lower():
                params['localization']['GAUSS_SEIDEL_DECOMP'] = int(eval(line.strip().split('=')[1]))
            if 'loc_visualization' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['localization']['LOC_VISUALIZATION'] = True
                else:
                    params['localization']['LOC_VISUALIZATION'] = False

            if 'video' in line.lower():
                params['tracking']['VIDEO'] = line.strip().split('=')[1]
            if 'output_dir' in line.lower():
                params['tracking']['OUTPUT_DIR'] = line.strip().split('=')[1]
            if 'pixel_microns' in line.lower():
                params['tracking']['PIXEL_MICRONS'] = float(eval(line.strip().split('=')[1]))
            if 'frame_rate' in line.lower():
                params['tracking']['FRAME_RATE'] = float(eval(line.strip().split('=')[1]))
            if 'blink_lag' in line.lower():
                params['tracking']['BLINK_LAG'] = int(eval(line.strip().split('=')[1]))
            if 'cutoff' in line.lower():
                params['tracking']['CUTOFF'] = int(eval(line.strip().split('=')[1]))
            if 'amp_max_len' in line.lower():
                params['tracking']['AMP_MAX_LEN'] = float(eval(line.strip().split('=')[1]))
            if 'tracking_parallel' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['tracking']['VAR_PARALLEL'] = True
                else:
                    params['tracking']['VAR_PARALLEL'] = False
            if 'track_visualization' in line.lower():
                if 'true' in line.lower().strip().split('=')[1]:
                    params['tracking']['TRACK_VISUALIZATION'] = True
                else:
                    params['tracking']['TRACK_VISUALIZATION'] = False

        return params
    except Exception as e:
        print(f"Unexpected error, check the config file")
        print(f'ERR msg: {e}')
        exit(1)


def check_video_ext(args, andi2=False):
    if len(args) == 0:
        print(f'no input file')
        exit(1)
    if '.tif' not in args and '.tiff' not in args:
        print(f'video format err, only .tif or .tiff are acceptable')
        exit(1)
    else:
        return read_tif(args)
