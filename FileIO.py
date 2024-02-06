import numpy as np


def read_trajectory(file: str) -> dict:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx']
    localizations = {}
    tmp = {}
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
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


def write_trajectory(file: str, trajectory_list: list):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input = ''
            for trajectory_obj in trajectory_list:
                index = trajectory_obj.get_index()
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input += f'{str(index)}\t{xpos}\t{ypos}\t{zpos}\t{time/100}\n'
            f.write(input)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


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


def write_localization(output_dir, coords, infos):
    lines = f''
    for frame, (coord, info) in enumerate(zip(coords, infos)):
        for pos, (x_var, y_var, rho, amp) in zip(coord, info):
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
            lines += f',{x_var},{y_var},{rho},{amp}'
            lines += f'\n'

    with open(f'{output_dir}/localization.txt', 'w') as f:
        f.write(lines)


def read_localization(input_file):
    locals = {}
    locals_info = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\n')[0].split(',')
            if int(line[0]) not in locals:
                locals[int(line[0])] = []
                locals_info[int(line[0])] = []
            pos_line = []
            info_line = []
            for dt in line[1:4]:
                pos_line.append(np.round(float(dt), 5))
            for dt in line[4:]:
                info_line.append(np.round(float(dt), 5))
            locals[int(line[0])].append(pos_line)
            locals_info[int(line[0])].append(info_line)
    return locals, locals_info
