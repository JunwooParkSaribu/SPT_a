from TrajectoryObject import TrajectoryObj as trajobj
import numpy as np


def write_xml(output_file: str, trajectory_list: list, snr='4',
              density='low', scenario='VESICLE', cutoff=0):
    input_str = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    input_str += '<root>\n'
    input_str += (f'<TrackContestISBI2012 SNR="{snr}" density="{density}" '
                  f'generationDateTime="Mon Nov 20 09:00:00 CET 2023" '
                  f'info="http://bioimageanalysis.org/track/" scenario="{scenario}">\n')
    with open(output_file, 'w', encoding="utf-8") as f:
        for trajectory_obj in trajectory_list:
            if trajectory_obj.get_trajectory_length() >= cutoff:
                input_str += '<particle>\n'
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'<detection t="{time - 1}" x="{xpos}" y="{ypos}" z="{zpos}"/>\n'
                input_str += '</particle>\n'
        input_str += '</TrackContestISBI2012>\n'
        input_str += '</root>\n'
        f.write(input_str)


def xml_to_object(input_file):
    obj_index = 0
    trajectory_list = []
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()[3:]
        for line in lines:
            l = line.split('\n')[0]
            if l == '<particle>':
                trajectory_list.append(trajobj(index=obj_index, max_pause=5))
            elif l == '</particle>':
                obj_index += 1
            elif 'detection' in l:
                c = l.split('\"')
                t, x, y, z = int(c[1]) + 1, float(c[3]), float(c[5]), float(c[7])
                trajectory_list[obj_index].add_trajectory_position(t, x, y, z)
    return trajectory_list


def read_xml(input_file):
    localizations = {}
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()[3:]
        for line in lines:
            l = line.split('\n')[0]
            if 'detection' in l:
                c = l.split('\"')
                t, x, y, z = int(c[1]) + 1, float(c[3]), float(c[5]), float(c[7])
                if t in localizations:
                    localizations[t].append([x, y, z])
                else:
                    localizations[t] = [[x, y, z]]
    return localizations


def trxyt_to_xml(input_file: str, output_file: str, cutoff=0):
    filetypes = ['trxyt', 'trx']
    trajectories = {}
    # Check filetype.
    assert input_file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    try:
        with open(input_file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')
        for line in lines:
            temp = line.split('\t')
            index = int(temp[0].strip())
            x_pos = float(temp[1].strip())
            y_pos = float(temp[2].strip())
            time_step = int(np.round(((float(temp[3].strip()) * 100) - 1)))
            if index in trajectories:
                trajectories[index].append([x_pos, y_pos, time_step])
            else:
                trajectories[index] = [[x_pos, y_pos, time_step]]
        f.close()

        with open(output_file, 'w', encoding="utf-8") as fxml:
            input = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            input += '<root>\n'
            input += ('<TrackContestISBI2012 SNR="7" density="mid" '
                      'generationDateTime="Mon Mar 12 17:20:58 CET 2012" info="http://bioimageanalysis.org/track/" scenario="MICROTUBULE">\n')
            for index in list(trajectories.keys()):
                if len(trajectories[index]) >= cutoff:
                    input += '<particle>\n'
                    for xpos, ypos, t in trajectories[index]:
                        input += f'<detection t="{t}" x="{xpos}" y="{ypos}" z="0"/>\n'
                    input += '</particle>\n'
            input += '</TrackContestISBI2012>\n'
            input += '</root>\n'
            fxml.write(input)
    except Exception as e:
        print(f"Unexpected error, check the file: {input_file}")
        print(e)


def mosaic_to_xml(input_file: str, output_file: str, cutoff=0):
    filetypes = ['csv']
    tmp = {}
    # Check filetype.
    assert input_file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    try:
        with open(input_file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')[1:]
        for line in lines:
            temp = line.split(',')
            x_pos = float(temp[3].strip())
            y_pos = float(temp[4].strip())
            time_step = int(temp[2].strip()) + 1
            obj_index = int(temp[1].strip())
            if obj_index in tmp:
                tmp[obj_index].add_trajectory_position(time_step, x_pos, y_pos, 0)
            else:
                tmp[obj_index] = trajobj(index=obj_index, localizations=None, max_pause=0)
                tmp[obj_index].add_trajectory_position(time_step, x_pos, y_pos, 0)

        objs_list = []
        for obj in tmp:
            objs_list.append(tmp[obj])
        write_xml(output_file=output_file, trajectory_list=objs_list, snr='7', cutoff=cutoff)
    except Exception as e:
        print(f"Unexpected error, check the file: {input_file}")
        print(e)
