import os
import numpy as np
from andi_datasets.datasets_phenom import datasets_phenom


public_data_path = 'public_data_validation/'
path_results = 'res_validation/'
if not os.path.exists(path_results):
    os.makedirs(path_results)


def write_config(exp_n, fov_n):
    with open('config.txt', 'w') as f:
        input_str = (f'VIDEO=./analysis_modeling/public_data/track_1/exp_{exp_n}/videos_fov_{fov_n}.tiff\n'
                     f'OUTPUT_DIR=./analysis_modeling/public_data/track_1/exp_{exp_n}\n'
                     f'# LOCALIZATION\n'
                     f'SIGMA = 4.0\n'
                     f'MIN_WIN = 5\n'
                     f'MAX_WIN = 7\n'
                     f'THRESHOLD_ALPHA = 1.0\n'
                     f'DEFLATION_LOOP_IN_BACKWARD = 2\n'
                     f'LOC_PARALLEL = True\n'
                     f'CORE = 4\n'
                     f'DIV_Q = 50\n'
                     f'SHIFT = 2\n'
                     f'GAUSS_SEIDEL_DECOMP = 2\n'
                     f'LOC_VISUALIZATION = False\n'
                     f'# TRACKING\n'
                     f'PIXEL_MICRONS = 1.0\n'
                     f'FRAME_RATE = 1.0\n'
                     f'BLINK_LAG = 1\n'
                     f'CUTOFF = 0\n'
                     f'TRACKING_PARALLEL = False\n'
                     f'AMP_MAX_LEN = 1.7\n'
                     f'TRACK_VISUALIZATION = False\n')
        f.write(input_str)


for track in [1, 2]:
    # Create the folder of the track if it does not exists
    path_track = path_results + f'track_{track}/'
    if not os.path.exists(path_track):
        os.makedirs(path_track)

    for exp in range(10):
        # Create the folder of the experiment if it does not exits
        path_exp = path_track + f'exp_{exp}/'
        if not os.path.exists(path_exp):
            os.makedirs(path_exp)
        file_name = path_exp + 'ensemble_labels.txt'

        with open(file_name, 'a') as f:
            # Save the model (random) and the number of states (2 in this case)
            model_name = np.random.choice(datasets_phenom().avail_models_name, size=1)[0]
            f.write(f'model: {model_name}; num_state: {2} \n')

            # Create some dummy data for 2 states. This means 2 columns
            # and 5 rows
            data = np.random.rand(5, 2)

            data[-1, :] /= data[-1, :].sum()

            # Save the data in the corresponding ensemble file
            np.savetxt(f, data, delimiter=';')

# Define the number of experiments and number of FOVS
N_EXP = 12
N_FOVS = 30

path_results = 'res/'

for exp in range(0, N_EXP):
    for fov in range(0, N_FOVS):
        write_config(exp, fov)
        with open("Localization.py") as file:
            exec(file.read())
        with open("main.py") as file:
            exec(file.read())




