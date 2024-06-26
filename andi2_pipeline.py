import os
import numpy as np

N_EXP = 13
N_FOVS = 30
public_data_path = 'public_data_validation_v1'


def write_config(exp_n, fov_n):
    with open('config.txt', 'w') as f:
        input_str = (f'VIDEO=./analysis_modeling/{public_data_path}/track_1/exp_{exp_n}/videos_fov_{fov_n}.tiff\n'
                     f'OUTPUT_DIR=./analysis_modeling/{public_data_path}/track_1/exp_{exp_n}\n'
                     f'# LOCALIZATION\n'
                     f'SIGMA = 4.0\n'
                     f'MIN_WIN = 5\n'
                     f'MAX_WIN = 7\n'
                     f'THRESHOLD_ALPHA = 1.0\n'
                     f'DEFLATION_LOOP_IN_BACKWARD = 2\n'
                     f'LOC_PARALLEL = False\n'
                     f'CORE = 4\n'
                     f'DIV_Q = 100\n'
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


for exp in range(0, N_EXP):
    for fov in range(0, N_FOVS):
        write_config(exp, fov)
        with open("Localization.py") as file:
            exec(file.read())
        with open("main.py") as file:
            exec(file.read())
