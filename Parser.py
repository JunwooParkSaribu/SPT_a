import scipy.io
import numpy as np
from TrajectoryObject import TrajectoryObj
from XmlModule import write_xml
WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
WINDOWS_PATH = 'C:/Users/jwoo/Desktop'

if __name__ == '__main__':
    filename = f'Test_Bead_Diffusion_Analysis.mat'
    mat_file = f'{WINDOWS_PATH}/{filename}'
    output_xml = f'{WINDOWS_PATH}/ground_truth.xml'
    mat_tracks = scipy.io.loadmat(mat_file)
    tracks = mat_tracks['tracks'][0]

    ## x[0], y[1], z[2], frames[3], tracklength[4], tracklength_with_blink[5], n_subtracks[6], subtracks_frame_start[7],
    ## subtracks_frame_end[8], subtracks_length[9], off_times[10], segments[11], molecule_id[12], fluorophore_id[13], tot_tracklength[14]

    trajectory_list = []
    for index, track in enumerate(tracks):
        traj = TrajectoryObj(index=index)
        track_length = track[4][0][0]
        xs = track[0].reshape(track_length)
        ys = track[1].reshape(track_length)
        if len(track[2]) == 0:
            zs = np.zeros(track_length)
        else:
            zs = track[2].reshape(track_length)
        frames = track[3][0]

        for x, y, z, frame in zip(xs, ys, zs, frames):
            traj.add_trajectory_position(frame, x, y, 0)
        trajectory_list.append(traj)

    write_xml(output_file=output_xml, trajectory_list=trajectory_list)
