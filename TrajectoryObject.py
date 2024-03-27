import numpy as np


class TrajectoryObj:
    def __init__(self, index, localizations=None, max_pause=1):
        self.index = index
        self.paused_time = 0
        self.max_pause = max_pause
        self.trajectory_tuples = []
        self.localizations = localizations
        self.times = []
        self.closed = False
        self.color = (np.random.randint(0, 256)/255.,
                      np.random.randint(0, 256)/255.,
                      np.random.randint(0, 256)/255.)
        self.optimality = 0.
        self.positions = []

    def add_trajectory_tuple(self, next_time, next_position):
        assert self.localizations is not None
        self.trajectory_tuples.append((next_time, next_position))
        x, y, z = self.localizations[next_time][next_position][:3]
        self.positions.append([x, y, z])
        self.times.append(next_time)
        self.paused_time = 0

    def get_trajectory_tuples(self):
        return self.trajectory_tuples

    def add_trajectory_position(self, time, x, y, z):
        self.times.append(time)
        self.positions.append([x, y, z])
        self.paused_time = 0

    def get_positions(self):
        return np.array(self.positions)

    def trajectory_status(self):
        return self.closed

    def close(self):
        self.paused_time = 0
        self.closed = True

    def wait(self):
        if self.paused_time == self.max_pause:
            self.close()
            return self.trajectory_status()
        else:
            self.paused_time += 1
            return self.trajectory_status()

    def get_index(self):
        return self.index

    def get_times(self):
        return np.array(self.times)

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def set_trajectory_tuple(self, trajectory):
        self.trajectory_tuples = trajectory
        self.paused_time = 0

    def get_last_tuple(self):
        return self.trajectory_tuples[-1]

    def get_trajectory_length(self):
        return len(self.get_positions())

    def get_paused_time(self):
        return self.paused_time

    def set_optimality(self, val):
        self.optimality = val

    def get_optimality(self):
        return self.optimality

    def get_expected_pos(self, t):
        if len(self.get_times()) < t+1:
            return np.array(self.positions[-1]), None
        else:
            vector = (np.array(self.positions[-1]) - np.array(self.positions[-1 - t])) / t
            return np.array(self.positions[-1]) + (vector * (self.paused_time + 1)), np.sqrt(vector[0]**2 + vector[1]**2)

    def delete(self, cutoff=2):
        if len(self.positions) < cutoff:
            return True
        else:
            return False

    def get_inst_diffusion_coefs(self, time_interval, t_range=None, ndim=2):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        diff_coefs = []
        for i in range(len(considered_positions) - 1):
            j = i + 1
            prev_x, prev_y, prev_z = considered_positions[i]
            prev_t = considered_times[i]
            x, y, z = considered_positions[j]
            t = considered_times[j]
            diff_coef = ((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2) / (2 * ndim * (t - prev_t))
            diff_coefs.append(diff_coef)
        diff_coefs = np.array(diff_coefs)
        diff_coefs_intervals = []
        for i in range(len(diff_coefs)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            diff_coefs_intervals.append(np.mean(diff_coefs[max(0, left_idx):min(len(diff_coefs), right_idx+1)]))

        # make length equal to length of xy pos
        #diff_coefs_intervals.append(0.0)
        return np.array(diff_coefs_intervals)

    def get_trajectory_angles(self, time_interval, t_range=None):
        """
        available only for 2D data.
        """
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        angles = []
        for i in range(len(considered_positions) - 2):
            prev_x, prev_y, prev_z = considered_positions[i]
            prev_t = considered_times[i]
            x, y, z = considered_positions[i+1]
            t = considered_times[i+1]
            next_x, next_y, next_z = considered_positions[i+2]
            next_t = considered_times[i+2]
            vec_prev_cur = np.array([x - prev_x, y - prev_y, z - prev_z]) / (t - prev_t)
            vec_cur_next = np.array([next_x - x, next_y - y, next_z - z]) / (next_t - t)

            ang = np.arccos((vec_prev_cur @ vec_cur_next) /
                            (np.sqrt(vec_prev_cur[0] ** 2 + vec_prev_cur[1] ** 2 + vec_prev_cur[2] ** 2)
                             * np.sqrt(vec_cur_next[0] ** 2 + vec_cur_next[1] ** 2 + vec_cur_next[2] ** 2)))
            angles.append(ang)
        angles = np.array(angles)

        angles_intervals = []
        for i in range(len(angles)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            angles_intervals.append(np.mean(angles[max(0, left_idx):min(len(angles), right_idx+1)]))

        # make length equal to length of xy pos
        angles_intervals.append(0.0)
        angles_intervals.append(0.0)
        return np.array(angles_intervals)

    def get_msd(self, time_interval, t_range=None):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        MSD = []
        for (x, y, z), t in zip(considered_positions, considered_times):
            MSD.append(np.sqrt((x - considered_positions[0][0]) ** 2 +
                               (y - considered_positions[0][1]) ** 2 +
                               (z - considered_positions[0][2]) ** 2))
        MSD = np.array(MSD)

        MSD_intervals = []
        for i in range(len(MSD)):
            left_idx = i - time_interval//2
            right_idx = i + time_interval//2
            MSD_intervals.append(np.mean(MSD[max(0, left_idx):min(len(MSD), right_idx+1)]))
        return np.array(MSD_intervals)

    def get_density(self, radius, t_range=None):
        if t_range is None:
            t_range = [0, len(self.get_positions())]
        considered_positions = self.get_positions()[t_range[0]: t_range[1]]
        considered_times = self.get_times()[t_range[0]: t_range[1]]

        density = []
        for i in range(len(considered_positions)):
            nb = 0
            for (x, y, z), t in zip(considered_positions, considered_times):
                disp = np.sqrt((x - considered_positions[i][0]) ** 2 +
                               (y - considered_positions[i][1]) ** 2 +
                               (z - considered_positions[i][2]) ** 2)
                if disp < radius:
                    nb += 1
            density.append(nb)
        return np.array(density).astype(np.float32)
