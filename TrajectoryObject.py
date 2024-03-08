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
        return self.times

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

    def get_diffusion_coefs(self):
        diff_coefs = []
        for i in range(len(self.get_positions()) - 1):
            j = i + 1
            prev_x, prev_y, prev_z = self.get_positions()[i]
            prev_t = self.get_times()[i]
            x,y,z = self.get_positions()[j]
            t = self.get_times()[j]
            diff_coef = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2) / (t - prev_t)
            diff_coefs.append(diff_coef)
        return diff_coefs
