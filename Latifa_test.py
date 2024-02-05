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
        self.color = (np.random.randint(0, 150)/255.,
                      np.random.randint(0, 255)/255.,
                      np.random.randint(0, 255)/255.)
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


def trajectory_to_segments(trajectory_list, blink_lag):
    segment_distrib = {lag: [] for lag in range(blink_lag + 1)}
    for traj_obj in trajectory_list:
        pos = traj_obj.get_positions()
        times = traj_obj.get_times()
        for lag in range(blink_lag + 1):
            for i in range(len(pos) - 1 - lag):
                x, y, z = pos[i]
                next_x, next_y, next_z = pos[i+1+lag]
                t = times[i]
                next_t = times[i+1+lag]
                if (int(next_t - t) - 1) in segment_distrib:
                    segment_distrib[int(next_t - t) - 1].append(
                        [np.sqrt((next_x - x)**2 + (next_y - y)**2 + (next_z - z)**2)]
                    )
    for lag in segment_distrib:
        segment_distrib[lag] = np.array(segment_distrib[lag])
    return segment_distrib


def xml_to_object(input_file):
    obj_index = 0
    trajectory_list = []
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()[3:]
        for line in lines:
            l = line.split('\n')[0]
            if l == '<particle>':
                trajectory_list.append(TrajectoryObj(index=obj_index, max_pause=5))
            elif l == '</particle>':
                obj_index += 1
            elif 'detection' in l:
                c = l.split('\"')
                t, x, y, z = int(c[1]) + 1, float(c[3]), float(c[5]), float(c[7])
                trajectory_list[obj_index].add_trajectory_position(t, x, y, z)
    return trajectory_list


def kl_divergence(distribution, ground_truth_xml):
    """
    @parameters
    distribution(1d array): your segment length distribution
    ground_truth_xml(ground-truth xml file): ground-truth
    """
    bins_ = np.arange(0, 50, 0.25)  # must use this bins_ to make segments distribution into a histogram
    ref_distribution = trajectory_to_segments(xml_to_object(ground_truth_xml), blink_lag=MAX_DELTA_T)
    ref_proba, bins = np.histogram(ref_distribution[DELTA_T], bins=bins_)
    if type(distribution) is str:
        distribution = trajectory_to_segments(xml_to_object(distribution), blink_lag=MAX_DELTA_T)
    proba, bins = np.histogram(distribution, bins=bins_)

    assert proba.shape == ref_proba.shape  # check the shape whether both inputs used bins_ or not.
    proba = np.array(proba, dtype=np.float64) + 1e-10
    ref_proba = np.array(ref_proba, dtype=np.float64) + 1e-10
    proba = proba / np.sum(proba)
    ref_proba = ref_proba / np.sum(ref_proba)
    return np.sum(proba * np.log(proba/ref_proba))


MAX_DELTA_T = 5  # don't need to touch it
DELTA_T = 0   # gap between frames (0: consecutive frame, 1: between frame 0 and 2, 1 and 3, and so on...)
ground_truth_xml = f'simulated_data/ground_truth/RECEPTOR snr 7 density low.xml'  # ground truth file path

"""
After collecting segments, you can make a histogram for a given delta_t with bins_.
and compare the histograms(ground-truth and yours) with kl-divergence.

If the result value equal to 0: exactly same as ground truth, perfect match,
else the value is higher: far from the ground-truth.
"""
WSL_PATH = '/mnt/c/Users/jwoo/Desktop/my_test1/receptor_7_low/receptor_7_low.xml'
mine = np.zeros(np.arange(0, 50, 0.25).shape)  # change this for your segments distribution.
result_value = kl_divergence(WSL_PATH, ground_truth_xml)
print(f'Result of KL-divergence entropy between two PDFs:{result_value}')
