from scipy.optimize import linear_sum_assignment
import numpy as np


def hungarian_algo_max(cost):
    cost = -np.array(cost)
    print(cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    val = -cost[row_ind, col_ind].sum()
    return row_ind, col_ind, val
