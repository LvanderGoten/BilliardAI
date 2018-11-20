import numpy as np
from numba import jit, int64
from time import process_time_ns

a0 = np.arange(1, 10 ** 7)
b0 = np.arange(-10 ** 7, -1)


# @jit(int64[:](int64[:], int64[:]), nopython=True)
def sum_sequence(a, b):
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] - b[i]
    return result


start = process_time_ns()
sum_sequence(a0, b0)
duration = (process_time_ns() - start)/(10**6)
print(duration)