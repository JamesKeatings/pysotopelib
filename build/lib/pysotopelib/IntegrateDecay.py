import numpy as np

def integrate_decay(N0: float, decay_constant: float, start: float, stop: float):
    area_start = np.exp(-decay_constant * start)
    area_stop = np.exp(-decay_constant * stop)
    area = N0 * (area_start - area_stop)
    return area

def integrate_ddecay(N_0, decay_constant_a, decay_constant_b, start, stop):
    area_start = -N_0 * decay_constant_a * decay_constant_b * (np.exp(-decay_constant_b * start) / decay_constant_b - np.exp(-decay_constant_a * start) / decay_constant_a) / (decay_constant_b - decay_constant_a)
    area_stop = -N_0 * decay_constant_a * decay_constant_b * (np.exp(-decay_constant_b * stop) / decay_constant_b - np.exp(-decay_constant_a * stop) / decay_constant_a) / (decay_constant_b - decay_constant_a)
    area = area_start - area_stop
    return area
