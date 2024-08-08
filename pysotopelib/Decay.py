import numpy as np

def DCtoHL(decay_constant: float):
    return np.log(2)/decay_constant

def HLtoDC(halflife: float):
    return np.log(2)/halflife

def decay(N0: float, decay_constant: float, t: float):

    return N0 * np.exp(-1*decay_constant*t)

def ddecay(N0: float, decay_constant_a: float, decay_constant_b: float, t: float):

    return N0 * (decay_constant_a/(decay_constant_b-decay_constant_a)) * (np.exp(-decay_constant_a*t)-np.exp(-decay_constant_b*t))

def gddecay(N0: float, decay_constant_a: float, decay_constant_b: float, decay_constant_c: float, t: float):
    term1 = np.exp(-decay_constant_a*t) / ((decay_constant_b-decay_constant_a)*(decay_constant_c-decay_constant_a))
    term2 = np.exp(-decay_constant_b*t) / ((decay_constant_a-decay_constant_b)*(decay_constant_c-decay_constant_b))
    term3 = np.exp(-decay_constant_c*t) / ((decay_constant_a-decay_constant_c)*(decay_constant_b-decay_constant_c))
    return N0 * decay_constant_a * decay_constant_b * (term1 + term2 + term3)

def ggddecay(N0: float, decay_constant_a: float, decay_constant_b: float, decay_constant_c:float, decay_constant_d: float, t: float):
    term1 = np.exp(-decay_constant_a*t) / ((decay_constant_b - decay_constant_a)*(decay_constant_c - decay_constant_a)*(decay_constant_d - decay_constant_a)) 
    term2 = np.exp(-decay_constant_b*t) / ((decay_constant_a - decay_constant_b)*(decay_constant_c - decay_constant_b)*(decay_constant_d - decay_constant_b))
    term3 = np.exp(-decay_constant_c*t) / ((decay_constant_a - decay_constant_c)*(decay_constant_b - decay_constant_c)*(decay_constant_d - decay_constant_c))
    term4 = np.exp(-decay_constant_d*t) / ((decay_constant_a - decay_constant_d)*(decay_constant_b - decay_constant_d)*(decay_constant_c - decay_constant_d))
    return N0 * decay_constant_a * decay_constant_b * decay_constant_c * (term1 + term2 + term3 + term4)
