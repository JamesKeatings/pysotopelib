import numpy as np
from scipy import constants
import re

def Beta(v: float):
    return v / constants.c

def Gamma(beta: float):
    gamma = 1/np.sqrt(1-beta**2)
    return gamma

def RelDopplerShift(E_0: float, beta: float, theta: float):
    E = E_0 * np.sqrt( 1 - beta**2) / (1 - beta * np.cos(theta))
    return E

def CoulombBarrier(A1: int, Z1: int, A2: int, Z2: int):
    R = 1.16 * (np.cbrt(A1) + np.cbrt(A2))
    E_barrier = 1.44 * Z1 * Z2 / R
    return E_barrier

def RateEstimate(xsection: float, beamcurrent: float, A_target: int):
    rate = 3763.8 * xsection * beamcurrent / A_target
    return rate

def NChooseR(n, r):
    if r == 0:
        return 1
    if r > n // 2:
        return NChooseR(n, n - r)    
    res = 1
    for k in range(1, r + 1):
        res *= n - k + 1
        res //= k
    return res

def GammaEstimate(efficiency, fold, multiplicity):
    efficiency2 = efficiency / 100.0
    nchooser = NChooseR(fold, multiplicity)
    count = nchooser * (efficiency2 ** multiplicity) * ((1 - efficiency2) ** (fold - multiplicity))
    return count 
    
def GrazingAngle(A1: int, Z1: int , A2: int, Z2: int, E_lab: float):
    barrier = CoulombBarrier(A1, Z1, A2, Z2) 
    theta = 180. / np.pi * 2 * np.arcsin(1 / (2. * E_lab / barrier - 1));
    
    return theta

def GrazingAngle2(A1: int, Z1: int , A2: int, Z2: int, E_lab: float):
    reduced_mass = float(A1) * float(A2) / (A1 + A2)
    R_0 = 1.4
    R = R_0 * (np.cbrt(A1) + np.cbrt(A2))
    E_per_A = E_lab / A1
    #e = 1.60217663E-19
    #epsilon_0 = 8.8541878128E-12
    barrier = constants.e * constants.e / (4. * np.pi * constants.epsilon_0 * 1E-15) * Z1 * Z2 / R
    epsilon_c = (barrier / constants.e) / reduced_mass / 1E6
    temp = epsilon_c / (2. * E_per_A - epsilon_c)
    print(f"R = {R}, E/A = {E_per_A}, barrier = {barrier}, epsilon_c = {epsilon_c}, temp = {temp}")
    theta_gr_CM = 180. / np.pi * 2. * np.arcsin( epsilon_c / (2. * E_per_A - epsilon_c))
    theta_gr_LAB = 180. / np.pi * np.arctan(np.sin(np.pi / 180. * theta_gr_CM) / (np.cos(np.pi / 180. * theta_gr_CM) + float(A1) / float(A2)))
    return theta_gr_LAB

def GrazingAngle3(A1: int, Z1: int, A2: int, Z2: int, Tlab: float):
    R1 = 1.28 * np.cbrt(A1) - 0.76 + 0.8 * np.cbrt(A1)
    R2 = 1.28 * np.cbrt(A2) - 0.76 + 0.8 * np.cbrt(A2)
    C1 = R1 * (1 - R1**(-2))
    C2 = R2 * (1 - R2**(-2))
    Rint = C1 + C2 + 4.49 - (C1 + C2) / 6.35
    EperA = Tlab/A1
    numerator = 2.88 * Z1 * Z2 * (931.5 + EperA)
    denominator = A1 * ( EperA**2 + 1863 * EperA)

    theta = numerator/denominator / Rint
    theta_deg = theta * 180. / np.pi
    return theta_deg

def weisskopf_estimate(multipolarity: str, energy: float, mass: int) -> float:
    """
    Calculate the Weisskopf single-particle transition probability (Tsp) for a given multipolarity.

    Parameters:
    - multipolarity (str): Transition multipolarity, e.g., 'E1', 'M2'
    - energy (float): Gamma-ray energy Eγ in MeV
    - mass (int): Mass number A

    Returns:
    - float: Transition probability in s⁻¹
    """

    # Validate and parse multipolarity
    match = re.fullmatch(r'([EM])(\d+)', multipolarity.upper())
    if not match:
        raise ValueError("Multipolarity must be in the form 'E1', 'M2', etc.")
    
    type_, l_str = match.groups()
    l = int(l_str)

    energy = energy/1000
    
    if type_ == 'E':
        if l == 1:
            return 1.025e14 * energy**3 * mass**(2/3)
        elif l == 2:
            return 7.276e7 * energy**5 * mass**(4/3)
        elif l == 3:
            return 3.339e1 * energy**7 * mass**2
        elif l == 4:
            return 1.1e-5 * energy**9 * mass**(8/3)
        else:
            raise ValueError(f"E{l} not supported. Only E1, E2, E3, E4 are implemented.")
    elif type_ == 'M':
        if l == 1:
            return 5.6e13 * energy**3
        elif l == 2:
            return 3.56e7 * energy**5 * mass**(2/3)
        elif l == 3:
            return 16 * energy**7 * mass**(4/3)
        elif l ==4:
            return 4.5e-6 * energy **9 * mass**2
        else:
            raise ValueError(f"M{l} not supported. Only M1, M2, M3, M4 are implemented.")


