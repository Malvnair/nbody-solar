import numpy as np
from .constants import G

def total_energy(state, epsilon_soft: float, pair_weight=lambda i,j: 1.0):
    KE = 0.0
    for b in state:
        if b[6] <= 0: continue
        v2 = b[3]*b[3] + b[4]*b[4] + b[5]*b[5]
        KE += 0.5 * b[6] * v2
    PE = 0.0
    n = len(state)
    for i in range(n):
        if state[i][6] <= 0: continue
        for j in range(i+1, n):
            if state[j][6] <= 0: continue
            r = np.linalg.norm(np.array(state[i][:3]) - np.array(state[j][:3]))
            if r > epsilon_soft:
                w = pair_weight(i, j)
                PE -= G * state[i][6] * state[j][6] / r
    return KE + PE  

def total_momentum(state):
    P = np.zeros(3)
    for b in state:
        if b[6] <= 0: continue
        P += b[6] * np.array(b[3:6])
    return P

def total_angular_momentum(state):
    L = np.zeros(3)
    for b in state:
        if b[6] <= 0: continue
        r = np.array(b[:3]); v = np.array(b[3:6])
        L += b[6] * np.cross(r, v)
    return L

def check_conservation(state, E0, P0, L0, tol: float, epsilon_soft: float):
    E = total_energy(state, epsilon_soft)
    P = total_momentum(state)
    L = total_angular_momentum(state)
    dE = abs((E - E0) / E0) if E0 != 0 else 0.0
    dP = np.linalg.norm(P - P0) / (np.linalg.norm(P0) or 1.0)
    dL = np.linalg.norm(L - L0) / (np.linalg.norm(L0) or 1.0)
    return dict(energy=E, momentum=P, angular_momentum=L,
                energy_error=dE, momentum_error=dP, angular_momentum_error=dL,
                conservation_ok=(dE < tol and dP < tol and dL < tol))
