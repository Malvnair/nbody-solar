import numpy as np
from .constants import G

def volume_add_radius(Ra: float, Rb: float) -> float:
    return (Ra**3 + Rb**3) ** (1/3)

def is_collision(body_i, body_j) -> bool:
    ri = np.array(body_i[:3]); rj = np.array(body_j[:3])
    return np.linalg.norm(ri - rj) <= (body_i[7] + body_j[7])

def compute_acceleration_softened(i: int, state, epsilon: float):
    ax = ay = az = 0.0
    xi, yi, zi = state[i][:3]
    for j in range(len(state)):
        if j == i: continue
        mj = state[j][6]
        if mj <= 0: continue
        xj, yj, zj = state[j][:3]
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        r2_soft = dx*dx + dy*dy + dz*dz + epsilon*epsilon
        if r2_soft < epsilon*epsilon: r2_soft = epsilon*epsilon
        inv_r3 = 1.0 / (r2_soft ** 1.5)
        ax += G * mj * dx * inv_r3
        ay += G * mj * dy * inv_r3
        az += G * mj * dz * inv_r3
    return ax, ay, az
