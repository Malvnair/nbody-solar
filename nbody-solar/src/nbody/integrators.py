import copy
import numpy as np
from .physics import compute_acceleration_softened

def velocity_verlet_step(state, dt: float, epsilon: float):
    n = len(state)
    next_state = copy.deepcopy(state)
    a_now = [compute_acceleration_softened(i, state, epsilon) for i in range(n)]

    for i in range(n):
        ax, ay, az = a_now[i]
        vx, vy, vz = state[i][3:6]
        next_state[i][0] += vx*dt + 0.5*ax*dt*dt
        next_state[i][1] += vy*dt + 0.5*ay*dt*dt
        next_state[i][2] += vz*dt + 0.5*az*dt*dt

    a_new = [compute_acceleration_softened(i, next_state, epsilon) for i in range(n)]

    for i in range(n):
        ax_old, ay_old, az_old = a_now[i]
        ax_new, ay_new, az_new = a_new[i]
        next_state[i][3] += 0.5 * (ax_old + ax_new) * dt
        next_state[i][4] += 0.5 * (ay_old + ay_new) * dt
        next_state[i][5] += 0.5 * (az_old + az_new) * dt

    return next_state

def adaptive_timestep(state, eta: float, dt_min: float, dt_max: float, epsilon: float):
    from .physics import compute_acceleration_softened
    dt_candidates = []
    n = len(state)
    for i in range(n):
        ax, ay, az = compute_acceleration_softened(i, state, epsilon)
        a_mag = (ax*ax + ay*ay + az*az) ** 0.5
        if a_mag > 1e-20:
            r_min = float('inf')
            ri = np.array(state[i][:3])
            for j in range(n):
                if i == j: continue
                if state[j][6] <= 0: continue
                rj = np.array(state[j][:3])
                r_min = min(r_min, np.linalg.norm(ri - rj))
            dt_i = eta * (r_min / a_mag) ** 0.5
            dt_candidates.append(dt_i)
    dt = min(dt_candidates) if dt_candidates else dt_max
    return max(dt_min, min(dt_max, dt))
