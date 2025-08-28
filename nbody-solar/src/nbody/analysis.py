import numpy as np
import logging
import matplotlib.pyplot as plt
from .constants import AU
from .simulation import NBodySimulation

log = logging.getLogger(__name__)

def osculating_elements(state, body_idx: int, central_idx: int, G: float):
    import numpy as np
    r_vec = np.array(state[body_idx][:3]) - np.array(state[central_idx][:3])
    v_vec = np.array(state[body_idx][3:6]) - np.array(state[central_idx][3:6])
    r = np.linalg.norm(r_vec)
    v2 = float(np.dot(v_vec, v_vec))
    mu = G * state[central_idx][6]
    h_vec = np.cross(r_vec, v_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
    e = float(np.linalg.norm(e_vec))
    epsilon = v2/2 - mu/r
    a = -mu / (2*epsilon) if epsilon != 0 else np.inf
    q = a * (1 - e) if np.isfinite(a) else np.nan
    return {"a": a, "e": e, "q": q, "r": r}

def run_mercury_analysis(sim: NBodySimulation, n_frames: int, intruder_frames: tuple[int,int], G: float):
    data = {k: [] for k in ['t_days','r','a','e','q','x','y','energy_error','momentum_error']}
    activate_frame, deactivate_frame = intruder_frames

    for frame in range(n_frames):
        if frame == activate_frame:
            sim.set_intruder_trajectory(True)
        elif frame == deactivate_frame:
            sim.set_intruder_trajectory(False)

        sim.run_frame(sim.cfg.simulation.steps_per_frame)

        el = osculating_elements(sim.state, 1, 0, G)
        data['t_days'].append(sim.time/86400.0)
        data['r'].append(el['r']); data['a'].append(el['a'])
        data['e'].append(el['e']); data['q'].append(el['q'])
        data['x'].append(sim.state[1][0]); data['y'].append(sim.state[1][1])

        if sim.conservation_log:
            data['energy_error'].append(sim.conservation_log[-1]['energy_error'])
            data['momentum_error'].append(sim.conservation_log[-1]['momentum_error'])
        else:
            data['energy_error'].append(0.0); data['momentum_error'].append(0.0)

        if frame % 100 == 0:
            log.info("Frame %d/%d  t=%.1f d  dt=%.1fs  dE/E=%.2e",
                     frame, n_frames, sim.time/86400, sim.dt_current, data['energy_error'][-1])

    return {k: np.array(v) for k, v in data.items()}

def diagnostic_plots(baseline, perturbed):
    # Orbit XY
    plt.figure(figsize=(8, 8))
    plt.plot(baseline['x']/AU, baseline['y']/AU, label='Baseline', lw=1.5, alpha=0.7)
    plt.plot(perturbed['x']/AU, perturbed['y']/AU, label='With intruder', lw=1.5, alpha=0.7)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x (AU)'); plt.ylabel('y (AU)'); plt.title('Mercury: Baseline vs Perturbed (XY)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('mercury_xy_comparison.png', dpi=150); plt.close()

    # Deviation
    dr = ((perturbed['x']-baseline['x'])**2 + (perturbed['y']-baseline['y'])**2) ** 0.5
    plt.figure(figsize=(10,5))
    plt.plot(baseline['t_days'], dr/1000)
    plt.xlabel('Time (days)'); plt.ylabel('Position deviation (km)')
    plt.title('Mercury Position Deviation Due to Intruder')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('mercury_deviation.png', dpi=150); plt.close()

    # Eccentricity
    plt.figure(figsize=(10,5))
    plt.plot(baseline['t_days'], baseline['e'], label='Baseline', lw=1.5)
    plt.plot(perturbed['t_days'], perturbed['e'], label='With intruder', lw=1.5)
    plt.xlabel('Time (days)'); plt.ylabel('Eccentricity'); plt.title('Mercury Eccentricity Evolution')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('mercury_eccentricity.png', dpi=150); plt.close()

    # Conservation
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.semilogy(baseline['t_days'], abs(baseline['energy_error'])+1e-16, label='Baseline')
    plt.semilogy(perturbed['t_days'], abs(perturbed['energy_error'])+1e-16, label='With intruder')
    plt.xlabel('Time (days)'); plt.ylabel('|ΔE/E|'); plt.title('Energy Conservation'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.semilogy(baseline['t_days'], abs(baseline['momentum_error'])+1e-16, label='Baseline')
    plt.semilogy(perturbed['t_days'], abs(perturbed['momentum_error'])+1e-16, label='With intruder')
    plt.xlabel('Time (days)'); plt.ylabel('|ΔP/P|'); plt.title('Momentum Conservation'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('conservation_metrics.png', dpi=150); plt.close()

