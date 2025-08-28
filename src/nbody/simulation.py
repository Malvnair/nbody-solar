import copy
import numpy as np
import logging
from .config import AppConfig
from .integrators import velocity_verlet_step, adaptive_timestep
from .physics import is_collision, volume_add_radius
from .diagnostics import total_energy, total_momentum, total_angular_momentum, check_conservation
from .initial_conditions import get_initial_state

log = logging.getLogger(__name__)

class NBodySimulation:
    def __init__(self, cfg: AppConfig, intruder_on: bool = True):
        self.cfg = cfg
        self.state = get_initial_state()
        self.n_bodies = len(self.state)
        self.time = 0.0
        self.step_count = 0
        self.dt_current = cfg.integration.dt_initial

        # Intruder
        self.intruder_idx = 9
        self.intruder_base_mass = self.state[self.intruder_idx][6]
        self.intruder_active_flag = True
        self.intruder_approaching = intruder_on

        # Trails
        self.position_history = [[] for _ in range(self.n_bodies)]
        self.max_history = cfg.simulation.max_trail_length

        # Set intruder state
        if intruder_on:
            self.set_intruder_trajectory(True)
        else:
            self.set_intruder_trajectory(False)

        self.state[self.intruder_idx][6] = self.intruder_base_mass * self.intruder_multiplier()

        # Baselines
        self.E0 = total_energy(self.state, cfg.integration.epsilon_soft)
        self.P0 = total_momentum(self.state)
        self.L0 = total_angular_momentum(self.state)

        # Logs
        self.conservation_log = []
        self.timestep_log = []

    # ---- intruder management ----
    def set_intruder_trajectory(self, approaching: bool):
        idx = self.intruder_idx
        if approaching:
            vals = self.cfg.intruder.approach_state
            self.state[idx][0:6] = vals[:6]
            self.intruder_approaching = True
            self.position_history[idx].clear()
            self.position_history[idx].append(self.state[idx][:3].copy())
        else:
            vals = self.cfg.intruder.parked_state
            self.state[idx][0:6] = vals[:6]
            self.intruder_approaching = False
            self.position_history[idx].clear()
        self.state[idx][6] = self.intruder_base_mass * self.intruder_multiplier()
    # ---- collisions & merging ----
    def _merge_pair(self, keep_idx: int, kill_idx: int):
        bi = self.state[keep_idx]
        bj = self.state[kill_idx]
        mi, mj = bi[6], bj[6]
        M = mi + mj
        if M <= 0: return
        ri = np.array(bi[:3]); rj = np.array(bj[:3])
        vi = np.array(bi[3:6]); vj = np.array(bj[3:6])
        r_new = (mi*ri + mj*rj) / M
        v_new = (mi*vi + mj*vj) / M
        bi[0:3] = r_new.tolist()
        bi[3:6] = v_new.tolist()
        bi[6] = M
        bi[7] = volume_add_radius(bi[7], bj[7])
        bj[6] = 0.0; bj[7] = 1.0
        bj[0:3] = [1e15, 1e15, 1e15]
        bj[3:6] = [0.0, 0.0, 0.0]
        self.position_history[kill_idx].clear()
        if kill_idx == self.intruder_idx:
            self.intruder_approaching = False

    def _resolve_collisions(self) -> bool:
        collided = False
        for i in range(self.n_bodies):
            if self.state[i][6] <= 0: continue
            for j in range(i+1, self.n_bodies):
                if self.state[j][6] <= 0: continue
                if is_collision(self.state[i], self.state[j]):
                    keep = i if self.state[i][6] >= self.state[j][6] else j
                    kill = j if keep == i else i
                    self._merge_pair(keep, kill)
                    collided = True
        return collided
    

    def _shrink_dt_if_close_approach(self):
        factor = self.cfg.integration.close_approach_factor
        min_sep = np.inf; min_sumR = np.inf
        for i in range(self.n_bodies):
            if self.state[i][6] <= 0: continue
            for j in range(i+1, self.n_bodies):
                if self.state[j][6] <= 0: continue
                rij = np.linalg.norm(np.array(self.state[i][:3]) - np.array(self.state[j][:3]))
                sumR = self.state[i][7] + self.state[j][7]
                if rij < min_sep:
                    min_sep, min_sumR = rij, sumR
        if min_sep < factor * min_sumR:
            self.dt_current = max(self.cfg.integration.dt_min, min(self.dt_current, 0.1 * self.cfg.integration.dt_max))

    # ---- integration step ----
    def integrate_step(self):
        I = self.cfg.integration
        mult = self.intruder_multiplier()
        self.state[self.intruder_idx][6] = self.intruder_base_mass * mult
        self.dt_current = adaptive_timestep(
            self.state, I.eta_timestep, I.dt_min, I.dt_max, I.epsilon_soft,
            intruder_idx=self.intruder_idx, ramp_func=self.intruder_multiplier, time=self.time
        )
        if not np.isfinite(self.dt_current):
            self.dt_current = I.dt_initial

        self._shrink_dt_if_close_approach()

        prev_state = copy.deepcopy(self.state)
        try:
            self.state = velocity_verlet_step(
                self.state, self.dt_current, I.epsilon_soft,
                intruder_idx=self.intruder_idx, ramp_func=self.intruder_multiplier, time=self.time
            )
        except Exception:
            self.dt_current = max(I.dt_min, 0.5 * self.dt_current)
            self.state = velocity_verlet_step(
                prev_state, self.dt_current, I.epsilon_soft,
                intruder_idx=self.intruder_idx, ramp_func=self.intruder_multiplier, time=self.time
            )

        self.time += self.dt_current
        self.step_count += 1

        if self._resolve_collisions():
            log.info("[Step %d] Collision/merger occurred.", self.step_count)

        # NaN guard
        for i, body in enumerate(self.state):
            if not all(np.isfinite(body[:6])):
                self.state[i][:3] = [1e12*(i+1), 1e12*(i+1), 1e11*(i+1)]
                self.state[i][3:6] = [0.0, 0.0, 0.0]

    def update_history(self):
        for i in range(self.n_bodies):
            if i == self.intruder_idx and not self.intruder_approaching:
                continue
            self.position_history[i].append(self.state[i][:3].copy())
            if len(self.position_history[i]) > self.max_history:
                self.position_history[i].pop(0)

    def run_frame(self, steps_per_frame: int):
        for _ in range(steps_per_frame):
            self.integrate_step()
            if self.step_count % self.cfg.diagnostics.log_every_steps == 0:
                diag = self.check_and_log_conservation()
                if not diag["conservation_ok"]:
                    log.warning("[Step %d] dE/E=%.2e dP/P=%.2e dL/L=%.2e",
                                self.step_count, diag["energy_error"], diag["momentum_error"], diag["angular_momentum_error"])
        self.update_history()
        self.timestep_log.append(self.dt_current)

    def check_and_log_conservation(self):
        diag = check_conservation(
            self.state, self.E0, self.P0, self.L0,
            self.cfg.diagnostics.conservation_tol,
            self.cfg.integration.epsilon_soft,
        )
        self.conservation_log.append(diag)
        return diag
    
        # simulation.py
    def pair_weight(self, i, j):
        # 9 is your intruder index in your script
        if i == 9 or j == 9:
            return self.intruder_lambda   # 0→1 during ramp
        return 1.0


    # ---- intruder force ramp ----
    def intruder_multiplier(self, time=None, state=None):
        t = self.time if time is None else time
        st = self.state if state is None else state
        r_cfg = self.cfg.intruder.ramp
        if r_cfg.mode == "time":
            return 1.0 / (1.0 + np.exp(-(t - r_cfg.t0) / r_cfg.tau))
        elif r_cfg.mode == "distance":
            r = np.linalg.norm(np.array(st[self.intruder_idx][:3]))
            return 1.0 / (1.0 + np.exp((r - r_cfg.r0) / r_cfg.dr))
        else:
            return 1.0
        
    

