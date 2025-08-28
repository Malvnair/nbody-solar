from pathlib import Path
from nbody.config import load_config
from nbody.simulation import NBodySimulation


def test_intruder_ramp_active_from_start():
    cfg = load_config(Path("config/config.yaml"))
    sim = NBodySimulation(cfg)
    assert sim.intruder_approaching is True
    m0 = sim.intruder_multiplier()
    assert 0 <= m0 <= 1
    sim.run_frame(cfg.simulation.steps_per_frame)
    m1 = sim.intruder_multiplier()
    assert m1 >= m0
