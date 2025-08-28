from nbody.config import load_config
from pathlib import Path
from nbody.simulation import NBodySimulation

def test_smoke():
    cfg = load_config(Path("config/config.yaml"), None)
    sim = NBodySimulation(cfg)
    sim.set_intruder_trajectory(False)
    sim.run_frame(cfg.simulation.steps_per_frame)
    assert sim.step_count > 0
