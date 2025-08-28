import argparse
from pathlib import Path
import logging
from .config import load_config
from .simulation import NBodySimulation
from .analysis import run_mercury_analysis, diagnostic_plots
from .viz import create_animation
from .constants import G

log = logging.getLogger("nbody")

def main():
    p = argparse.ArgumentParser(description="Solar System N-Body (VV + adaptive dt)")
    p.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    p.add_argument("--logging", type=Path, default=Path("config/logging.yaml"))
    p.add_argument("--no-animation", action="store_true", help="Skip live animation window")
    p.add_argument("--frames", type=int, help="Override analysis frames")
    args = p.parse_args()

    cfg = load_config(args.config, args.logging)

    # 1) Baseline analysis
    log.info("Running BASELINE scenario...")
    sim_baseline = NBodySimulation(cfg, intruder_on=False)
    n_frames = args.frames or cfg.simulation.max_frames
    baseline = run_mercury_analysis(sim_baseline, n_frames, G)

    # 2) Perturbed analysis
    log.info("Running PERTURBED scenario...")
    sim_perturbed = NBodySimulation(cfg, intruder_on=True)
    perturbed = run_mercury_analysis(sim_perturbed, n_frames, G)

    # 3) Plots
    log.info("Generating diagnostic plots...")
    diagnostic_plots(baseline, perturbed)

    # 4) Stats
    dr = ((perturbed['x']-baseline['x'])**2 + (perturbed['y']-baseline['y'])**2) ** 0.5
    pos_dev_km = dr.max() / 1000.0
    de_final = float(perturbed['e'][-1] - baseline['e'][-1])
    log.info("Mercury: Max XY position deviation = %.1f km | Δe(final) = %.6f", pos_dev_km, de_final)

    # 5) Live animation
    if not args.no_animation:
        log.info("Starting animation...")
        sim_anim = NBodySimulation(cfg)
        _, anim = create_animation(sim_anim)
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()
