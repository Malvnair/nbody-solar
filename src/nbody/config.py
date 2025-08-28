from dataclasses import dataclass
from typing import List, Tuple
import yaml
import logging
import logging.config
from pathlib import Path

log = logging.getLogger(__name__)

@dataclass
class SimulationCfg:
    steps_per_frame: int
    max_frames: int
    fps: int
    show_trails: bool
    max_trail_length: int
    num_background_stars: int
    intruder_trail_n: int

@dataclass
class IntegrationCfg:
    epsilon_soft: float
    eta_timestep: float
    dt_min: float
    dt_max: float
    dt_initial: float
    close_approach_factor: float

@dataclass
class DiagnosticsCfg:
    conservation_tol: float
    log_every_steps: int
    log_every_frames: int

@dataclass
class VizCfg:
    ortho_proj: bool
    view_elev_deg: float
    view_azim_deg: float
    initial_zoom_AU: float
    max_zoom_AU: float
    bg_color: Tuple[float, float, float]

@dataclass
class IntruderCfg:
    approach_state: List[float]   # [x,y,z,vx,vy,vz]
    parked_state: List[float]

@dataclass
class AppConfig:
    simulation: SimulationCfg
    integration: IntegrationCfg
    diagnostics: DiagnosticsCfg
    viz: VizCfg
    intruder: IntruderCfg

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def load_config(cfg_path: Path, logging_path: Path | None = None) -> AppConfig:
    if logging_path and logging_path.exists():
        logging.config.dictConfig(load_yaml(logging_path))
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    raw = load_yaml(cfg_path)
    try:
        raw["integration"]["epsilon_soft"] = float(raw["integration"]["epsilon_soft"])
    except (KeyError, ValueError, TypeError):
        pass
    try:
        raw["intruder"]["approach_state"] = [float(x) for x in raw["intruder"]["approach_state"]]
        raw["intruder"]["parked_state"] = [float(x) for x in raw["intruder"]["parked_state"]]
    except (KeyError, TypeError, ValueError):
        pass

    appcfg = AppConfig(
        simulation=SimulationCfg(**raw["simulation"]),
        integration=IntegrationCfg(**raw["integration"]),
        diagnostics=DiagnosticsCfg(**raw["diagnostics"]),
        viz=VizCfg(**raw["viz"]),
        intruder=IntruderCfg(**raw["intruder"]),
    )
    log.info("Loaded config from %s", cfg_path)
    return appcfg
