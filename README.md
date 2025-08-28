# N-Body Solar System (Config-Driven)

Velocity Verlet + adaptive timestep, intruder flyby, collision/merger, diagnostics, and a live 3D animation — now organized cleanly with YAML configs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m nbody --config config/config.yaml
