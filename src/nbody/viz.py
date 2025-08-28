import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from .constants import AU
from .simulation import NBodySimulation

def generate_background_stars(n_stars: int):
    stars = []
    for _ in range(n_stars):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(60*AU, 80*AU)
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        stars.append([x, y, z])
    return stars

def create_animation(sim: NBodySimulation):
    cfg = sim.cfg
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if cfg.viz.ortho_proj:
        ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 0.5))

    bg = tuple(cfg.viz.bg_color)
    ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    ax.set_xlim3d(-AU, AU); ax.set_ylim3d(-AU, AU); ax.set_zlim3d(-AU/2, AU/2)
    ax.view_init(cfg.viz.view_elev_deg, cfg.viz.view_azim_deg)
    ax.set_xlabel('X (AU)', color='white'); ax.set_ylabel('Y (AU)', color='white'); ax.set_zlabel('Z (AU)', color='white')
    ax.grid(True, alpha=0.2)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.set_pane_color((0,0,0,0)); ax.yaxis.set_pane_color((0,0,0,0)); ax.zaxis.set_pane_color((0,0,0,0))
    ax.tick_params(colors='white', which='both')

    stars = generate_background_stars(cfg.simulation.num_background_stars)
    ax.scatter([s[0] for s in stars], [s[1] for s in stars], [s[2] for s in stars],
               c='white', s=0.5, alpha=0.6, marker='.')

    ax_inset = fig.add_axes([0.65, 0.65, 0.3, 0.3])
    ax_inset.set_facecolor(bg)
    ax_inset.set_xlim(-2.5*AU, 2.5*AU); ax_inset.set_ylim(-2.5*AU, 2.5*AU)
    ax_inset.set_aspect('equal'); ax_inset.set_title('Inner Solar System (XY)', color='white', fontsize=10)
    ax_inset.tick_params(colors='white', which='both', labelsize=8)
    ax_inset.grid(True, alpha=0.2, color='white')
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('white'); spine.set_linewidth(1)

    lines_3d, lines_2d = [], []
    scatters_3d, scatters_2d, text_labels = [], [], []
    start_time = datetime.now()

    def animate(frame):
        nonlocal scatters_3d, scatters_2d, text_labels

        sim.run_frame(sim.cfg.simulation.steps_per_frame)

        ax.view_init(30 + 10*np.sin(frame*0.01), -45 - frame*0.2)
        zoom = min(AU * cfg.viz.initial_zoom_AU * (1.008**frame), cfg.viz.max_zoom_AU*AU)
        ax.set_xlim3d(-zoom, zoom); ax.set_ylim3d(-zoom, zoom); ax.set_zlim3d(-zoom/2, zoom/2)

        elapsed = (datetime.now() - start_time).total_seconds()
        fps = frame / elapsed if elapsed > 0 else 0
        years = (sim.time / 86400) / 365.25
        title = (f"Solar System N-Body (VV + Adaptive dt) | "
                 f"t={years:.2f} y | zoom={zoom/AU:.1f} AU | fps={fps:.1f} | dt={sim.dt_current:.1f}s")
        ax.set_title(title, color='white', fontsize=14, pad=20)

        for e in scatters_3d + scatters_2d + text_labels:
            e.remove()
        scatters_3d.clear(); scatters_2d.clear(); text_labels.clear()

        while len(lines_3d) < sim.n_bodies:
            lines_3d.append(ax.plot([], [], [], lw=0.5, alpha=0.6)[0])
        while len(lines_2d) < 6:
            lines_2d.append(ax_inset.plot([], [], lw=0.8, alpha=0.7)[0])

        for i in range(sim.n_bodies):
            if len(sim.position_history[i]) > 1:
                trail = np.array(sim.position_history[i])
                if i == sim.intruder_idx:
                    trail = trail[-sim.cfg.simulation.intruder_trail_n:]
                lines_3d[i].set_data(trail[:, 0], trail[:, 1])
                lines_3d[i].set_3d_properties(trail[:, 2])
                lines_3d[i].set_color(sim.state[i][8])
                if i == sim.intruder_idx and sim.intruder_approaching:
                    lines_3d[i].set_linewidth(2); lines_3d[i].set_alpha(0.9)

        for i in range(5):
            if len(sim.position_history[i]) > 1:
                trail = np.array(sim.position_history[i][-50:])
                lines_2d[i].set_data(trail[:, 0], trail[:, 1])
                lines_2d[i].set_color(sim.state[i][8])

        if sim.intruder_approaching and len(sim.position_history[sim.intruder_idx]) > 1:
            trail = np.array(sim.position_history[sim.intruder_idx][-sim.cfg.simulation.intruder_trail_n:])
            lines_2d[5].set_data(trail[:, 0], trail[:, 1])
            lines_2d[5].set_color('white'); lines_2d[5].set_linewidth(2)
        else:
            lines_2d[5].set_data([], [])

        for i in range(sim.n_bodies):
            b = sim.state[i]
            if b[6] <= 0: continue
            if i == sim.intruder_idx and not sim.intruder_approaching:
                continue
            size = 100 if i == 0 else (20 if i <= 4 else 30)
            if i == sim.intruder_idx:
                s = ax.scatter(b[0], b[1], b[2], color='white', s=150, alpha=1.0,
                               edgecolors='yellow', linewidth=2, marker='*')
                scatters_3d.append(s)
                glow = ax.scatter(b[0], b[1], b[2], color='yellow', s=300, alpha=0.3, marker='o')
                scatters_3d.append(glow)
                t = ax.text(b[0], b[1], b[2], '  INTRUDER!', color='red', fontsize=10, fontweight='bold')
                text_labels.append(t)
            else:
                s = ax.scatter(b[0], b[1], b[2], color=b[8], s=size, alpha=0.9,
                               edgecolors='white', linewidth=0.5)
                scatters_3d.append(s)

        for i in range(5):
            b = sim.state[i]
            if b[6] <= 0: continue
            s = ax_inset.scatter(b[0], b[1], color=b[8], s=(50 if i == 0 else 15), alpha=0.9,
                                 edgecolors='white', linewidth=0.5)
            scatters_2d.append(s)

        if sim.intruder_approaching and sim.state[sim.intruder_idx][6] > 0:
            b = sim.state[sim.intruder_idx]
            s = ax_inset.scatter(b[0], b[1], color='white', s=80, alpha=1.0,
                                 edgecolors='red', linewidth=2, marker='*')
            scatters_2d.append(s)

        return lines_3d + lines_2d + scatters_3d + scatters_2d + text_labels

    anim = FuncAnimation(fig, animate, frames=sim.cfg.simulation.max_frames,
                         interval=1000/sim.cfg.simulation.fps, blit=False)
    return fig, anim
