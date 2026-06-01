"""Plotting utilities for trajectory inspection.

The functions in this module do not block by default when called from main.py.
Use show=True if you explicitly want interactive Matplotlib windows.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from . import cts, dynamics


def _prepare_output_path(path):
    """Create parent directories and return a Path object, or None."""
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finish_figure(fig, save_path=None, show=False, block=False, close=True):
    """Save/show/close a Matplotlib figure without forcing a blocking pause."""
    save_path = _prepare_output_path(save_path)

    if save_path is not None:
        fig.savefig(save_path, dpi=80)

    if show:
        plt.show(block=block)

    if close and not show:
        plt.close(fig)


def plot2D(t, Y, title="Heliocentric trajectory animation", show=True, block=True):
    """Animate the heliocentric trajectory in the x-y plane.

    This function is intended for manual inspection. It opens a Matplotlib window
    when show=True. For report figures, use plot_trajectory_2d_static instead.
    """
    x_sc = Y[0, :]
    y_sc = Y[1, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def animate(i):
        ax.clear()
        ax.grid(True)

        ax.plot(x_sc[: i + 1], y_sc[: i + 1], "-", markersize=1, label="Spacecraft")
        ax.plot(x_sc[i], y_sc[i], "o", markersize=5)

        earth_x, earth_y = dynamics.earth_position(t[i])
        ax.plot(earth_x, earth_y, "o", markersize=6, label="Earth")

        target = plt.Circle((0.0, 0.0), cts.R_orb_B, fill=False, linestyle="--")
        ax.add_patch(target)

        ax.plot(0.0, 0.0, "o", markersize=8, label="Sun")
        # Keep automatic aspect for robust non-interactive rendering.
        ax.set_title(title)
        ax.text(0.02, 0.95, f"time = {t[i] / cts.YEAR_TO_S:.3f} years", transform=ax.transAxes)
        ax.legend(loc="best")

        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50)

    if show:
        plt.show(block=block)

    return ani


def plot_trajectory_2d_static(t, Y, save_path=None, title="Heliocentric trajectory", show=False, block=False, close=True, end_label="End"):
    """Create a robust static heliocentric 2D trajectory figure in Gm."""
    t = np.asarray(t)
    Y = np.asarray(Y)
    scale = 1e6

    # Downsample for robust rendering.  This does not affect the numerical data,
    # only the report figure.
    n = Y.shape[1]
    if n > 1200:
        idx = np.unique(np.linspace(0, n - 1, 1200).astype(int))
    else:
        idx = np.arange(n)

    x_sc = Y[0, idx] / scale
    y_sc = Y[1, idx] / scale
    t_plot = t[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_sc, y_sc, linewidth=1.2, label="Spacecraft trajectory")
    ax.plot(Y[0, 0] / scale, Y[1, 0] / scale, "o", markersize=5, label="Start")
    ax.plot(Y[0, -1] / scale, Y[1, -1] / scale, "o", markersize=5, label=end_label)

    earth_xy = np.array([dynamics.earth_position(ti) for ti in t_plot]) / scale
    ax.plot(earth_xy[:, 0], earth_xy[:, 1], "--", linewidth=1.0, label="Earth circular orbit")

    # Draw target radius with a normal line rather than a patch; this is faster
    # and avoids backend-specific slowdowns during savefig.
    ang = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot((cts.R_orb_B / scale) * np.cos(ang), (cts.R_orb_B / scale) * np.sin(ang), ":", label="Target radius R_B")

    ax.plot(0.0, 0.0, "o", markersize=8, label="Sun")
    ax.set_xlabel("x [Gm]")
    ax.set_ylabel("y [Gm]")
    ax.set_title(title)
    # Keep automatic aspect for robust non-interactive rendering.
    ax.grid(True)
    ax.legend(loc="best")

    lim = 1.05 * max(cts.R_orb_B / scale, np.nanmax(np.abs(x_sc)), np.nanmax(np.abs(y_sc)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.92)

    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_state_variables(t, Y, save_path=None, title="Dynamical variables", show=False, block=False, close=True):
    """Plot only the state variables, without a reference-error curve."""
    t_yr = t / cts.YEAR_TO_S
    var_labels = ["x", "y", "v_x", "v_y"]
    var_units = ["km", "km", "km/s", "km/s"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :], label=var_labels[i])
        axes[i].set_xlabel("Time [years]")
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(var_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle(title, fontsize=14)
    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_radial_and_earth_distance(t, Y, save_path=None, title="Radial and Earth-relative distances", show=False, block=False, close=True):
    """Plot heliocentric radius and Earth-relative distance versus time.

    Uses two small independent figures internally combined in a single canvas,
    avoiding slow log-axis rendering in some backends.
    """
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    t_yr = t / cts.YEAR_TO_S
    r_sun = np.hypot(Y[0, :], Y[1, :]) / 1e6

    earth_xy = np.array([dynamics.earth_position(ti) for ti in t])
    r_earth = np.hypot(Y[0, :] - earth_xy[:, 0], Y[1, :] - earth_xy[:, 1])
    log_rearth = np.log10(np.maximum(r_earth, 1.0))

    fig = plt.figure(figsize=(10, 7))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    ax0.plot(t_yr, r_sun, linewidth=1.0, label="|r| spacecraft")
    ax0.axhline(cts.R_orb_B / 1e6, linestyle="--", label="R_B target")
    ax0.set_xlabel("Time [years]")
    ax0.set_ylabel("Heliocentric radius [Gm]")
    ax0.set_title("Distance from Sun")
    ax0.grid(True)
    ax0.legend(loc="best")

    ax1.plot(t_yr, log_rearth, linewidth=1.0, label="log10(|r - R_Earth|)")
    ax1.axhline(np.log10(cts.earth_SOI_radius), linestyle="--", label="Earth SOI")
    ax1.axhline(np.log10(cts.R_Earth), linestyle=":", label="Earth radius")
    idx = int(np.argmin(r_earth))
    ax1.plot(t_yr[idx], log_rearth[idx], "o", markersize=4, label="Closest sampled approach")
    ax1.set_xlabel("Time [years]")
    ax1.set_ylabel("log10(distance from Earth [km])")
    ax1.set_title("Distance from Earth")
    ax1.grid(True)
    ax1.legend(loc="best")

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(hspace=0.42, top=0.90)
    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_geocentric_trajectory_static(t, Y, save_path=None, title="Earth-relative trajectory", show=False, block=False, close=True):
    """Plot the trajectory in Earth-centered coordinates x-X_Earth, y-Y_Earth."""
    x_rel = np.empty_like(t, dtype=float)
    y_rel = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        ex, ey = dynamics.earth_position(ti)
        x_rel[i] = Y[0, i] - ex
        y_rel[i] = Y[1, i] - ey

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_rel, y_rel, label="Spacecraft relative to Earth")
    ax.plot(x_rel[0], y_rel[0], "o", markersize=5, label="Start")
    ax.plot(x_rel[-1], y_rel[-1], "o", markersize=5, label="End")

    earth = plt.Circle((0.0, 0.0), cts.R_Earth, fill=False, linestyle=":", label="Earth radius")
    soi = plt.Circle((0.0, 0.0), cts.earth_SOI_radius, fill=False, linestyle="--", label="Earth SOI")
    ax.add_patch(earth)
    ax.add_patch(soi)

    ax.set_xlabel("x - X_Earth [km]")
    ax.set_ylabel("y - Y_Earth [km]")
    ax.set_title(title)
    # Keep automatic aspect for robust non-interactive rendering.
    ax.grid(True)
    ax.legend(loc="best")

    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_earth_encounter_zoom(t, Y, save_path=None, title="Earth encounter zoom", show=False, block=False, close=True, center_time=None, window_days=80.0):
    """Zoom the Earth-relative trajectory around the closest approach.

    The full Earth-relative trajectory can be unreadable because most of the
    path is millions of kilometres away.  This plot crops around the closest
    approach or around a supplied center_time.
    """
    t = np.asarray(t)
    if len(t) < 2:
        return None

    x_rel = np.empty_like(t, dtype=float)
    y_rel = np.empty_like(t, dtype=float)
    r_rel = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        ex, ey = dynamics.earth_position(ti)
        x_rel[i] = Y[0, i] - ex
        y_rel[i] = Y[1, i] - ey
        r_rel[i] = np.hypot(x_rel[i], y_rel[i])

    if center_time is None:
        # Ignore the initial parking-orbit departure; otherwise the zoom would
        # focus on t=0 rather than on the resonant return/flyby.
        mask = t > 0.5 * cts.YEAR_TO_S
        if np.any(mask):
            idx_global = np.where(mask)[0][int(np.argmin(r_rel[mask]))]
        else:
            idx_global = int(np.argmin(r_rel))
        center_time = float(t[idx_global])

    half_window = 0.5 * window_days * cts.DAY_TO_S
    mask = (t >= center_time - half_window) & (t <= center_time + half_window)
    if np.count_nonzero(mask) < 2:
        mask = slice(None)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_rel[mask] / 1e6, y_rel[mask] / 1e6, label="Spacecraft relative to Earth")
    ax.plot(x_rel[mask][0] / 1e6, y_rel[mask][0] / 1e6, "o", markersize=5, label="Window start")
    ax.plot(x_rel[mask][-1] / 1e6, y_rel[mask][-1] / 1e6, "o", markersize=5, label="Window end")

    earth = plt.Circle((0.0, 0.0), cts.R_Earth / 1e6, fill=False, linestyle=":", label="Earth radius")
    soi = plt.Circle((0.0, 0.0), cts.earth_SOI_radius / 1e6, fill=False, linestyle="--", label="Earth SOI")
    ax.add_patch(earth)
    ax.add_patch(soi)

    ax.set_xlabel("x - X_Earth [Gm]")
    ax.set_ylabel("y - Y_Earth [Gm]")
    ax.set_title(title)
    # Keep automatic aspect for robust non-interactive rendering.
    ax.grid(True)
    ax.legend(loc="best")
    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_case_II_energy_diagnostics(t, Y, save_path=None, title="Case II solar-energy diagnostics", show=False, block=False, close=True):
    """Plot heliocentric speed, solar specific energy and osculating aphelion."""
    t_yr = t / cts.YEAR_TO_S
    r = np.hypot(Y[0, :], Y[1, :])
    v = np.hypot(Y[2, :], Y[3, :])
    eps = 0.5 * v * v - cts.mu_sun / r
    a = np.full_like(eps, np.inf, dtype=float)
    bound = eps < 0.0
    a[bound] = -cts.mu_sun / (2.0 * eps[bound])
    r_apo = 2.0 * a - r

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))
    axes[0].plot(t_yr, v)
    axes[0].set_ylabel("Speed [km/s]")
    axes[0].grid(True)

    axes[1].plot(t_yr, eps)
    axes[1].set_ylabel("Solar specific energy [km²/s²]")
    axes[1].grid(True)

    r_apo_plot = r_apo / 1e6
    # Avoid unreadable autoscaling or slow rendering if the osculating energy
    # gets very close to zero.  Values beyond this cap are only saying
    # "effectively very large" for our diagnostic purposes.
    cap = 2.5 * cts.R_orb_B / 1e6
    r_apo_plot = np.where(np.isfinite(r_apo_plot) & (r_apo_plot < cap), r_apo_plot, np.nan)

    axes[2].plot(t_yr, r_apo_plot, label="Osculating aphelion")
    axes[2].axhline(cts.R_orb_B / 1e6, linestyle="--", label="R_B target")
    axes[2].set_ylim(0.0, cap)
    axes[2].set_xlabel("Time [years]")
    axes[2].set_ylabel("Aphelion [Gm]")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(title, fontsize=14)
    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_solution(t, Y, Y_ref, save_dir=None, prefix="case_I", show=False, block=False, close=True):
    """Plot state variables and numerical errors against a reference solution."""
    t_yr = t / cts.YEAR_TO_S
    save_dir = Path(save_dir) if save_dir is not None else None

    var_labels = ["x", "y", "v_x", "v_y"]
    var_units = ["km", "km", "km/s", "km/s"]
    err_labels = [
        "x - x_ref",
        "y - y_ref",
        "v_x - v_x_ref",
        "v_y - v_y_ref",
    ]

    fig_vars, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :], label=var_labels[i])
        axes[i].set_xlabel("Time [years]")
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(var_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig_vars.suptitle("Dynamical variables", fontsize=14)

    var_path = save_dir / f"{prefix}_state_variables.png" if save_dir is not None else None
    _finish_figure(fig_vars, save_path=var_path, show=show, block=False, close=close)

    fig_err, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :] - Y_ref[i, :], label=err_labels[i])
        axes[i].set_xlabel("Time [years]")
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(err_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig_err.suptitle("Numerical error with respect to reference solution", fontsize=14)

    err_path = save_dir / f"{prefix}_numerical_errors.png" if save_dir is not None else None
    _finish_figure(fig_err, save_path=err_path, show=show, block=block, close=close)

    return fig_vars, fig_err


def plot_orbital_elements(t, Y, center="sun", save_path=None, title=None, show=False, block=False, close=True):
    """Plot planar osculating orbital elements requested in the guide.

    For the Sun-centered plot the length variable is the semimajor axis a.
    For the Earth-centered plot the length variable is the periapsis distance rp,
    which remains meaningful for the hyperbolic Earth flyby.
    """
    elems = dynamics.planar_orbital_elements_series(t, Y, center=center)
    t_yr = np.asarray(t) / cts.YEAR_TO_S

    if title is None:
        title = f"Osculating orbital elements relative to {center}"

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

    e = np.asarray(elems["e"], dtype=float)
    axes[0].plot(t_yr, e)
    axes[0].set_ylabel("e [-]")
    axes[0].set_title("Eccentricity")
    axes[0].grid(True)

    if center == "sun":
        length = np.asarray(elems["a"], dtype=float) / 1e6
        label = "a [Gm]"
        # Avoid exploding the autoscale near almost-parabolic states.
        cap = 3.0 * cts.R_orb_B / 1e6
        length = np.where(np.isfinite(length) & (np.abs(length) < cap), length, np.nan)
        axes[1].axhline(cts.R_orb_A / 1e6, linestyle=":", label="Earth orbit")
        axes[1].axhline(cts.R_orb_B / 1e6, linestyle="--", label="Saturn orbit")
        axes[1].legend(loc="best")
    else:
        length = np.asarray(elems["rp"], dtype=float)
        label = "r_p [km]"
        length = np.where(np.isfinite(length) & (length > 0.0), length, np.nan)
        axes[1].axhline(cts.R_Earth, linestyle=":", label="Earth radius")
        axes[1].axhline(cts.earth_SOI_radius, linestyle="--", label="Earth SOI")
        axes[1].set_yscale("log")
        axes[1].legend(loc="best")

    axes[1].plot(t_yr, length)
    axes[1].set_ylabel(label)
    axes[1].set_title("Length orbital element")
    axes[1].grid(True, which="major")

    omega_deg = np.rad2deg(elems["omega"])
    axes[2].plot(t_yr, omega_deg)
    axes[2].set_xlabel("Time [years]")
    axes[2].set_ylabel("omega [deg]")
    axes[2].set_title("Argument of periapsis")
    axes[2].grid(True)

    fig.suptitle(title, fontsize=14)
    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def plot_comparison_summary(case_I_best, case_II_best, save_path=None, show=False, block=False, close=True):
    """Bar plot comparing ignition, final and total delta-v for both optimized cases."""
    labels = ["dv_ign", "dv_fin", "dv_tot"]
    vals_I = [case_I_best["dv_ign"], case_I_best["dv_fin"], case_I_best["dv_tot"]]
    vals_II = [case_II_best["dv_ign"], case_II_best["dv_fin"], case_II_best["dv_tot"]]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, vals_I, width, label="Case I")
    ax.bar(x + width / 2, vals_II, width, label="Case II")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta-v [km/s]")
    ax.set_title("Optimized delta-v comparison")
    ax.grid(True, axis="y")
    ax.legend(loc="best")

    _finish_figure(fig, save_path=save_path, show=show, block=block, close=close)
    return fig


def save_heliocentric_animation(t, Y, save_path, title="Heliocentric trajectory animation", n_frames=180, fps=20):
    """Save a heliocentric GIF animation of the spacecraft and Earth.

    The animation is intentionally saved as GIF through Pillow because it is
    portable and does not require ffmpeg.
    """
    save_path = _prepare_output_path(save_path)
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(t) < 2:
        return None

    n_frames = int(max(2, min(n_frames, len(t))))
    frame_idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    scale = 1e6
    x = Y[0, :] / scale
    y = Y[1, :] / scale
    earth_xy = np.array([dynamics.earth_position(ti) for ti in t]) / scale

    fig, ax = plt.subplots(figsize=(7, 7))
    target = plt.Circle((0.0, 0.0), cts.R_orb_B / scale, fill=False, linestyle=":", label="R_B")
    earth_orbit = plt.Circle((0.0, 0.0), cts.R_orb_A / scale, fill=False, linestyle="--", label="Earth orbit")
    ax.add_patch(target)
    ax.add_patch(earth_orbit)
    ax.plot(0.0, 0.0, "o", markersize=7, label="Sun")

    path_line, = ax.plot([], [], linewidth=1.2, label="Spacecraft path")
    sc_dot, = ax.plot([], [], "o", markersize=5, label="Spacecraft")
    earth_dot, = ax.plot([], [], "o", markersize=5, label="Earth")
    time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes)

    lim = 1.08 * max(cts.R_orb_B / scale, np.nanmax(np.abs(x)), np.nanmax(np.abs(y)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # Keep automatic aspect for robust non-interactive rendering.
    ax.set_xlabel("x [Gm]")
    ax.set_ylabel("y [Gm]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right")

    def update(k):
        i = int(frame_idx[k])
        path_line.set_data(x[: i + 1], y[: i + 1])
        sc_dot.set_data([x[i]], [y[i]])
        earth_dot.set_data([earth_xy[i, 0]], [earth_xy[i, 1]])
        time_text.set_text(f"t = {t[i] / cts.YEAR_TO_S:.2f} yr")
        return path_line, sc_dot, earth_dot, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=True)

    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(str(save_path), writer=writer, dpi=120)
    finally:
        plt.close(fig)

    return save_path
