import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from . import cts, orbital_elements


def finish_figure(fig, save_path=None, show=True, dpi=300):
    # Función auxiliar para no repetir savefig/show/close en todos los plots.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def saturn_initial_angle_for_arrival(t, Y):
    # Fase inicial visual de Saturno para que, moviéndose con su velocidad
    # angular real, llegue al ángulo de la nave en el último instante del plot.
    t_fin = float(t[-1])
    theta_arrival = np.arctan2(Y[1, -1], Y[0, -1])
    return theta_arrival - cts.frec_B * t_fin


def saturn_position_for_plot(t_value, theta_saturn_0):
    theta = theta_saturn_0 + cts.frec_B * t_value
    return np.array([
        cts.R_orb_B * np.cos(theta),
        cts.R_orb_B * np.sin(theta),
    ])


def plot2D(t, dt, Y, R_f, save_path=None, gif_path=None, show=True, n_frames=300): # Plot 2D animation

    x1 = Y[0, :]
    y1 = Y[1, :]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.grid()

    # Para que el gif no sea gigante si hay muchos puntos
    if len(t) > n_frames:
        frames = np.linspace(0, len(t) - 1, n_frames).astype(int)
    else:
        frames = np.arange(len(t))

    alpha = np.linspace(0.0, 2.0*np.pi, 600)
    earth_orbit_x = cts.R_orb_A*np.cos(alpha)
    earth_orbit_y = cts.R_orb_A*np.sin(alpha)
    saturn_orbit_x = cts.R_orb_B*np.cos(alpha)
    saturn_orbit_y = cts.R_orb_B*np.sin(alpha)

    theta_saturn_0 = saturn_initial_angle_for_arrival(t, Y)

    def animate(frame_index):
        i = int(frame_index)

        ax.clear()
        ax.grid()

        ax.plot(earth_orbit_x, earth_orbit_y, '--', linewidth=0.8, color='tab:blue', label='Órbita Tierra')
        ax.plot(saturn_orbit_x, saturn_orbit_y, '--', linewidth=0.8, color='tab:brown', label='Órbita Saturno')

        ax.plot(x1[:i+1], y1[:i+1], '-', markersize=1, color='red', label='Nave')
        ax.plot(x1[i], y1[i], 'o', markersize=5, color='red')

        ax.set_aspect('equal', adjustable='box')

        ax.set_title('Trayectoria heliocéntrica')
        ax.text(0.02, 0.95, f'time = {t[i]/cts.year2seconds:.2f} years', transform=ax.transAxes)

        # plottear tierra, saturno y sol
        [RtA_x, RtA_y] = R_f(t[i], cts.R_orb_A, cts.frec_A)
        ax.plot(RtA_x, RtA_y, 'o', markersize=6, color='blue', label='Tierra')

        [RtB_x, RtB_y] = saturn_position_for_plot(t[i], theta_saturn_0)
        ax.plot(RtB_x, RtB_y, 'o', markersize=6, color='brown', label='Saturno')

        ax.plot(0, 0, 'o', markersize=8, color='orange', label='Sol')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.legend(loc='upper right')

        return []

    # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=50)

    if gif_path is not None:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(gif_path, writer='pillow', fps=20)
        except Exception as exc:
            print('Could not save animation:', exc)

    if save_path is not None:
        # Guardamos también una imagen estática final.
        animate(frames[-1])
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_solution(t, Y, Y_ref, save_prefix=None, show=True): # Plot variables and errors

    t_yr = t / (365.25 * 24 * 3600)

    var_labels = ['$x$', '$y$', '$v_x$', '$v_y$']
    var_units  = ['km', 'km', 'km/s', 'km/s']
    err_labels = ['$x - x_{ref}$', '$y - y_{ref}$', '$v_x - v_{x,ref}$', '$v_y - v_{y,ref}$']

    # Dynamic variables
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :], color='black', label=var_labels[i])
        axes[i].set_xlabel('Time [years]')
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(var_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle('Dynamical variables', fontsize=14)

    if save_prefix is None:
        finish_figure(fig, show=show)
    else:
        finish_figure(fig, str(save_prefix) + '_state_variables.png', show=False)

    # Numerical errors
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :] - Y_ref[i, :], color='black', label=err_labels[i])
        axes[i].set_xlabel('Time [years]')
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(err_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle('Numerical error with respect to reference solution', fontsize=14)

    if save_prefix is None:
        finish_figure(fig, show=show)
    else:
        finish_figure(fig, str(save_prefix) + '_numerical_errors.png', show=False)


#Plot the complete optimized Case II trajectory
def plot2D_caseII(t, Y, R_f, t_SOI_in=None, t_SOI_out=None, t_MED=None, save_path=None, show=True):

    x = Y[0, :]
    y = Y[1, :]

    GM = 1.0e6

    alpha = np.linspace(0.0, 2.0 * np.pi, 800)

    earth_orbit_x = cts.R_orb_A * np.cos(alpha)
    earth_orbit_y = cts.R_orb_A * np.sin(alpha)

    saturn_orbit_x = cts.R_orb_B * np.cos(alpha)
    saturn_orbit_y = cts.R_orb_B * np.sin(alpha)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    ax.plot(earth_orbit_x/GM, earth_orbit_y/GM, "--", linewidth=1.0, label="Earth orbit")
    ax.plot(saturn_orbit_x/GM, saturn_orbit_y/GM, "--", linewidth=1.0, label="Saturn orbit")
    ax.plot(x/GM, y/GM, "-", linewidth=1.2, label="Spacecraft trajectory")

    ax.plot(0.0, 0.0, "o", markersize=7, label="Sun")
    ax.plot(x[0]/GM, y[0]/GM, "o", markersize=5, label="Departure")
    ax.plot(x[-1]/GM, y[-1]/GM, "o", markersize=5, label="Arrival at R_B")

    theta_saturn_0 = saturn_initial_angle_for_arrival(t, Y)
    saturn_final = saturn_position_for_plot(t[-1], theta_saturn_0)
    ax.plot(saturn_final[0]/GM, saturn_final[1]/GM, "o", markersize=6, label="Saturn at arrival")

    if t_SOI_in is not None:
        i_in = int(np.argmin(np.abs(t - t_SOI_in)))
        ax.plot(x[i_in]/GM, y[i_in]/GM, "s", markersize=5, label="SOI entry")

    if t_SOI_out is not None:
        i_out = int(np.argmin(np.abs(t - t_SOI_out)))
        ax.plot(x[i_out]/GM, y[i_out]/GM, "s", markersize=5, label="SOI exit")

    if t_MED is not None:
        i_med = int(np.argmin(np.abs(t - t_MED)))
        ax.plot(x[i_med]/GM, y[i_med]/GM, "x", markersize=7, label="Closest Earth approach")

    ax.set_title("Case II complete heliocentric trajectory")
    ax.set_xlabel("x [Gm]")
    ax.set_ylabel("y [Gm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)


def plot_heliocentric_trajectory(t, Y, R_f, title='', save_path=None, show=True,
                                t_SOI_in=None, t_SOI_out=None, t_MED=None):

    x = Y[0, :]
    y = Y[1, :]

    GM = 1.0e6

    alpha = np.linspace(0.0, 2.0*np.pi, 800)

    earth_orbit_x = cts.R_orb_A*np.cos(alpha)
    earth_orbit_y = cts.R_orb_A*np.sin(alpha)
    saturn_orbit_x = cts.R_orb_B*np.cos(alpha)
    saturn_orbit_y = cts.R_orb_B*np.sin(alpha)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    ax.plot(earth_orbit_x/GM, earth_orbit_y/GM, '--', linewidth=1.0, label='Earth orbit')
    ax.plot(saturn_orbit_x/GM, saturn_orbit_y/GM, '--', linewidth=1.0, label='Saturn orbit')
    ax.plot(x/GM, y/GM, '-', linewidth=1.2, label='Spacecraft')

    ax.plot(0.0, 0.0, 'o', markersize=7, label='Sun')
    ax.plot(x[0]/GM, y[0]/GM, 'o', markersize=5, label='Departure')
    ax.plot(x[-1]/GM, y[-1]/GM, 'o', markersize=5, label='Arrival')

    theta_saturn_0 = saturn_initial_angle_for_arrival(t, Y)
    saturn_final = saturn_position_for_plot(t[-1], theta_saturn_0)
    ax.plot(saturn_final[0]/GM, saturn_final[1]/GM, 'o', markersize=6, label='Saturn at arrival')

    if t_SOI_in is not None:
        i_in = int(np.argmin(np.abs(t - t_SOI_in)))
        ax.plot(x[i_in]/GM, y[i_in]/GM, 's', markersize=5, label='SOI entry')

    if t_SOI_out is not None:
        i_out = int(np.argmin(np.abs(t - t_SOI_out)))
        ax.plot(x[i_out]/GM, y[i_out]/GM, 's', markersize=5, label='SOI exit')

    if t_MED is not None:
        i_med = int(np.argmin(np.abs(t - t_MED)))
        ax.plot(x[i_med]/GM, y[i_med]/GM, 'x', markersize=7, label='Closest Earth approach')

    if title == '':
        title = 'Heliocentric trajectory'

    ax.set_title(title)
    ax.set_xlabel('x [Gm]')
    ax.set_ylabel('y [Gm]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)


def plot_geocentric_trajectory(t, Y, R_f, title='', save_path=None, show=True,
                               t_SOI_in=None, t_SOI_out=None, t_MED=None):

    earth_pos = np.array([R_f(ti, cts.R_orb_A, cts.frec_A) for ti in t])

    x_rel = Y[0, :] - earth_pos[:, 0]
    y_rel = Y[1, :] - earth_pos[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    ax.plot(x_rel, y_rel, '-', linewidth=1.2, label='Spacecraft wrt Earth')
    ax.plot(0.0, 0.0, 'o', markersize=6, label='Earth')

    alpha = np.linspace(0.0, 2.0*np.pi, 500)
    ax.plot(cts.R_Earth*np.cos(alpha), cts.R_Earth*np.sin(alpha), '--', linewidth=1.0, label='Earth radius')
    ax.plot(cts.earth_SOI_radius*np.cos(alpha), cts.earth_SOI_radius*np.sin(alpha), ':', linewidth=1.0, label='Earth SOI')

    if t_SOI_in is not None:
        i_in = int(np.argmin(np.abs(t - t_SOI_in)))
        ax.plot(x_rel[i_in], y_rel[i_in], 's', markersize=5, label='SOI entry')

    if t_SOI_out is not None:
        i_out = int(np.argmin(np.abs(t - t_SOI_out)))
        ax.plot(x_rel[i_out], y_rel[i_out], 's', markersize=5, label='SOI exit')

    if t_MED is not None:
        i_med = int(np.argmin(np.abs(t - t_MED)))
        ax.plot(x_rel[i_med], y_rel[i_med], 'x', markersize=7, label='Closest approach')

    if title == '':
        title = 'Earth-centered trajectory'

    ax.set_title(title)
    ax.set_xlabel('x - X_Earth [km]')
    ax.set_ylabel('y - Y_Earth [km]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)


def plot_distances(t, Y, R_f, title="", t_SOI_in=None, t_SOI_out=None, t_MED=None, save_path=None, show=True):

    x = Y[0, :]
    y = Y[1, :]

    t_years = t / cts.year2seconds

    # Distance to Sun
    r_sun = np.hypot(x, y)

    # Earth / planet A position with the same phase convention as main.py
    earth_pos = np.array([R_f(ti, cts.R_orb_A, cts.frec_A) for ti in t])
    earth_x = earth_pos[:, 0]
    earth_y = earth_pos[:, 1]

    # Distance to Earth / planet A
    d_earth = np.hypot(x - earth_x, y - earth_y)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        constrained_layout=True,
    )

    # Distance to Sun
    ax = axes[0]

    ax.plot(t_years, r_sun, linewidth=1.2, label="r(t)")
    ax.axhline(cts.R_orb_A, linestyle="--", linewidth=1.0, label="Earth orbit radius")
    ax.axhline(cts.R_orb_B, linestyle="--", linewidth=1.0, label="Saturn orbit radius")

    if t_SOI_in is not None:
        ax.axvline(t_SOI_in / cts.year2seconds, linestyle=":", linewidth=1.0, label="SOI entry")

    if t_SOI_out is not None:
        ax.axvline(t_SOI_out / cts.year2seconds, linestyle=":", linewidth=1.0, label="SOI exit")

    if t_MED is not None:
        ax.axvline(t_MED / cts.year2seconds, linestyle="-.", linewidth=1.0, label="Closest Earth approach")

    ax.set_title(title + " - distance to Sun")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("r(t) [km]")
    ax.grid(True)
    ax.legend()

    # Distance to Earth / planet A
    ax = axes[1]

    ax.plot(t_years, d_earth, linewidth=1.2, label="|r(t) - R_Earth(t)|")
    ax.axhline(cts.R_Earth, linestyle="--", linewidth=1.0, label="Earth radius")
    ax.axhline(cts.earth_SOI_radius, linestyle="--", linewidth=1.0, label="Earth SOI radius")

    if t_SOI_in is not None:
        ax.axvline(t_SOI_in / cts.year2seconds, linestyle=":", linewidth=1.0, label="SOI entry")

    if t_SOI_out is not None:
        ax.axvline(t_SOI_out / cts.year2seconds, linestyle=":", linewidth=1.0, label="SOI exit")

    if t_MED is not None:
        ax.axvline(t_MED / cts.year2seconds, linestyle="-.", linewidth=1.0, label="Closest Earth approach")

    ax.set_yscale("log")
    ax.set_title(title + " - distance to Earth")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("|r(t) - R_Earth(t)| [km]")
    ax.grid(True, which="both")
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)


def plot_orbital_elements(t, Y, R_f, center="sun", title="", t_SOI_in=None, t_SOI_out=None, t_MED=None, time_window = None, save_path=None, show=True):
    # time_window:
    # Optional tuple (t_min, t_max) in seconds.
    # to zoom Earth-relative elements near departure or flyby.

    t = np.asarray(t)
    Y = np.asarray(Y)

    if time_window is not None:
        t_min, t_max = time_window
        mask = (t >= t_min) & (t <= t_max)

        if np.count_nonzero(mask) < 2:
            print("Not enough points inside selected time_window for orbital elements plot.")
            return

        t_plot = t[mask]
        Y_plot = Y[:, mask]
    else:
        t_plot = t
        Y_plot = Y

    elements = orbital_elements.compute_planar_orbital_elements(
        t=t_plot,
        Y=Y_plot,
        R_f=R_f,
        center=center,
    )

    t_years = t_plot / cts.year2seconds

    e = elements["e"]
    p = elements["p"]
    omega_deg = np.degrees(elements["omega"])

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 9),
        constrained_layout=True,
    )

    if title == "":
        title = "Orbital elements wrt " + center

    def add_event_lines(ax):
        if t_SOI_in is not None and t_plot[0] <= t_SOI_in <= t_plot[-1]:
            ax.axvline(
                t_SOI_in / cts.year2seconds,
                linestyle=":",
                linewidth=1.0,
                label="SOI entry",
            )

        if t_SOI_out is not None and t_plot[0] <= t_SOI_out <= t_plot[-1]:
            ax.axvline(
                t_SOI_out / cts.year2seconds,
                linestyle=":",
                linewidth=1.0,
                label="SOI exit",
            )

        if t_MED is not None and t_plot[0] <= t_MED <= t_plot[-1]:
            ax.axvline(
                t_MED / cts.year2seconds,
                linestyle="-.",
                linewidth=1.0,
                label="Closest Earth approach",
            )

    # ============================================================
    # Eccentricity
    # ============================================================
    ax = axes[0]
    ax.plot(t_years, e, linewidth=1.2, label="e(t)")
    add_event_lines(ax)

    ax.set_title(title + " - eccentricity")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("e [-]")
    ax.grid(True)
    ax.legend()

    # ============================================================
    # Semi-latus rectum
    # ============================================================
    ax = axes[1]
    ax.plot(t_years, p, linewidth=1.2, label="p(t)")
    add_event_lines(ax)

    ax.set_title(title + " - semi-latus rectum")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("p [km]")
    ax.grid(True)
    ax.legend()

    # ============================================================
    # Argument of periapsis
    # ============================================================
    ax = axes[2]
    ax.plot(t_years, omega_deg, linewidth=1.2, label=r"$\omega(t)$")
    add_event_lines(ax)

    ax.set_title(title + " - argument of periapsis")
    ax.set_xlabel("time [years]")
    ax.set_ylabel(r"$\omega$ [deg]")
    ax.grid(True)
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)


def plot_delta_v_comparison(caseI, caseII, save_path=None, show=True):

    labels = ['Initial', 'Final', 'Total']

    caseI_values = [
        caseI['dv_ign'],
        caseI['dv_fin'],
        caseI['dv_tot']
    ]

    caseII_values = [
        caseII['dv_ign'],
        caseII['dv_fin'],
        caseII['dv_tot']
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.bar(x - width/2, caseI_values, width, label='Case I')
    ax.bar(x + width/2, caseII_values, width, label='Case II')

    ax.set_ylabel('Delta-v [km/s]')
    ax.set_title('Delta-v comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis='y')
    ax.legend()

    finish_figure(fig, save_path=save_path, show=show)
