import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import cts

def plot2D(t, dt, Y, R_f): # Plot 2D animation

    x1 = Y[0, :]
    y1 = Y[1, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()

    def animate(i):
        ax.clear()
        ax.grid()

        ax.plot(x1[:i], y1[:i], '-', markersize=1, color='red', label="nave")
        ax.plot(x1[i], y1[i], 'o', markersize=5, color='red')

        ax.set_aspect('equal', adjustable='box')

        ax.set_title('Simulación Órbita')
        ax.text(0.02, 0.95, f'time = {t[i]:.1f} s', transform=ax.transAxes)

        # plottear tierra, saturno y sol 
        [RtA_x, RtA_y] = R_f(i*dt, cts.R_orb_A, cts.frec_A)
        ax.plot(RtA_x, RtA_y, 'o', markersize=6, color='blue', label='Tierra')

        [RtB_x, RtB_y] = R_f(i*dt, -cts.R_orb_B, cts.frec_B)
        ax.plot(RtB_x, RtB_y, 'o', markersize=6, color='brown', label='Saturno')


        ax.plot(0, 0, 'o', markersize=8, color='orange', label='Sol')

        return []

    # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50) # np.arange(len(t)),

    plt.show()

def plot_solution(t, Y, Y_ref): # Plot variables and errors
      
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

    plt.show()

#Plot the complete optimized Case II trajectory
def plot2D_caseII(t, Y, R_f, t_SOI_in=None, t_SOI_out=None, t_MED=None):

    x = Y[0, :]
    y = Y[1, :]

    alpha = np.linspace(0.0, 2.0 * np.pi, 800)

    earth_orbit_x = cts.R_orb_A * np.cos(alpha)
    earth_orbit_y = cts.R_orb_A * np.sin(alpha)

    saturn_orbit_x = cts.R_orb_B * np.cos(alpha)
    saturn_orbit_y = cts.R_orb_B * np.sin(alpha)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    ax.plot(earth_orbit_x, earth_orbit_y, "--", linewidth=1.0, label="Earth orbit")
    ax.plot(saturn_orbit_x, saturn_orbit_y, "--", linewidth=1.0, label="Saturn orbit")
    ax.plot(x, y, "-", linewidth=1.2, label="Spacecraft trajectory")

    ax.plot(0.0, 0.0, "o", markersize=7, label="Sun")
    ax.plot(x[0], y[0], "o", markersize=5, label="Departure")
    ax.plot(x[-1], y[-1], "o", markersize=5, label="Arrival at R_B")

    if t_SOI_in is not None:
        i_in = int(np.argmin(np.abs(t - t_SOI_in)))
        ax.plot(x[i_in], y[i_in], "s", markersize=5, label="SOI entry")

    if t_SOI_out is not None:
        i_out = int(np.argmin(np.abs(t - t_SOI_out)))
        ax.plot(x[i_out], y[i_out], "s", markersize=5, label="SOI exit")

    if t_MED is not None:
        i_med = int(np.argmin(np.abs(t - t_MED)))
        ax.plot(x[i_med], y[i_med], "x", markersize=7, label="Closest Earth approach")

    ax.set_title("Case II complete heliocentric trajectory")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    plt.show()

def plot_distances(t, Y, R_f, title="", t_SOI_in=None, t_SOI_out=None, t_MED=None):

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

    plt.show()