import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include import plotter

print("Initializing simulation, pls wait ...")
k = 0.8 # asegurar que llega a R_B
k = 1 # asegurar que llega a R_B

nstep = int(4e2)
atol = np.array([1e0, 1e0, 1e-4, 1e-4])*1e-4  # km, km, km/s, km/s
rtol = 1e-8

def R(t, R_orb, frec):
    return [
        R_orb * np.cos(frec * t - IC.delta0),
        R_orb * np.sin(frec * t - IC.delta0),
    ]

def F(t, Y):
    x, y, vx, vy = Y
    r = np.hypot(x, y)
    mu_r3 = cts.mu_sun / (r**3)

    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    Rm = np.hypot(Rx, Ry)
    Rm3 = 1.0 / (Rm**3)

    dx = x - Rx
    dy = y - Ry
    dm = np.hypot(dx, dy)
    dm3 = 1.0 / (dm**3)

    ax = (-x * mu_r3) - cts.mu_earth * (dx * dm3 + Rx * Rm3)
    ay = (-y * mu_r3) - cts.mu_earth * (dy * dm3 + Ry * Rm3)

    return np.array([vx, vy, ax, ay])

V_ign = k * analitical.deltaV_ignI * IC.initial_conditions(analitical.theta_0I)[1] # habria que optimizar
Y0 = IC.initial_conditions(analitical.theta_0I)[0] + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

def simulate(nstep, atol, rtol, tf, t0 = 0.0, k=1.0, Y0 = IC.Y0, check_errors = True): ## mejor renombrar las variables internas para que no se pisen

    t = np.linspace(t0, tf, nstep + 1, endpoint=True)

    V_ign = k * analitical.deltaV_ignI * IC.t_hat_theta # habria que optimizar
    Y0 += np.array([0.0, 0.0, V_ign[0], V_ign[1]])

    print("T_transfer (years) =", tf / (365.25 * 24 * 3600))
    print("deltaV_ignI (km/s) =", analitical.deltaV_ignI)
    print("deltaV_1H   (km/s) =", analitical.deltaV_1H)

    t1 = time.time()

    sol = solve_ivp(F, (t0, tf), Y0, t_eval=t, method="DOP853", atol=atol, rtol=rtol)

    t2 = time.time()

    r = np.hypot(sol.y[0], sol.y[1])
    print("runtime =", t2 - t1)
    print("r_max =", int(r.max()), "target =", cts.R_orb_B)

    # Solución de referencia para errores (paso 6)
    if check_errors:
        sol_ref = solve_ivp(F, (t0, tf), Y0, t_eval=t, method="DOP853", atol=np.array([1e-6, 1e-6, 1e-10, 1e-10]), rtol=1e-12)
        return sol, sol_ref
    else:
        return sol, None


sol, sol_ref = simulate(
    nstep = nstep, 
    atol = atol, 
    rtol = rtol,
    t0 = 0.0,
    tf = float(analitical.T_transfer),
    Y0 = IC.Y0,
    k=k,
    check_errors = True)

print("plotting...")
dt = (analitical.T_transfer - 0) / nstep
plotter.plot_solution(sol.t, sol.y, sol_ref.y)
plotter.plot2D(sol.t, dt, sol.y, R)
#plotter.plot_solution(sol.t, sol.y, sol_ref.y)
#plotter.plot2D(sol.t, dt, sol.y, R)
