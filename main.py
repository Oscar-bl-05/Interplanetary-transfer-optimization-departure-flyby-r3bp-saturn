import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include import plotter

print("Initializing simulation, pls wait ...")

k_def = 1
k = 1

nstep = int(4e3)
atol = np.array([1e0, 1e0, 1e-4, 1e-4])  # km, km, km/s, km/s
rtol = 1e-6

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

def simulate(nstep, atol, rtol, tf, t0=0.0, k=1.0, theta0=analitical.theta_0I, check_errors=True):
    t = np.linspace(t0, tf, nstep + 1, endpoint=True)

    baseY0, t_hat_theta = IC.ICtoY0(IC.rho0, theta0=theta0, delta0=IC.delta0)

    V_ign = k * analitical.deltaV_ignI * t_hat_theta
    simY0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

    print("T_transfer (years) =", tf / (365.25 * 24 * 3600))
    print("deltaV_ignI (km/s) =", np.hypot(V_ign[0], V_ign[1]))
    print("deltaV_1H   (km/s) =", analitical.deltaV_1H)

    t1 = time.time()
    sol = solve_ivp(F, (t0, tf), simY0, t_eval=t, method="DOP853", atol=atol, rtol=rtol)
    t2 = time.time()

    r = np.hypot(sol.y[0], sol.y[1])
    print("runtime =", t2 - t1)
    print("r_max =", int(r.max()), "target =", cts.R_orb_B, "dif =", cts.R_orb_B - int(r.max()), "k =", k)

    if check_errors:
        t3 = time.time()
        print("Checking errors ...")
        sol_ref = solve_ivp(
            F, (t0, tf), simY0, t_eval=t, method="DOP853",
            atol=np.array([1e-6, 1e-6, 1e-10, 1e-10]), rtol=1e-12
        )
        t4 = time.time()
        print("errCheck runtime =", t4 - t3)
        return sol, sol_ref
    else:
        return sol, None

sol, sol_ref = simulate(
    nstep=nstep,
    atol=atol,
    rtol=rtol,
    t0=0.0,
    tf=float(analitical.T_transfer),
    k=k_def,
    theta0=analitical.theta_0I,
    check_errors=True
)
dt = (analitical.T_transfer - 0) / nstep
k_sweep = np.linspace(0.4, 0.99, 60) #valores de k a probar, hay que hacer otro de teta
theta_sweep = np.linspace(0, 1.5, 6)

def sweep(values_to_sweep, parameter): # Queda por implementar que haga optimización de ambos valores a la vez
    results = []
    if parameter == "k" or parameter == "deltaV":
        for vts in values_to_sweep:

            sol, sol_ref = simulate(
                nstep = nstep, 
                atol = atol, 
                rtol = rtol,
                t0 = 0.0,
                tf = float(analitical.T_transfer),
                k=vts,
                check_errors = False)
            
            #print("plot debug check")
            #plotter.plot_solution(sol.t, sol.y, sol_ref.y)
            results.append(sol)

    elif parameter == "theta":
        for vts in values_to_sweep:
            sol, sol_ref = simulate(
                nstep = nstep, 
                atol = atol, 
                rtol = rtol,
                t0 = 0.0,
                tf = float(analitical.T_transfer),
                k=k_def,
                check_errors = False)
            print("Theta0 =", vts)
            results.append(sol)

    else:
        print("No specified paramater to sweep")

    return results

sweep(theta_sweep, parameter="theta")

plotter.plot_solution(sol.t, sol.y, sol_ref.y)
plotter.plot2D(sol.t, dt, sol.y, R)