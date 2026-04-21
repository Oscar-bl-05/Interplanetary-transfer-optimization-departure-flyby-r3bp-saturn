import numpy as np
import time
import math
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include import plotter

print("Initializing simulation, pls wait ...")
k = 1 # asegurar que llega a R_B

nstep = int(1.6e2)
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

def solution (v,theta):
        V_ign = k * v * IC.initial_conditions(theta)[1]
        Y0 = IC.initial_conditions(theta)[0] + np.array([0.0, 0.0, V_ign[0], V_ign[1]])
        sol = solve_ivp(F, (t0, tf), Y0, t_eval=t, method="DOP853", atol=atol, rtol=rtol)
        return(sol)

t0 = 0.0
tf = float(analitical.T_transfer)
dt = (tf - t0) / nstep
t = np.linspace(t0, tf, nstep + 1, endpoint=True)

seed = (analitical.deltaV_ignI, analitical.theta_0I)

t1 = time.time()

def optimize(f, variables, maxiter):
    vel0 = variables[0]
    theta0 = variables[1]
    gradvel0 = 1000
    gradtheta0 = 0.5
    ans = f(variables[0], variables[1])
    rthetaans = np.hypot(ans.y[0], ans.y[1])
    rvelans = np.hypot(ans.y[0], ans.y[1])
    for i in range(maxiter):
        testvel = vel0 + gradvel0
        testtheta = theta0 + gradtheta0

        soltheta = f(vel0, testtheta)
        solvel = f(testvel,theta0)

        rtheta = np.hypot(soltheta.y[0], soltheta.y[1])
        rvel = np.hypot(solvel.y[0], solvel.y[1])

        rdiftheta = rtheta.max() - rthetaans.max()
        rdifvel = rvel.max() - rvelans.max()

        fromgoaltheta = cts.R_orb_B - rtheta.max()
        fromgoalvel = cts.R_orb_B - rvel.max()
        
        thetaratio = math.copysign(math.exp(-abs(rdiftheta/fromgoaltheta)),fromgoaltheta)
        velratio = math.copysign(math.exp(-abs(rdifvel/fromgoalvel)),fromgoalvel)

        gradvel0 = gradvel0*(1+velratio)
        gradtheta0 = gradtheta0*(1+thetaratio)

        vel0 = testvel
        theta0 = testtheta

        rthetaans = rtheta
        rvelans = rvel

    return fromgoaltheta, fromgoalvel

t2 = time.time()

x1, x2 = optimize(solution, seed, 100)

print(f"Diferencia theta {x1}")
print (f"Diferencia vel {x2}")

print("runtime =", t2 - t1)
#print("r_max =", r.max(), "target =", cts.R_orb_B)
