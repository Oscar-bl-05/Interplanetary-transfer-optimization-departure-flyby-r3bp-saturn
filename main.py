import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include import plotter
from optimizer import optimize_case_I

"""
1. Documentarse
2. Escribir las ecuaciones dinámicas ˙Y = F(t,Y ) 
3. Implementar las ecuaciones dinámicas y las condiciones iniciales como funciones
en un código de Python para su solución numérica.
4. Usar las estimaciones analíticas del tiempo de transferencia en cada caso para
elegir la duración de las simulaciones.
5. Probar el código solucionando las ecuaciones dinámicas en dos ejemplos (no
optimizados), uno del caso I y otro del II, eligiendo como condiciones iniciales
las estimaciones analíticas de la sección B.5 (en el caso II, elegir un valor v∞
intermedio, por ejemplo v∞ = 0,8∆vH)
6. Representar las soluciones del punto anterior, obteniendo las gráficas de las
variables dinámicas (posición y velocidad con respecto al sol) y de sus errores.
7. Comprobar que los errores numéricos globales cuando se alcanza la distancia
RB sean aceptables. Si hay margen, variar las tolerancias para encontrar valores
mayores que sigan dando errores aceptables (por ejemplo menores de 1000 km
para las longitudes y de 10 m/s para las velocidades). Elegir así tolerancias que
sean suficientes para tener errores aceptables, pero que aceleren los cálculos
para los bucles de optimización.
8. En cada caso considerado (empezando con el I), definir un array de 10 valores
de θ y un array de 10 valores de ∆vign, teniendo en cuenta las estimaciones de
la sección B.5.1.
9. Solucionar numéricamente las ecuaciones dinámicas para cada pareja de valores
de θ y ∆vign (doble bucle) usando las tolerancias más laxas compatibles con
los errores que se encontraron en el punto 7.
10. Para cada una de tales soluciones, determinar tfin, definido como el primer
tiempo en que r alcanza la distancia RB (dentro de un margen de error) Si no
se alcanza esa distancia, se descarta el par de valores de θ y ∆vign. (Descartar
los casos en que no se llega a alcanzar RB.)
11. Usar los valores de ∆vign y de v(tfin) para calcular δv(I)
tot correspondiente a cada
par de valores de θ y ∆vign.
12. Elegir los valores de θ y ∆vign que den lugar al menor gasto en delta v total.
13. Repetir los pasos del 8 al 12 incluidos con array refinados alrededor de los
valores de θ y ∆vign obtenidos en el paso 12. Se obtiene así un cálculo más
preciso de los valores óptimos de θ y ∆vign (si necesario, repetir de nuevo el
refinamiento hasta obtener resultados con la precisión deseada)
14. Repetir los pasos del 8 al 13 para el caso II (cambiando oportunamente los
arrays de condiciones iniciales y el tiempo de simulación).
15. Para las soluciones óptimas de los dos casos I y II obtenidas, representar las
gráficas de la evolución temporal de las variables dinámicas (posición y velocidad
con respecto al sol) y de sus errores.
16. Comprobar que los errores numéricos sean aceptables.
17. Siempre para las soluciones óptimas de los dos casos I y II obtenidas, incluir
también gráficas de r(t) (distancia del sol) y de |r(t) - R(t)| (distancia del
planeta). Y gráficas en dos dimensiones de la trayectoria vista desde el sol
(poniendo x, y en los ejes) y desde el centro del planeta A (poniendo x - X,
y -Y en los ejes, donde X,Y son las componentes del vector R(t)).
18. En los dos casos óptimos, representar la evolución de los elementos orbitales
calculados respecto al sol y los relativos al planeta A
19. Realizar una animación de la trayectoria.
20. Discutir los resultados de los puntos anteriores. En particular, comparar el
consumo en delta v total de los casos I y II optimizados, e interpretar el
resultado.
21. Redactar las conclusiones.
22. Adjuntar los códigos y las figuras a la memoria. (El nombre de los ficheros de
cada grupo contendrá el nombre de los planetas A-B. En la primera página de
memoria aparecerán los nombres de los integrantes del grupo.
"""
# 1 check
# 2 check
# 3 check
# 4 check
# 5 check ; no tenemos un caso II funcional
# 6 check
# 7 check (revisar al acabar)
# 8 check
# 9 hace falta doble bucle (solo tenemos bucle simple)
# 10 falta implementarlo
# 11 falta implementarlo
# 12 falta implementarlo (optimizador en proceso...)
# 13 si construimos un optimizador de otra forma esto no es necesario (?)
# 14 más de lo mismo pal caso II
# 15 ya tenemos el plotter, solo falta la solución
# 16 (revisar al acabar)
# 17 falta implementarlo
# 18 falta implementarlo
# 19 Hecho
# 20 #21 #22 YAPPING


print("Initializing simulation, pls wait ...")

atol = np.array([1e0, 1e0, 1e-4, 1e-4])
rtol = 1e-6

atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12

nstep_opt = 750
nstep_plot = 4000

t0 = 0.0
tf = float(analitical.T_transfer)

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

def reach_RB(t, Y):
    return np.hypot(Y[0], Y[1]) - cts.R_orb_B

reach_RB.terminal = True
reach_RB.direction = 1

print("\nStarting optimization...")
t_opt_start = time.time()

best = optimize_case_I(
    F=F,
    nstep=nstep_opt,
    atol=atol,
    rtol=rtol,
    tf=tf,
    t0=t0,
    n_grid=10,
    n_refines=2
)

t_opt_end = time.time()

if best is None:
    print("No valid (theta, dv_ign) reached R_B in the explored grids.")
    raise SystemExit(1)

theta_opt = float(best["theta"])
dv_ign_opt = float(best["dv_ign"])

print("\n--- OPTIMUM (Case I) ---")
print("theta_opt (rad) =", theta_opt)
print("dv_ign_opt (km/s) =", dv_ign_opt)
print("dv_fin_opt (km/s) =", float(best["dv_fin"]))
print("dv_tot_opt (km/s) =", float(best["dv_tot"]))
print("t_fin_opt (years) =", float(best["t_fin"]) / (365.25 * 24 * 3600))
print("optimizer runtime (s) =", t_opt_end - t_opt_start)

baseY0, t_hat_theta = IC.ICtoY0(IC.rho0, theta0=theta_opt, delta0=IC.delta0)
V_ign = dv_ign_opt * t_hat_theta
Y0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

dt_event = (tf - t0) / nstep_opt

t_sim_start = time.time()
sol = solve_ivp(
    F, (t0, tf), Y0,
    method="DOP853",
    atol=atol, rtol=rtol,
    events=reach_RB,
    max_step=dt_event
)
t_sim_end = time.time()

if len(sol.t_events[0]) == 0:
    rmax = np.hypot(sol.y[0], sol.y[1]).max()
    print("\nDid NOT reach R_B with optimal parameters (unexpected).")
    print("r_max =", rmax, "missing =", cts.R_orb_B - rmax, "km")
    raise SystemExit(1)

t_hit = float(sol.t_events[0][0])
y_hit = sol.y_events[0][0]
r_hit = float(np.hypot(y_hit[0], y_hit[1]))

print("\n--- HIT (fast tolerances) ---")
print("t_hit (years) =", t_hit / (365.25 * 24 * 3600))
print("r_hit (km) =", r_hit)
print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))
print("sim runtime (s) =", t_sim_end - t_sim_start)

t_ref_start = time.time()
sol_ref_event = solve_ivp(
    F, (t0, tf), Y0,
    method="DOP853",
    atol=atol_ref, rtol=rtol_ref,
    events=reach_RB,
    max_step=dt_event
)
t_ref_end = time.time()

if len(sol_ref_event.t_events[0]) == 0:
    print("\nReference run did NOT reach R_B (unexpected).")
    raise SystemExit(1)

t_hit_ref = float(sol_ref_event.t_events[0][0])
y_hit_ref = sol_ref_event.y_events[0][0]

pos_err_km = float(np.hypot(y_hit[0] - y_hit_ref[0], y_hit[1] - y_hit_ref[1]))
vel_err_ms = float(1000.0 * np.hypot(y_hit[2] - y_hit_ref[2], y_hit[3] - y_hit_ref[3]))

print("\n--- GLOBAL NUMERICAL ERROR @ R_B (vs reference) ---")
print("t_hit_fast  (years) =", t_hit / (365.25 * 24 * 3600))
print("t_hit_ref   (years) =", t_hit_ref / (365.25 * 24 * 3600))
print("pos_err (km) =", pos_err_km)
print("vel_err (m/s) =", vel_err_ms)
print("ref runtime (s) =", t_ref_end - t_ref_start)

# -------- PLOTS DEL ÓPTIMO --------
t_plot = np.linspace(t0, t_hit, nstep_plot + 1, endpoint=True)
dt_plot = (t_hit - t0) / nstep_plot

sol_plot = solve_ivp(
    F, (t0, t_hit), Y0,
    t_eval=t_plot,
    method="DOP853",
    atol=atol, rtol=rtol
)

sol_plot_ref = solve_ivp(
    F, (t0, t_hit), Y0,
    t_eval=t_plot,
    method="DOP853",
    atol=atol_ref, rtol=rtol_ref
)

print("\nPlotting optimum trajectory and errors...")
plotter.plot_solution(sol_plot.t, sol_plot.y, sol_plot_ref.y)
plotter.plot2D(sol_plot.t, dt_plot, sol_plot.y, R)