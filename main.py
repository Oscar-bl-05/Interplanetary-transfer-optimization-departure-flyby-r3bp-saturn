import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC, plotter


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
# 9 check
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

k_def = 1 # duplicado xd, quitar uno
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
k_sweep = np.linspace(0.85, 1.05, 11) #valores de k a probar, hay que hacer otro de teta
theta_sweep = np.linspace(-1.5, 1.5, 11)

def sweep(values_to_sweep, parameter):
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
                theta0=vts,
                k=k_def,
                check_errors = False)
            print("Theta0 =", vts)
            results.append(sol)
    elif parameter == ["k","theta"]:
        for vts_k in values_to_sweep[0]:
            for vts_th in values_to_sweep[1]:
                sol, sol_ref = simulate(
                nstep = nstep, 
                atol = atol, 
                rtol = rtol,
                t0 = 0.0,
                tf = float(analitical.T_transfer),
                theta0=vts_th,
                k=vts_k,
                check_errors = False)
                print("Theta0 =", vts_th)
                results.append(sol)
    else:
        print("Unrecognized sweeping parameter(s)")

    return results

sweep([k_sweep,theta_sweep],["k", "theta"])

plotter.plot_solution(sol.t, sol.y, sol_ref.y)
plotter.plot2D(sol.t, dt, sol.y, R)