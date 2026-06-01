import numpy as np
from scipy.integrate import solve_ivp
from include import cts, analitical, IC, plotter ##quitar ploter despues de debugear

def optimize_case_I(F, nstep, atol, rtol, tf, t0=0.0, n_grid=10, n_refines=2,
                    theta_center=None, dv_center=None, theta_span=0.5, dv_span=None):
    if theta_center is None:
        theta_center = float(analitical.theta_0I)
    if dv_center is None:
        dv_center = float(analitical.deltaV_ignI)
    if dv_span is None:
        dv_span = 0.3 * dv_center


    def reach_RB(t, Y): #detecta la llegada a punto lagrange de planeta B
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B
    reach_RB.terminal = True
    reach_RB.direction = 1

    def v_circ_at_r(x, y):
        r = np.hypot(x, y)
        vmod = np.sqrt(cts.mu_sun / r)
        t_hat = np.array([-y / r, x / r])
        return vmod * t_hat

    def evaluate(theta0, dv_ign, dt):
        baseY0, t_hat_theta = IC.ICtoY0(IC.rho0, theta0=theta0, delta0=IC.delta0)
        V_ign = dv_ign * t_hat_theta
        Y0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

        sol = solve_ivp(
            F, (t0, tf), Y0,
            method="DOP853",
            atol=atol, rtol=rtol,
            events=reach_RB,
            max_step=dt
        )

        if len(sol.t_events[0]) == 0:
            return None

        t_fin = sol.t_events[0][0]
        y_fin = sol.y_events[0][0]

        v_target = v_circ_at_r(y_fin[0], y_fin[1])
        v_sc = np.array([y_fin[2], y_fin[3]])
        dv_fin = np.linalg.norm(v_target - v_sc)
        dv_tot = abs(dv_ign) + abs(dv_fin)

        return {
            "theta": theta0,
            "dv_ign": dv_ign,
            "dv_fin": dv_fin,
            "dv_tot": dv_tot,
            "t_fin": t_fin,
            "y_fin": y_fin,
            "sol": sol
        }

    def grid_search(theta_c, dv_c, th_span, dv_sp):
        dt = (tf - t0) / nstep
        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid)

        best = None
        best_cost = np.inf

        for th in theta_vals:
            for dv in dv_vals:
                if dv <= 0:
                    continue
                out = evaluate(th, dv, dt)
                if out is None:
                    continue
                if out["dv_tot"] < best_cost:
                    best_cost = out["dv_tot"]
                    best = out

        return best

    best = grid_search(theta_center, dv_center, theta_span, dv_span)
    if best is None:
        return None

    th_span = theta_span
    dv_sp = dv_span
    for _ in range(n_refines):
        th_span *= 0.25
        dv_sp *= 0.25
        best = grid_search(best["theta"], best["dv_ign"], th_span, dv_sp)
        if best is None:
            return None

    return best

def optimize_case_II(F, nstep, atol, rtol, tf, t0=0.0, n_grid_deltav=10, n_grid_theta=10, n_refines=2, theta_center=None, dv_center=None, theta_span=0.5, dv_span=None, narrowband_exponent = 2, mode = "deltaV"):
    
    if theta_center is None:
        theta_center = float(analitical.theta_0II)
    if dv_center is None:
        dv_center = analitical.deltaV_ignII
    if dv_span is None:
        dv_span = 1/(2**narrowband_exponent) * dv_center ## a unit increase in the narrowband exponent halves the dv span

    def reach_RB(t, Y):
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B
    reach_RB.terminal = True
    reach_RB.direction = 1

    if False: #borrar?
        def reach_desired_a(t, Y):
            return np.hypot(Y[0], Y[1]) - analitical.desired_R_max
        reach_desired_a.terminal = True
        reach_desired_a.direction = 1
    

    def v_circ_at_r(x, y):
        r = np.hypot(x, y)
        vmod = np.sqrt(cts.mu_sun / r)
        t_hat = np.array([-y / r, x / r])
        return vmod * t_hat
    
    def get_MED(sol):
        earth_x = cts.R_orb_A * np.cos(cts.frec_A * sol.t - IC.delta0)
        earth_y = cts.R_orb_A * np.sin(cts.frec_A * sol.t - IC.delta0)

        earth_distance = np.hypot(sol.y[0] - earth_x, sol.y[1] - earth_y)

        after_departure = sol.t > 0.5 * cts.year_to_s

        earth_distance_after_departure = earth_distance[after_departure]
        time_after_departure = sol.t[after_departure]

        closest_index = earth_distance_after_departure.argmin()

        minimum_earth_distance = earth_distance_after_departure[closest_index]
        time_of_closest_earth_return = time_after_departure[closest_index]

        return minimum_earth_distance, time_of_closest_earth_return
    
    def get_r_max(sol):
        r = np.hypot(sol.y[0], sol.y[1])
        r_max = r[r.argmax()] # r.max() no me funcionaba????
        return r_max

    def evaluate(theta0, dv_ign, dt):
        baseY0, t_hat_theta = IC.ICtoY0(IC.rho0, theta0=theta0, delta0=IC.delta0)
        V_ign = dv_ign * t_hat_theta
        Y0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

        sol = solve_ivp(
            F, (t0, tf), Y0,
            method="DOP853",
            atol=atol, rtol=rtol,
            events=reach_RB, #para hallar el deltaV hasta conseguir (a) deseado cambiar evento a reach_desired_a()
            max_step=dt
        )
        ############################################
        def R(t, R_orb, frec):
            return np.array([
                R_orb * np.cos(frec * t - IC.delta0),
                R_orb * np.sin(frec * t - IC.delta0),
            ])

        if False:
            dt_plot=(tf - t0) / nstep
            plotter.plot2D(sol.t, dt_plot, sol.y, R)
        ##############################################
        if mode == "deltaV":
            if len(sol.t_events[0]) == 0:
                return None


            t_fin = sol.t_events[0][0]
            y_fin = sol.y_events[0][0]

            v_target = v_circ_at_r(y_fin[0], y_fin[1])
            v_sc = np.array([y_fin[2], y_fin[3]])
            dv_fin = np.linalg.norm(v_target - v_sc)
            dv_tot = abs(dv_ign) + abs(dv_fin)


            return {
                "theta": theta0,
                "dv_ign": dv_ign,
                "dv_fin": dv_fin,
                "dv_tot": dv_tot,
                "t_fin": t_fin,
                "y_fin": y_fin,
                "sol": sol,
            }
        
        elif mode == "MED":

            minimum_earth_distance, time_of_closest_earth_return = get_MED(sol)

            if len(sol.t_events[0]) != 0:
                reached_RB = True
                t_fin = sol.t_events[0][0]
                y_fin = sol.y_events[0][0]

                v_target = v_circ_at_r(y_fin[0], y_fin[1])
                v_sc = np.array([y_fin[2], y_fin[3]])
                dv_fin = np.linalg.norm(v_target - v_sc)
                dv_tot = abs(dv_ign) + abs(dv_fin)

                return {
                    "reached_R_B": reached_RB,
                    "theta": theta0,
                    "dv_ign": dv_ign,
                    "dv_fin": dv_fin,
                    "dv_tot": dv_tot,
                    "t_fin": t_fin,
                    "y_fin": y_fin,
                    "sol": sol,
                    "MED": minimum_earth_distance,
                    "t_MED": time_of_closest_earth_return
                }
   
            else:
                reached_RB = False

                return {
                    "reached_R_B": reached_RB,
                    "theta": theta0,
                    "dv_ign": dv_ign,
                    "sol": sol,
                    "MED": minimum_earth_distance,
                    "t_MED": time_of_closest_earth_return
                }
            
        elif mode == "deltaV*":

            minimum_earth_distance, time_of_closest_earth_return = get_MED(sol)
            r_max = get_r_max(sol) #podría implementarse de otra manera pero paso

            if len(sol.t_events[0]) != 0:
                reached_RB = True
                t_fin = sol.t_events[0][0]
                y_fin = sol.y_events[0][0]

                v_target = v_circ_at_r(y_fin[0], y_fin[1])
                v_sc = np.array([y_fin[2], y_fin[3]])
                dv_fin = np.linalg.norm(v_target - v_sc)
                dv_tot = abs(dv_ign) + abs(dv_fin)

                return {
                    "reached_R_B": reached_RB,
                    "theta": theta0,
                    "dv_ign": dv_ign,
                    "dv_fin": dv_fin,
                    "dv_tot": dv_tot,
                    "t_fin": t_fin,
                    "y_fin": y_fin,
                    "sol": sol,
                    "MED": minimum_earth_distance,
                    "t_MED": time_of_closest_earth_return
                }
   
            else:
                reached_RB = False

                return {
                    "reached_R_B": reached_RB,
                    "theta": theta0,
                    "dv_ign": dv_ign,
                    "sol": sol,
                    "MED": minimum_earth_distance,
                    "t_MED": time_of_closest_earth_return,
                    "r_max": r_max
                }

    def grid_search_deltaV(theta_c, dv_c, th_span, dv_sp):
        dt = (tf - t0) / nstep
        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid_theta)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid_deltav)

        best = None
        best_cost = np.inf

        for th in theta_vals:
            for dv in dv_vals:
                if dv <= 0:
                    continue
                out = evaluate(th, dv, dt)
                if out is None:
                    continue
                if out["dv_tot"] < best_cost:
                    best_cost = out["dv_tot"]
                    best = out

        return best
    
    def grid_search_MED(theta_c, dv_c, th_span, dv_sp):
        dt = (tf - t0) / nstep
        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid_theta)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid_deltav)

        best = None
        best_MED = np.inf

        reached_R_B = False

        for th in theta_vals:
            for dv in dv_vals:
                if dv <= 0:
                    continue
                out = evaluate(th, dv, dt)
                if out is None:
                    continue

                if out["reached_R_B"] and (reached_R_B == False):
                    print("OCURRIO")
                    reached_R_B = True
                    best_MED = out["dv_tot"]
                    best = out
                    
                # Chekear que MED este por encima del radio inicial de la nave (1.1 x R_A)
                if out["MED"] <= IC.rho0:
                    continue
                
                # Chekear que MED esté dentro de la SOI
                if out["MED"] >= cts.earth_SOI_radius:
                    #print("No gravity assist")
                    continue

                if (out["MED"] < best_MED) and not reached_R_B:
                    best_MED = out["MED"]
                    best = out

        return best
    
    def grid_search_assisted_deltaV(theta_c, dv_c, th_span, dv_sp):
        dt = (tf - t0) / nstep
        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid_theta)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid_deltav)

        best = None
        best_cost = np.inf
        best_radius = 0

        reached_R_B = False

        for th in theta_vals:
            for dv in dv_vals:
                if dv <= 0:
                    continue

                out = evaluate(th, dv, dt)
                if out is None:
                    continue

                if out["reached_R_B"]:
                    if reached_R_B == False:
                        print("OCURRIO")
                        reached_R_B = True
                        best_cost = out["dv_tot"]
                        best = out
                    elif (out["dv_tot"] < best_cost):
                        best_cost = out["dv_tot"]
                        best = out
                    
                # Chekear que MED este por encima del radio inicial de la nave (1.1 x R_A)
                if out["MED"] <= IC.rho0:
                    #print("Fell into the earth")
                    if out["reached_R_B"]:
                        print("reached RB after plunging into earth")
                    continue
                
                # Chekear que MED esté dentro de la SOI
                if out["MED"] >= cts.earth_SOI_radius:
                    #print("No gravity assist")
                    continue

                if (out["r_max"] > best_radius) and not reached_R_B:
                    best_radius = out["r_max"]
                    best = out

        return best

    if mode == "deltaV":
        best = grid_search_deltaV(theta_center, dv_center, theta_span, dv_span)
        if best is None:
            return None

        th_span = theta_span
        dv_sp = dv_span
        for _ in range(n_refines):
            th_span *= 0.1
            dv_sp *= 0.1
            best = grid_search_deltaV(best["theta"], best["dv_ign"], th_span, dv_sp)
            if best is None:
                return None
            
    elif mode == "MED":

        best = grid_search_MED(theta_center, dv_center, theta_span, dv_span)
        th_span = theta_span
        dv_sp = dv_span
        if best is None:
            print("Something went wrong")
            return None
        for _ in range(n_refines):
            th_span *= 0.1
            dv_sp *= 0.1
            best = grid_search_MED(best["theta"], best["dv_ign"], th_span, dv_sp)
    
    elif mode == "deltaV*":

        best = grid_search_assisted_deltaV(theta_center, dv_center, theta_span, dv_span)
        th_span = theta_span
        dv_sp = dv_span
        if best is None:
            print("Something went wrong")
            return None
        for _ in range(n_refines):
            th_span *= 0.1
            dv_sp *= 0.1
            best = grid_search_assisted_deltaV(best["theta"], best["dv_ign"], th_span, dv_sp)
        
    else:
        print("Invalid 'mode' argument")
        return None

    return best