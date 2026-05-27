from . import cts 
from . import IC
from numpy import sqrt, arcsin, power


# --- FACTORES DE TIEMPO DE SIMULACIÓN ---

transfertime_safety_factor = 1.25

# Caso II:
caseII_transfertime_multiplier = 3 # Earth rotations before grav assist


# --- TIEMPOS DE TRANSFERENCIA ---

# Caso I:
T_transfer_case_I = transfertime_safety_factor * cts.pi * sqrt((cts.R_orb_A + cts.R_orb_B)**3/(8*cts.mu_sun))

# Caso II:
T_transfer_case_II = transfertime_safety_factor * caseII_transfertime_multiplier * cts.T_orb_A + T_transfer_case_I


# --- CASO II - ESTIMACIÓN POR RESONANCIA ---

# Para favorecer la asistencia gravitacional, la primera elipse heliocéntrica
# debería estar cerca de una resonancia con la Tierra:

# n * T(a) = n_earth * T_earth
# con n_earth > n

# Primera prueba:
# n = 1
# n_earth = 2

resonance_n = 1
resonance_n_earth = 2

# Periodo objetivo de la primera elipse resonante
T_resonance = (resonance_n_earth / resonance_n) * cts.T_orb_A

# T(a) = 2*pi*sqrt(a^3/mu_sol)
# => a = ((T/2*pi)^2*mu_sol)^(1/3)
desired_a = power(cts.mu_sun*(T_resonance/(2*cts.pi))**2, 1/3)

# a = (r_peri + r_apo)/2
# r_peri ~= R_orb_A:
desired_R_max = 2*desired_a - cts.R_orb_A


# --- CASO I - ESTIMACIONES ANALÍTICAS ---

# Cts 5.1 (Parámetros de integración)

# Primer delta-v de Hohmann entre la órbita de A y la órbita de B
coc = cts.R_orb_B/(cts.R_orb_A*(cts.R_orb_A+cts.R_orb_B))
deltaV_1H = sqrt(2*cts.mu_sun*coc) - sqrt(cts.mu_sun/cts.R_orb_A)

# Delta-v de ignición estimado para el Caso I.
deltaV_ignI = sqrt(deltaV_1H*deltaV_1H + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)

# Segundo delta-v de Hohmann, usado como estimación analítica del delta-v final
coc = cts.R_orb_A/(cts.R_orb_B*(cts.R_orb_A+cts.R_orb_B))
deltaV_2H = sqrt((cts.mu_sun/cts.R_orb_B)) - sqrt(2*cts.mu_sun*coc)
deltaV_finI = abs(deltaV_2H) # Not sure


# --- CASO II - ESTIMACIÓN ANALÍTICA DE DELTA-V DE IGNICIÓN ---

#v_infII = 0.8 * deltaV_1H # Asumimos un vinf del 80% de deltaV_1H

#deltaV_ignII = sqrt(v_infII*v_infII + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)

# Nueva estimación tras revisión de Daniele:
# Velocidad en perihelio de la primera elipse resonante heliocéntrica
v_perihelion_resonance = sqrt(cts.mu_sun * (2.0/cts.R_orb_A - 1.0/desired_a))

# Velocidad heliocéntrica circular de la Tierra.
v_earth = sqrt(cts.mu_sun/cts.R_orb_A)

# Velocidad hiperbólica para la primera elipse
v_infII = v_perihelion_resonance - v_earth

# Delta-v de ignición del Caso II asociado a la condición resonante
# como en el Caso I, pero usando v_infII en lugar de deltaV_1H
deltaV_ignII = sqrt(v_infII*v_infII + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)

#completely_not_pulled_out_of_my_ass_value = 4.434731316839158
completely_not_pulled_out_of_my_ass_value = deltaV_ignII  # luego cambiar nombre en optimizer y test_case_ii
#theta = -0.9874019972342188 deltaV = 4.322797781064377 > encuentro con la Tierra tras 2 años


# --- THETA DE IGNICIÓN --

# Caso I:
# v_inf = deltaV_1H
e_I = 1 + (IC.rho0*deltaV_1H*deltaV_1H)/cts.mu_earth
theta_0I = -2 * arcsin(1/e_I)

# Caso II:
# v_inf esta contenido entre [0, deltaV_1H]
# v_infII viene de la primera elipse resonante.
e_II = 1 + (IC.rho0*v_infII*v_infII)/cts.mu_earth 
theta_0II = -2 * arcsin(1/e_II)