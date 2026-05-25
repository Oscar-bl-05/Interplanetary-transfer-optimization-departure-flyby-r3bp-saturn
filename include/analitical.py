from . import cts 
from . import IC
from numpy import sqrt, arcsin, pow

transfertime_safety_factor = 1.25
caseII_transfertime_multiplier = 3 # Earth rotations before grav assist

T_transfer_case_I = transfertime_safety_factor * cts.pi * sqrt((cts.R_orb_A + cts.R_orb_B)**3/(8*cts.mu_sun))
# tiempo analítico caso 2: 3 periodos de la tierra + el tiempo del caso 1
T_transfer_case_II = transfertime_safety_factor * caseII_transfertime_multiplier * cts.T_orb_A + T_transfer_case_I


#T(a) = 2*pi*sqrt(a^3/mu_sol) => a = ((T/2*pi)^2*mu_sol)^1/3

desired_a = pow(cts.mu_sun*(caseII_transfertime_multiplier*cts.T_orb_A/(2*cts.pi))**2 , 1/3)
# teniendo en cuenta que el periapside va a ser ~R_orb_A, calcular el apoápside deseado es simplemente R_Ap = a - R_orb_A
desired_R_max = desired_a-cts.R_orb_A

# dv_ign caso II (aprox: 4.434731316839158)
completely_not_pulled_out_of_my_ass_value = 4.434731316839158

# Cts 5.1(Parámetros de integración)

# Caso I
coc = cts.R_orb_B/(cts.R_orb_A*(cts.R_orb_A+cts.R_orb_B))
deltaV_1H = sqrt(2*cts.mu_sun*coc) - sqrt(cts.mu_sun/cts.R_orb_A)
deltaV_ignI = sqrt(deltaV_1H*deltaV_1H + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)
coc = cts.R_orb_A/(cts.R_orb_B*(cts.R_orb_A+cts.R_orb_B))
deltaV_2H = sqrt((cts.mu_sun/cts.R_orb_B)) - sqrt(2*cts.mu_sun*coc)
deltaV_finI = abs(deltaV_2H) # Not sure

# Caso II
v_infII = 0.8 * deltaV_1H # Asumimos un vinf del 80% de deltaV_1H
deltaV_ignII = sqrt(v_infII*v_infII + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)
# Theta de ignición

# Caso I: vinf = deltaV_1H
e = 1 + (IC.rho0*deltaV_1H*deltaV_1H)/cts.mu_earth
theta_0I = -2 * arcsin(1/e)

# Caso II: vinf esta contenido entre [0, deltaV_1H]
e = 1 + (IC.rho0*v_infII*v_infII)/cts.mu_earth 
theta_0II = -2 * arcsin(1/e)