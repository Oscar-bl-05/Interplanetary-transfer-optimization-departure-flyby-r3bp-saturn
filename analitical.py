import cts 
import IC
from numpy import sqrt, arcsin

transfertime_safety_margin = 1.4
T_transfer = transfertime_safety_margin * 3 * cts.pi * sqrt((cts.R_orb_A + cts.R_orb_B)**3/(8*cts.mu_sun))

# Cts 5.1(Parámetros de integración)
coc = cts.R_orb_B/(cts.R_orb_A*(cts.R_orb_A+cts.R_orb_B))
sec = cts.mu_sun/cts.R_orb_A
deltaV_1H = sqrt(2*cts.mu_sun*coc) - sqrt(sec)
deltaV_ignI = sqrt(deltaV_1H*deltaV_1H + 2*cts.mu_earth/IC.rho0)
deltaV_1H = sqrt(2*cts.mu_sun*coc) - sqrt(cts.mu_sun/cts.R_orb_A)
deltaV_ignI = sqrt(deltaV_1H*deltaV_1H + 2*cts.mu_earth/IC.rho0)

coc = cts.R_orb_A/(cts.R_orb_B*(cts.R_orb_A+cts.R_orb_B))
deltaV_2H = sqrt((cts.mu_sun/cts.R_orb_B)) - sqrt(2*cts.mu_sun*coc)
deltaV_finI = abs(deltaV_2H) # Not sure

# Theta de ignición
# Caso I: vinf = deltaV_1H
e = 1 + (IC.rho0*deltaV_1H*deltaV_1H)/cts.mu_earth
theta_0I = -2 * arcsin(1/e)
# Caso II: vinf esta contenido entre [0, deltaV_1H]
e = 1 + (IC.rho0*0.8*0.8*deltaV_1H*deltaV_1H)/cts.mu_earth # Asumimos un vinf del 80% de deltaV_1H
theta_0II = -2 * arcsin(1/e)