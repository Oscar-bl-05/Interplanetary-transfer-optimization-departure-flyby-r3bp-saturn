import numpy as np
coc = R_orb_B/(R_orb_A*(R_orb_A+R_orb_B))
sec = mu_sun/R_orb_A
deltaV_1H = np.sqrt(2*mu_sun*coc) - np.sqrt(sec)
deltaV_ignI = np.sqrt(deltaV_1H*deltaV_1H + 2*mu_earth/rho0)
