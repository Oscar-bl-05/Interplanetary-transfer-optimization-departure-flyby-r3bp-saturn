import cts 
import IC
from numpy import sqrt

transfertime_safety_margin = 1.4
T_transfer = transfertime_safety_margin * 3 * cts.pi * sqrt((cts.R_orb_A + cts.R_orb_B)**3/(8*cts.mu_sun))
