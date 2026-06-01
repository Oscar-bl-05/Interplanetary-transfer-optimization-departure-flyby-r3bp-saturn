from . import cts
from . import IC
from numpy import sqrt, arcsin, power


# --- FACTORES DE TIEMPO DE SIMULACIÓN ---

transfertime_safety_factor = 1.25

# Caso II:
# Se usa una resonancia 1:12 porque es la que da un primer arco heliocéntrico
# compatible con llegar a la distancia orbital de Saturno tras el flyby.
caseII_transfertime_multiplier = 12


# --- TIEMPOS DE TRANSFERENCIA ---

T_HOHMANN_HALF = cts.pi * sqrt((cts.R_orb_A + cts.R_orb_B)**3/(8*cts.mu_sun))

# Caso I:
T_transfer_case_I = transfertime_safety_factor * T_HOHMANN_HALF

# Caso II:
T_transfer_case_II = transfertime_safety_factor * (
    caseII_transfertime_multiplier * cts.T_orb_A + T_HOHMANN_HALF
)


# --- CASO I - ESTIMACIONES ANALÍTICAS ---

# Primer delta-v de Hohmann entre la órbita de A y la órbita de B
coc = cts.R_orb_B/(cts.R_orb_A*(cts.R_orb_A+cts.R_orb_B))
deltaV_1H = sqrt(2*cts.mu_sun*coc) - sqrt(cts.mu_sun/cts.R_orb_A)

# Delta-v de ignición estimado para el Caso I
deltaV_ignI = sqrt(deltaV_1H*deltaV_1H + 2*cts.mu_earth/IC.rho0) - sqrt(cts.mu_earth/IC.rho0)

# Segundo delta-v de Hohmann, usado como estimación analítica del delta-v final
coc = cts.R_orb_A/(cts.R_orb_B*(cts.R_orb_A+cts.R_orb_B))
deltaV_2H = sqrt((cts.mu_sun/cts.R_orb_B)) - sqrt(2*cts.mu_sun*coc)
deltaV_finI = abs(deltaV_2H)

# Theta de ignición del Caso I
e_I = 1 + (IC.rho0*deltaV_1H*deltaV_1H)/cts.mu_earth
theta_0I = -2 * arcsin(1/e_I)


# --- CASO II - ESTIMACIÓN POR RESONANCIA ---

caseII_time_safety_factor = 1.20


def resonance_case_II_estimate(n=1, n_earth=2):
    n = int(n)
    n_earth = int(n_earth)

    if n < 1:
        raise ValueError("n must be >= 1")

    if n_earth <= n:
        raise ValueError("n_earth must be greater than n")

    T_resonance = (n_earth/n) * cts.T_orb_A

    desired_a = power(
        cts.mu_sun*(T_resonance/(2*cts.pi))**2,
        1/3
    )

    desired_R_max = 2*desired_a - cts.R_orb_A

    v_perihelion_resonance = sqrt(
        cts.mu_sun * (2.0/cts.R_orb_A - 1.0/desired_a)
    )

    v_earth = sqrt(cts.mu_sun/cts.R_orb_A)

    v_infII = v_perihelion_resonance - v_earth

    deltaV_ignII = sqrt(
        v_infII*v_infII + 2*cts.mu_earth/IC.rho0
    ) - sqrt(cts.mu_earth/IC.rho0)

    e_II = 1 + (IC.rho0*v_infII*v_infII)/cts.mu_earth
    theta_0II = -2 * arcsin(1/e_II)

    T_transfer_case_II = caseII_time_safety_factor * (
        n_earth*cts.T_orb_A + T_transfer_case_I
    )

    return {
        "n": n,
        "n_earth": n_earth,
        "T_resonance": float(T_resonance),
        "desired_a": float(desired_a),
        "desired_R_max": float(desired_R_max),
        "v_infII": float(v_infII),
        "deltaV_ignII": float(deltaV_ignII),
        "theta_0II": float(theta_0II),
        "T_transfer_case_II": float(T_transfer_case_II),
        "valid_dv": bool(deltaV_ignII < deltaV_ignI)
    }


# Valor por defecto ligero. El barrido de test_case_II probará 2..12.
_default_caseII = resonance_case_II_estimate(n=1, n_earth=2)

resonance_n = _default_caseII["n"]
resonance_n_earth = _default_caseII["n_earth"]

T_resonance = _default_caseII["T_resonance"]
desired_a = _default_caseII["desired_a"]
desired_R_max = _default_caseII["desired_R_max"]

v_infII = _default_caseII["v_infII"]
deltaV_ignII = _default_caseII["deltaV_ignII"]
theta_0II = _default_caseII["theta_0II"]
T_transfer_case_II = _default_caseII["T_transfer_case_II"]
