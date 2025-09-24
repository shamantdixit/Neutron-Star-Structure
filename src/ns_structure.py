"""
Compares neutron star structure calculated with Newtonian gravity versus
the full general relativistic (TOV) equations.

This script highlights the importance of GR in describing compact objects.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from astropy import constants as const

# --- Physical Constants (CGS units) ---
G = const.G.cgs.value       # Gravitational constant (cm^3 g^-1 s^-2)
c = const.c.cgs.value       # Speed of light (cm s^-1)
M_sun = const.M_sun.cgs.value  # Solar mass (g)
CM_PER_KM = 1.0e-5          # Conversion factor from cm to km

# --- Equation of State (EoS) ---
# A simplified Fermi Gas model for degenerate neutron matter.
# The characteristic energy density scale is derived from fundamental constants.
eps0 = 5.346e36  # erg cm^-3 (nuclear saturation energy density)

# EoS is modeled as a sum of non-relativistic and relativistic components.
A_NR = 2.4216  # Coefficient for the non-relativistic term
A_R = 2.8663   # Coefficient for the ultra-relativistic term


def eps_bar(p_bar):
    """Calculates dimensionless energy density from dimensionless pressure."""
    return A_NR * p_bar**(3/5) + A_R * p_bar


def eps_phys(p_phys):
    """Calculates physical energy density (erg/cm^3) from physical pressure."""
    p_bar = p_phys / eps0  # Normalize pressure
    return eps0 * eps_bar(p_bar)


def newtonian_equations(r, y):
    """
    Defines the Newtonian equations of stellar structure.

    Args:
        r (float): Radius in cm.
        y (list): State vector [pressure, enclosed_mass].

    Returns:
        list: Derivatives [dP/dr, dM/dr].
    """
    p, M = y
    # Stop integration if pressure becomes non-physical.
    if p <= 0.0:
        return [0.0, 0.0]

    # Calculate density from pressure using the EoS.
    eps = eps_phys(p)
    rho = eps / c**2

    # Standard hydrostatic equilibrium: pressure gradient balances gravity.
    dpdr = -G * rho * M / r**2
    # Mass continuity equation.
    dMdr = 4.0 * np.pi * r**2 * rho
    return [dpdr, dMdr]


def tov_equations(r, y):
    """
    Defines the Tolman-Oppenheimer-Volkoff (TOV) equations for GR.

    Args:
        r (float): Radius in cm.
        y (list): State vector [pressure, enclosed_mass].

    Returns:
        list: Derivatives [dP/dr, dM/dr].
    """
    p, M = y
    if p <= 0.0:
        return [0.0, 0.0]

    eps = eps_phys(p)
    rho = eps / c**2

    # These factors represent the corrections from General Relativity.
    # If all factors were 1, this would reduce to the Newtonian case.
    num = G * rho * M / r**2
    fac1 = 1.0 + p / eps                      # Special relativity correction
    fac2 = 1.0 + 4.0 * np.pi * r**3 * p / (M * c**2)  # Curvature correction
    fac3 = 1.0 - 2.0 * G * M / (c**2 * r)     # Redshift correction

    # The full GR pressure gradient.
    dpdr = -num * fac1 * fac2 / fac3
    # Mass continuity is the same, but rho is mass-energy density.
    dMdr = 4.0 * np.pi * r**2 * rho
    return [dpdr, dMdr]


# --- Integration Setup ---
# We will model one star with a fixed central pressure.
p_c_bar = 0.01                 # Dimensionless central pressure
p_c = p_c_bar * eps0           # Physical central pressure (erg cm^-3)

r_start = 1.0e-2               # Small initial radius to avoid r=0 singularity
eps_c = eps_phys(p_c)
rho_c = eps_c / c**2
M_start = (4.0/3.0) * np.pi * r_start**3 * rho_c  # Initial mass in the core

# Initial conditions for the solver [pressure, mass].
y0 = [p_c, M_start]
r_max = 2.0e7                  # Max integration radius (200 km)


# --- Surface Detection Event ---
def surface(r, y):
    """Event function to stop integration when the surface is reached."""
    # The surface is where pressure drops to near zero.
    return y[0] - 1.0e-8 * eps0
surface.terminal, surface.direction = True, -1  # Stop when event is found


# --- Solve for Both Models ---
print(f"Integrating for a star with central pressure p_c/eps0 = {p_c_bar:.3g}")
sol_newt = solve_ivp(
    newtonian_equations, (r_start, r_max), y0,
    events=surface, rtol=1e-8, atol=1e-10, max_step=5.0e4
)
sol_tov = solve_ivp(
    tov_equations, (r_start, r_max), y0,
    events=surface, rtol=1e-8, atol=1e-10, max_step=5.0e4
)


# --- Extract Final Radius and Mass ---
def extract_surface_properties(sol):
    """Helper function to get R and M from a solution object."""
    if sol.t_events[0].size > 0:
        R_cm = sol.t_events[0][0]
        M_g = sol.y_events[0][0][1]
    else:  # Fallback if the surface wasn't found
        R_cm = sol.t[-1]
        M_g = sol.y[1, -1]
    return R_cm, M_g

R_new_cm, M_new_g = extract_surface_properties(sol_newt)
R_tov_cm, M_tov_g = extract_surface_properties(sol_tov)

# Convert to standard astrophysical units for comparison.
R_new_km = R_new_cm * CM_PER_KM
R_tov_km = R_tov_cm * CM_PER_KM
M_new_Msun = M_new_g / M_sun
M_tov_Msun = M_tov_g / M_sun

# --- Print Results ---
print("\n--- Comparison of Stellar Properties ---")
print(f"Newtonian:  R = {R_new_km:6.3f} km   M = {M_new_Msun:6.3f} M_sun")
print(f"TOV (GR) :  R = {R_tov_km:6.3f} km   M = {M_tov_Msun:6.3f} M_sun")
print("----------------------------------------")


# --- Plotting the Profiles ---
plt.style.use('seaborn-v0_8-notebook')
plt.figure(figsize=(12, 6))

# Plot 1: Pressure Profile
plt.subplot(1, 2, 1)
plt.plot(sol_newt.t * CM_PER_KM, sol_newt.y[0] / eps0, 'b-', label='Newtonian')
plt.plot(sol_tov.t * CM_PER_KM, sol_tov.y[0] / eps0, 'r--', label='TOV (GR)')
plt.xlabel('Radius $r$ (km)', fontsize=14)
plt.ylabel(r'Dimensionless Pressure $p(r) / \epsilon_0$', fontsize=14)
plt.title('Pressure Profile Comparison', fontsize=16)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Plot 2: Mass Profile
plt.subplot(1, 2, 2)
plt.plot(sol_newt.t * CM_PER_KM, sol_newt.y[1] / M_sun, 'b-', label='Newtonian')
plt.plot(sol_tov.t * CM_PER_KM, sol_tov.y[1] / M_sun, 'r--', label='TOV (GR)')
plt.xlabel('Radius $r$ (km)', fontsize=14)
plt.ylabel(r'Enclosed Mass $M(r)\,[M_\odot]$', fontsize=14)
plt.title('Enclosed Mass Profile', fontsize=16)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('newt_vs_tov_profiles.png', dpi=300, bbox_inches='tight')
plt.show()