"""
Solves the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron stars
using a simplified square-root nuclear Equation of State (EoS).

This EoS model, where energy density is proportional to the square root of
pressure, represents a basic approximation for strong nucleon-nucleon interactions.
The script generates and plots the mass-radius curve for this model.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy import constants as const

# --- Physical Constants (CGS units) ---
G = const.G.cgs.value        # Gravitational constant (cm^3/g/s^2)
c = const.c.cgs.value        # Speed of light (cm/s)
M_sun = const.M_sun.cgs.value  # Solar mass (g)
CM_PER_KM = 1e-5             # Conversion factor from cm to km

# --- Equation of State (EoS) Parameters ---
# This is a simplified model representing nucleon-nucleon interactions.
# The energy density is given by: eps(p) = A0 * sqrt(p * eps0)
eps0 = 5.346e36  # Characteristic energy density scale (erg/cm^3)
A0 = 0.8642      # Dimensionless coefficient for the EoS model

def eps_phys(p_phys):
    """Calculates physical energy density from pressure using the square-root EoS."""
    # Ensure pressure is non-negative to avoid domain errors with sqrt.
    if p_phys < 0:
        return 0.0
    return A0 * np.sqrt(eps0 * p_phys)

def tov_equations(r, y):
    """
    Defines the right-hand side of the TOV equations for integration.

    Args:
        r (float): Current radius in cm.
        y (list): State vector [pressure (erg/cm^3), enclosed_mass (g)].

    Returns:
        list: Derivatives [dP/dr, dM/dr].
    """
    p, M = y

    # Stop integration if pressure becomes non-physical (i.e., outside the star).
    if p <= 0.0:
        return [0.0, 0.0]

    # Determine local properties from the EoS.
    eps = eps_phys(p)
    rho = eps / c**2

    # --- Relativistic corrections for hydrostatic equilibrium ---
    # These factors modify the Newtonian gravity equation for GR.
    num = G * eps * M / (c**2 * r**2)
    fac1 = 1.0 + p / eps                      # Special relativity correction
    fac2 = 1.0 + 4.0 * np.pi * r**3 * p / (M * c**2)  # Pressure as a source of gravity
    fac3 = 1.0 - 2.0 * G * M / (c**2 * r)     # Spacetime curvature (redshift)

    # Full pressure gradient and mass continuity equation.
    dpdr = -num * fac1 * fac2 / fac3
    dMdr = 4.0 * np.pi * r**2 * rho

    return [dpdr, dMdr]

# --- Surface Detection Event ---
def surface(r, y):
    """Event function to stop the integration when the stellar surface is reached."""
    # The surface is defined as the point where pressure drops to a negligible value.
    return y[0] - 1e-8 * eps0
surface.terminal = True  # Stop integration when the event is triggered.
surface.direction = -1   # Trigger only when pressure is decreasing.

def integrate_star(p_c_bar, r_start=1e-2, r_max=5e6):
    """
    Integrates the TOV equations for a single star with a given central pressure.

    Args:
        p_c_bar (float): Dimensionless central pressure (p_c / eps0).
        r_start (float): Small initial radius to avoid the singularity at r=0.
        r_max (float): Maximum radius to integrate outwards.

    Returns:
        tuple: (Radius in km, Mass in solar masses).
    """
    # --- Initial Conditions ---
    p_c = p_c_bar * eps0  # Physical central pressure
    eps_c = eps_phys(p_c)
    rho_c = eps_c / c**2
    M0 = (4.0/3.0) * np.pi * r_start**3 * rho_c  # Mass inside the initial small core

    # --- Solve the ODE System ---
    sol = solve_ivp(
        tov_equations, (r_start, r_max), [p_c, M0],
        events=surface, rtol=1e-8, atol=1e-10, max_step=1e5
    )

    # --- Extract Surface Properties ---
    if sol.t_events[0].size > 0:  # If the surface event was triggered
        R_cm = sol.t_events[0][0]
        M_g = sol.y_events[0][0][1]
    else:  # Fallback if integration ends before surface is found
        R_cm = sol.t[-1]
        M_g = sol.y[1, -1]

    # Convert to standard astrophysical units.
    return R_cm * CM_PER_KM, M_g / M_sun

# --- Main Execution Block ---
if __name__ == "__main__":
    # To generate a mass-radius curve, we solve for a sequence of stars,
    # each defined by a different central pressure.
    p_c_bars = np.logspace(-4, 4, 200)
    R_arr, M_arr = [], []

    print("Computing TOV M-R curve (Square-Root Nuclear EoS)...")
    for i, p_bar in enumerate(p_c_bars):
        try:
            R_km, M_s = integrate_star(p_bar)
            R_arr.append(R_km)
            M_arr.append(M_s)
        except Exception:
            R_arr.append(np.nan)
            M_arr.append(np.nan)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{len(p_c_bars)}]  p_c/eps0 = {p_bar:.5g}")

    R_arr = np.array(R_arr)
    M_arr = np.array(M_arr)

    # --- Find and Report Maximum Mass ---
    # The peak of the M-R curve represents the maximum mass a non-rotating
    # star can support before collapsing into a black hole (the TOV limit).
    idx_max = np.nanargmax(M_arr)
    R_max = R_arr[idx_max]
    M_max = M_arr[idx_max]

    print("\n--- Maximum-Mass Star (Square-Root EoS) ---")
    print(f"  M_max = {M_max:.4f} M_sun")
    print(f"  Radius = {R_max:.4f} km")
    print("---------------------------------------------")

    # --- Plotting the Results ---
    plt.style.use('seaborn-v0_8-talk')
    plt.figure(figsize=(10, 8))
    plt.plot(R_arr, M_arr, 'b-', linewidth=2.5, label='Nuclear EoS')
    plt.plot(R_max, M_max, 'ro', markersize=8, label='Maximum Mass')
    plt.annotate(
        f'$M_{{\\rm max}} = {M_max:.2f}\\,M_\\odot$\n$R = {R_max:.2f}$ km',
        xy=(R_max, M_max), xytext=(R_max + 1, M_max - 0.5),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        fontsize=14
    )

    # Configure plot aesthetics.
    plt.xlabel('Radius $R$ (km)', fontsize=16)
    plt.ylabel(r'Mass $M$ ($M_{\odot}$)', fontsize=16)
    plt.title('Neutron Star M-R Relation (Square-Root Nuclear EoS)', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save figure with high resolution.
    plt.savefig('tov_mr_curve_nuclear.png', dpi=300, bbox_inches='tight')
    plt.show()