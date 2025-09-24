"""
A comprehensive TOV solver that computes and compares the mass-radius
relations for neutron stars using three different Equations of State (EoS):

1.  A hybrid Fermi Gas EoS (non-relativistic + relativistic components).
2.  A simplified square-root Nuclear EoS (for strong interactions).
3.  The realistic, tabulated SLy EoS for n-p-e matter.

The script plots all three M-R curves on a single graph for direct comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from astropy import constants as const

# --- Physical Constants (CGS units) ---
G = const.G.cgs.value       # Gravitational constant
c = const.c.cgs.value       # Speed of light
M_sun = const.M_sun.cgs.value  # Solar mass
CM_PER_KM = 1e-5            # cm to km conversion

# Common energy density scale (nuclear saturation)
eps0 = 5.346e36  # erg/cm^3

# ============================================================================
# --- PART 1: Hybrid Fermi Gas EoS ---
# A model for degenerate matter, combining non-relativistic and relativistic limits.
# ============================================================================
A_NR = 2.4216    # Non-relativistic coefficient
A_R = 2.8663     # Relativistic coefficient

def eps_phys_hybrid(p_phys):
    """Physical energy density for the hybrid Fermi Gas EoS."""
    p_bar = p_phys / eps0
    eps_bar = A_NR * p_bar**(3/5) + A_R * p_bar
    return eps0 * eps_bar

def tov_hybrid(r, y):
    """TOV equations specifically for the hybrid EoS."""
    p, M = y
    if p <= 0: return [0.0, 0.0]
    eps = eps_phys_hybrid(p)
    rho = eps / c**2
    num = G * eps * M / (c**2 * r**2)
    fac1 = 1 + p / eps
    fac2 = 1 + 4 * np.pi * r**3 * p / (M * c**2)
    fac3 = 1 - 2 * G * M / (c**2 * r)
    dpdr = -num * fac1 * fac2 / fac3
    dMdr = 4 * np.pi * r**2 * rho
    return [dpdr, dMdr]

# ============================================================================
# --- PART 2: Square-Root Nuclear EoS ---
# A simplified model for strong nucleon-nucleon interactions.
# ============================================================================
A0_nuc = 0.8642  # Dimensionless coefficient

def eps_phys_nuclear(p_phys):
    """Physical energy density for the square-root nuclear EoS."""
    if p_phys < 0: return 0.0
    return A0_nuc * np.sqrt(eps0 * p_phys)

def tov_nuclear(r, y):
    """TOV equations specifically for the nuclear EoS."""
    p, M = y
    if p <= 0: return [0.0, 0.0]
    eps = eps_phys_nuclear(p)
    rho = eps / c**2
    num = G * eps * M / (c**2 * r**2)
    fac1 = 1 + p / eps
    fac2 = 1 + 4 * np.pi * r**3 * p / (M * c**2)
    fac3 = 1 - 2 * G * M / (c**2 * r)
    dpdr = -num * fac1 * fac2 / fac3
    dMdr = 4 * np.pi * r**2 * rho
    return [dpdr, dMdr]

# ============================================================================
# --- PART 3: Tabulated SLy EoS ---
# A realistic, computationally derived EoS for cold n-p-e matter.
# ============================================================================
def load_sly_eos(filename="SLy.txt"):
    """Loads the SLy EoS data and creates interpolation functions."""
    try:
        sly_data = np.genfromtxt(filename, delimiter="  ")
        mass_density = sly_data[:, 2]    # g/cm^3
        pressure = sly_data[:, 3]        # dyne/cm^2
        # Create smooth functions from the discrete data points.
        pressure_from_density = CubicSpline(mass_density, pressure)
        density_from_pressure = CubicSpline(pressure, mass_density)
        print("Successfully loaded SLy EoS data from file.")
        return pressure_from_density, density_from_pressure
    except IOError:
        print(f"Warning: SLy EoS file '{filename}' not found. SLy model will be skipped.")
        return None, None

def tov_sly(r, y, density_from_pressure):
    """
    TOV equations for the SLy EoS.
    Note: The state vector y for this function is [Mass, Pressure],
    which is different from the other EoS models.
    """
    M, p = y
    if p <= 0: return [0.0, 0.0]
    rho = density_from_pressure(p)
    eps = rho * c**2
    num = G * M * rho / r**2
    fac1 = 1 + p / eps
    fac2 = 1 + 4 * np.pi * r**3 * p / (M * c**2)
    fac3 = 1 - 2 * G * M / (r * c**2)
    dMdr = 4 * np.pi * r**2 * rho
    dPdr = -num * fac1 * fac2 / fac3
    return [dMdr, dPdr]

# ============================================================================
# --- Common Integration Utilities ---
# ============================================================================
def surface_event(r, y):
    """Event function to find the stellar surface (where pressure -> 0)."""
    # This event applies to models where pressure is the first element of y.
    return y[0] - 1e-8 * eps0
surface_event.terminal = True
surface_event.direction = -1

def surface_event_sly(r, y):
    """Event function for SLy, where pressure is the second element of y."""
    return y[1] - 1e-10 * 1e35
surface_event_sly.terminal = True
surface_event_sly.direction = -1

def solve_for_star(p_c, tov_func, is_sly=False, sly_interpolator=None):
    """
    A generic function to integrate the TOV equations for any given EoS model.
    It handles the different state vector conventions between models.
    """
    r_start, r_max = 1e-2, 5e6

    # The SLy model requires special handling due to its state vector order [M, P].
    if is_sly:
        rho_c = sly_interpolator(p_c)
        M_start = (4.0/3.0) * np.pi * r_start**3 * rho_c
        y0 = [M_start, p_c]
        # Pass the interpolator as an argument to the solver.
        sol = solve_ivp(
            lambda r, y: tov_func(r, y, sly_interpolator), (r_start, r_max), y0,
            events=surface_event_sly, rtol=1e-7
        )
        # Extract results for [M, P] order.
        if sol.t_events[0].size:
            R_cm, M_g = sol.t_events[0][0], sol.y_events[0][0][0]
        else:
            R_cm, M_g = sol.t[-1], sol.y[0, -1]
    else:
        # For other models, the state vector is [P, M].
        eps_c = tov_func.eos_lookup(p_c) # Assumes a helper is attached to the function
        rho_c = eps_c / c**2
        M_start = (4.0/3.0) * np.pi * r_start**3 * rho_c
        y0 = [p_c, M_start]
        sol = solve_ivp(tov_func, (r_start, r_max), y0, events=surface_event, rtol=1e-7)
        # Extract results for [P, M] order.
        if sol.t_events[0].size:
            R_cm, M_g = sol.t_events[0][0], sol.y_events[0][0][1]
        else:
            R_cm, M_g = sol.t[-1], sol.y[1, -1]

    return R_cm * CM_PER_KM, M_g / M_sun

def calculate_mr_curve(eos_name, p_c_range, tov_func, **kwargs):
    """Calculates a full mass-radius curve for a given EoS."""
    print(f"--- Calculating M-R curve for {eos_name} EoS ---")
    R_arr, M_arr = [], []
    for i, p_c in enumerate(p_c_range):
        try:
            R, M = solve_for_star(p_c, tov_func, **kwargs)
            R_arr.append(R)
            M_arr.append(M)
        except Exception:
            R_arr.append(np.nan)
            M_arr.append(np.nan)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(p_c_range)} stars completed.")
    return np.array(R_arr), np.array(M_arr)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Attach EoS lookup functions to their respective TOV solvers for generic access.
    tov_hybrid.eos_lookup = eps_phys_hybrid
    tov_nuclear.eos_lookup = eps_phys_nuclear

    # --- Calculate for Hybrid EoS ---
    p_c_hybrid = np.logspace(-4, 4, 250) * eps0
    R_hybrid, M_hybrid = calculate_mr_curve("Hybrid Fermi Gas", p_c_hybrid, tov_hybrid)

    # --- Calculate for Nuclear EoS ---
    p_c_nuclear = np.logspace(-4, 4, 250) * eps0
    R_nuclear, M_nuclear = calculate_mr_curve("Square-Root Nuclear", p_c_nuclear, tov_nuclear)

    # --- Calculate for SLy EoS ---
    p_from_rho, rho_from_p = load_sly_eos()
    if rho_from_p:
        central_densities = np.logspace(np.log10(2.5e14), np.log10(5e15), 250)
        p_c_sly = p_from_rho(central_densities)
        R_sly, M_sly = calculate_mr_curve("SLy", p_c_sly, tov_sly, is_sly=True, sly_interpolator=rho_from_p)
    else:
        R_sly, M_sly = np.array([]), np.array([]) # Empty arrays if SLy fails to load

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each M-R curve.
    ax.plot(R_hybrid, M_hybrid, 'r-', linewidth=2.5, label='Hybrid Fermi Gas EoS')
    ax.plot(R_nuclear, M_nuclear, 'b-', linewidth=2.5, label='Square-Root Nuclear EoS')
    if R_sly.size > 0:
        ax.plot(R_sly, M_sly, 'g-', linewidth=2.5, label='Realistic SLy EoS')

    # Find and plot maximum mass points.
    models = {"Hybrid": (R_hybrid, M_hybrid), "Nuclear": (R_nuclear, M_nuclear), "SLy": (R_sly, M_sly)}
    colors = {"Hybrid": "red", "Nuclear": "blue", "SLy": "green"}
    for name, (R, M) in models.items():
        if len(M) > 0 and np.any(~np.isnan(M)):
            idx_max = np.nanargmax(M)
            ax.plot(R[idx_max], M[idx_max], 'o', color=colors[name], markersize=10, markeredgecolor='black')

    # Configure plot aesthetics.
    ax.set_xlabel('Radius $R$ (km)', fontsize=16)
    ax.set_ylabel(r'Mass $M$ ($M_{\odot}$)', fontsize=16)
    ax.set_title('Comparison of Neutron Star Mass-Radius Relations', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=14)
    ax.set_xlim(8, 22)
    ax.set_ylim(0, 3.0)
    plt.tight_layout()

    # Save the figure.
    plt.savefig('tov_mr_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()