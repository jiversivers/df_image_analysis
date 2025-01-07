import numpy as np


class Model:
    def __init__(self, function):
        self.function = function

    def fit(self, initials, method, threshold, max_iter):
        pass


# TODO: Update default kwargs in calculate_mus to hold or load actual information for Hb.
def calculate_mus(a, b, chb, alpha,
                  wavelength=(400, 500, 600, 700), wavelength_0=650,
                  hb=(0, 1), hbo2=(2, 3)):
    wavelength = np.asarray(wavelength)  # wavelengths of measurements
    hb = np.asarray(hb)  # extinction coefficient for deoxygenated hemoglobin
    hbo2 = np.asarray(hbo2)  # extinction coefficient for oxygenated hemoglobin

    chb = 0 if chb < 0 else chb
    mu_s = (a * 10) * (wavelength / wavelength_0) ** -b  # Reduced scattering coefficient, cm^-1
    mu_a = 2.303 * (chb * (alpha * hbo2 + (1 - alpha) * hb))  # Absorption coefficient, cm^-1
    return mu_s, mu_a


# TODO: Update default kwargs in diffusion approximation and determine how to handle RHO for multiple and non-normal
#  beams.
def diffusion_approximation(mu_s, mu_a,
                            rho=2.25, n_tissue=1.4, n_collection=1):
    # Optical properties (and derivatives)
    mu_t = mu_a + mu_s  # Total interaction coefficient, cm^-1
    mu_eff = np.sqrt(3 * mu_a * mu_t)  # In text (intro while discussing Patterson et al.)
    albedo = mu_s / mu_t  # In text (intro while discussing Patterson et al.)
    D = 1 / (3 * mu_t)  # Diffusion constant; Eqn. 3

    # System Properties
    n_rel = n_tissue / n_collection  # Relative refractive index; in text (preceding eqn. 7)
    r_d = 1.440 * (n_rel ** (-2)) + 0.710 * (n_rel ** (-1)) + 0.668 + 0.0636 * n_rel  # Eqn. 9
    A = (1 + r_d) / (1 - r_d)  # Eqn. 8

    z_b = 2 * A * D  # In text (fluence zero point; boundary condition explanation following Eqn. 9)
    z = 0  # In text (following solution explanation of eqn. 15)
    z_0 = 1 / mu_t  # In text (preamble to Eqn. 18)

    r1 = np.sqrt(((z - z_0) ** 2 + rho ** 2))  # Green's function for fluence parameter; Eqn. 11
    r2 = np.sqrt(((z + z_0 + 2 * z_b) ** 2 + rho ** 2))  # Green's function solved for a point source; Eqn. 13

    # Green's Function for Diffuse Reflectance; Eqn. 15
    R = (albedo / (4 * np.pi)) * (z_0 * (mu_eff + (1 / r1)) * (np.exp(-mu_eff * r1) / (r1 ** 2)) +
                                  ((z_0 + 2 * z_b) * (mu_eff + (1 / r2)) * (np.exp(-mu_eff * r2) / (r2 ** 2))))

    return R


# TODO: Add LUT load and get value and update get_optical_properties to use it. Make it match the same inputs and
#  returns style of diffusion_approximation

def mclut(mus, mua, rho=2.25, n_tissue=1.4, n_collection=1):
    pass


# TR = p50 from Boltzmann Function
# R is the average rate-limited distance from center of capillaries (dist when NADH int = 50% max)
# r is capillary radius measured from the HSDF image vascular map
# T0 is the oxygen tension within the capillary, as measured from the mclut or diffusion approximation
def krogh_cylinder(T0, TR, R, r, x):
    p_K = (T0 - TR) / ((1 / 2) * (R ** 2) * np.log(R / r) - ((R ** 2 - r ** 2) / 4))
    Tx = T0 - p_K * ((1 / 2) * (R ** 2) * np.log(R / r) - ((R ** 2 - r ** 2) / 4))
    return Tx


# po2 is the oxygen tension within the capillary, as measured from the mclut or diffusion approximation
def boltzmann_function(a1, a2, pO2, p50, dx):
    nadph_per = a2 + ((a1 + a2) / (1 + np.exp((pO2 - p50) ** dx)))
    return nadph_per


