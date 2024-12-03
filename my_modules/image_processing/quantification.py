import numpy as np


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

def monte_carlo_lut(mus, mua,
                    rho=2.25, n_tissue=1.4, n_collection=1):
    pass


def get_optical_properties(R_exp, method='monte_carlo_lut',
                           wavelength=(400, 500, 600, 700), wavelength_0=650,
                           hb=(0, 1), hbo2=(2, 3),
                           rho=2.25, n_tissue=1.4, n_collection=1):
    match method:
        case 'monte_carlo_lut':
            f = monte_carlo_lut
        case 'diffusion_approximation':
            f = diffusion_approximation
        case _:
            raise ValueError(f'Method {method} unrecognized/not supported')

    # Make (bounded) random guesses for the parameters
    # initial_parameters = # TODO: This should return parameters a, b, chb, alpha with appropriate/inputable bounds

    # Stop criteria
    i = 1
    loss = 1000

    # Loop until fit or 5000
    while loss > 1e-3 and i < 5000:
        parameters = intitial_parameters if i == 1 else parameters
        # TODO: Ensure that the dimensionality for each array is matched. mu_s, mu_a, and R should be ndarrays with
        #  elements for each wavelength, and loss should be a single value. Determine modelled reflectance from
        #  parameter guesses
        mu_s, mu_a = calculate_mus(*parameters,
                                   wavelength=wavelength, wavelength_0=wavelength_0, hb=hb, hbo2=hbo2)
        R = f(mu_s, mu_a, rho=rho, n_tissue=n_tissue, n_collection=n_collection)

        # Sum of squares error loss
        loss = np.sum((R - R_exp) ** 2)

        # Update counter
        i += 1

        # TODO: Find interior-point non-linear optimizer to update parameter guesses
        # parameters =

    return mu_s, mu_a

def reduced_chi_squared(o, e, n, p):
    return np.sum(((o - e) ** 2) / (n - p))
