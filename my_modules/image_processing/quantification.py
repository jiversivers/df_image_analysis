import sqlite3
import csv
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
                           wavelength=None, wavelength_0=650,
                           hb=None, hbo2=None,
                           rho=2.25, n_tissue=1.4, n_collection=1):
    if wavelength is None:
        wavelength = get_default_values('lambda')
        hb = get_default_values('hb')
        hbo2 = get_default_values('hbo2')
    elif wavelength is not None:
        wavelength, hbo2, hb = get_coefficients(wavelength) if hbo2 is None and hb is None else wavelength, hbo2, hb

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


# Load a tsv file of molar extinction coefficients into the project database. This will overwrite to allow updating.
# Just update the TSV file and run this function.
def load_extinction_database(data_file='hbo2_hb.tsv', db_file='hbo2_hb.db'):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS molar_extinction_data")

    # Create the table
    cursor.execute("""
    CREATE TABLE molar_extinction_data (
        "lambda" REAL,
        "hbo2" REAL,
        "hb" REAL
    )
    """)

    # Open the TSV file and read data
    with open(data_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")

        # Skip introductory and units rows
        next(reader)  # Ignore intro line
        header = next(reader)  # Read the header row
        next(reader)  # Ignore units row

        # Ensure the header matches the database table
        if header != ["lambda", "hbo2", "hb"]:
            raise ValueError(f"Header in TSV file does not match expected columns: {header}")

        # Print rows from reader to debug
        rows = [row for row in reader]

        # Insert data into the database
        cursor.executemany(
            "INSERT INTO molar_extinction_data (\"lambda\", \"hbo2\", \"hb\") VALUES (?, ?, ?)",
            rows
        )

    # Commit changes and close the connection
    conn.commit()
    conn.close()


# Returns all column data from the SQL db. Useful for setting defaults in optical properties function.
def get_default_values(column, db_file='hbo2_hb.db'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(f"SELECT {column} FROM molar_extinction_data")
    data = cursor.fetchall()
    conn.close()

    return [item[0] for item in data]

# Returns tuple of lambdas with associated extinction coefficients for given lambdas from the db
def get_coefficients(lambda_values, db_file='hbo2_hb.db'):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # If lambda_values is a single value, make it a list for uniform processing
    if isinstance(lambda_values, (int, float)):
        lambda_values = [lambda_values]

    # Prepare the query to fetch hb and hbo2 for the given lambda values
    # We'll use placeholders for the lambda values in the query
    query = f"SELECT lambda, hbo2, hb FROM molar_extinction_data WHERE lambda IN ({','.join(['?'] * len(lambda_values))})"

    # Execute the query and fetch results
    cursor.execute(query, lambda_values)
    data = cursor.fetchall()

    # Close the connection
    conn.close()

    # Return the results as a list of tuples (lambda, hbo2, hb)
    return data
