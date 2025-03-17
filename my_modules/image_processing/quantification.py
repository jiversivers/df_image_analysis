import sqlite3
import csv

try:
    import cupy as np

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np


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
## TODO: Add linear interpolation for cases when exact wavelength match is not present in .db
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
