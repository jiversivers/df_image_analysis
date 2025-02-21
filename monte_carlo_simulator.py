import itertools
import sqlite3

import my_modules.monte_carlo as mc
from tqdm import tqdm

try:
    import cupy as np

    if not np.is_available():
        raise RuntimeError("CUDA not available; reverting to NumPy.")
except (ImportError, RuntimeError) as e:
    import numpy as np

    np.is_available = lambda: False  # Mock the `is_available` method for consistency

print(f"Using {np.__name__}")

def main():
    # Init parameter sets
    mu_s_array = np.linspace(0, 100, 40)
    mu_a_array = np.linspace(1, 101, 40)
    g_array = [0.85]
    d_array = [float('inf')]  # Fix as semi-infinite
    n = 50000
    tissue_n = 1.33
    surroundings_n = 1

    # Make water medium
    di_water = mc.OpticalMedium(n=1, mu_s=0.003, mu_a=0, g=0, type='di water')

    # Simulate
    conn = sqlite3.connect('databases/hsdfm_data.db')
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mu_s REAL,
        mu_a REAL,
        g REAL,
        depth REAL,
        transmission REAL,
        reflectance REAL,
        absorption REAL,
        simulation_id INTEGER
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut_simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photon_count INTEGER NOT NULL,
    dimensionality INTEGER NOT NULL,
    water_n REAL NOT NULL,
    water_mu_s REAL NOT NULL,
    water_mu_a REAL NOT NULL,
    tissue_n REAL NOT NULL,
    surroundings_n REAL NOT NULL)""")

    c.execute(f"""
    INSERT INTO mclut_simulations (
    photon_count, dimensionality, water_n, water_mu_s, water_mu_a, tissue_n, surroundings_n
    ) VALUES (?, ?, ?, ?, ?, ?, ?)""", (
        n, 1, di_water.n, di_water.mu_s, di_water.mu_a, tissue_n, surroundings_n
    ))
    conn.commit()
    simulation_id = c.lastrowid

    # Set the total number of iterations for each loop level
    for (mu_s, mu_a, g, d) in tqdm(itertools.product(mu_s_array, mu_a_array, g_array, d_array), desc="Processing",
                                   total=len(mu_s_array) * len(mu_a_array) * len(g_array) * len(d_array)):
        # Make the system
        tissue = mc.OpticalMedium(n=tissue_n, mu_s=mu_s, mu_a=mu_a, g=g, type='tissue')
        system = mc.System(di_water, 0.01, tissue, d, surrounding_n=surroundings_n)
        T, R, A, = mc.simulate(system, n=10000)
        # Add results to db
        c.execute(f"""
                    INSERT INTO mclut (
                    mu_s, mu_a, g, depth, transmission, reflectance, absorption, simulation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
            mu_s, mu_a, g, d, T / n, R / n, A / n, simulation_id
        ))
    conn.commit()

if __name__ == '__main__':
    main()
