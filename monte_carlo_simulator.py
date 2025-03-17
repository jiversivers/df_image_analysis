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
    mu_s_array = np.arange(0, 101, 5)
    mu_a_array = np.arange(1, 102, 5)
    g_array = [0.9]
    d_array = [float('inf')]  # Fix as semi-infinite
    n = 50000
    tissue_n = 1.33
    surroundings_n = 1
    recurse = True

    # Make water medium
    di_water = mc.OpticalMedium(n=1.33, mu_s=0, mu_a=0, g=0, name='di water')
    glass = mc.OpticalMedium(n=1.523, mu_s=0, mu_a=0, g=0, name='glass')

    # Simulate
    conn = sqlite3.connect('databases/hsdfm_data.db')
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut (
        inner INTEGER PRIMARY KEY AUTOINCREMENT,
        mu_s REAL,
        mu_a REAL,
        g REAL,
        depth REAL,
        transmission REAL,
        reflectance REAL,
        absorption REAL,
        simulation_id INTEGER,
        forced BOOLEAN DEFAULT FALSE
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS mclut_simulations (
    inner INTEGER PRIMARY KEY AUTOINCREMENT,
    photon_count INTEGER NOT NULL,
    dimensionality INTEGER NOT NULL,
    water_n REAL NOT NULL,
    water_mu_s REAL NOT NULL,
    water_mu_a REAL NOT NULL,
    tissue_n REAL NOT NULL,
    surroundings_n REAL NOT NULL,
    recursive BOOLEAN DEFAULT FALSE,
    cover_glass BOOLEAN DEFAULT FALSE)""")

    c.execute(f"""
    INSERT INTO mclut_simulations (
    photon_count, dimensionality, water_n, water_mu_s, water_mu_a, tissue_n, surroundings_n, recursive, cover_glass
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
        n, 1, float(di_water.n), float(di_water.mu_s), float(di_water.mu_a), float(tissue_n), float(surroundings_n),
        recurse, True if 'glass' in locals() else False
    ))
    conn.commit()
    simulation_id = c.lastrowid

    # Set the total number of iterations for each loop level
    for (mu_s, mu_a, g, d) in tqdm(itertools.product(mu_s_array, mu_a_array, g_array, d_array), desc="Processing",
                                   total=len(mu_s_array) * len(mu_a_array) * len(g_array) * len(d_array)):
        # Make the system
        tissue = mc.OpticalMedium(n=tissue_n, mu_s=mu_s, mu_a=mu_a, g=g, name='tissue')
        system = mc.System(di_water, 0.1,  # 1mm
                           glass, 0.017,  # 0.17mm
                           tissue, d,
                           surrounding_n=surroundings_n)
        T, R, A = 3 * [0]

        for i in range(n):
            photon = mc.Photon(650, system=system, recurse=recurse)
            photon.simulate()
            T += photon.T
            R += photon.R
            A += photon.A

        # Add results to db
        c.execute(f"""
                    INSERT INTO mclut (
                    mu_s, mu_a, g, depth, transmission, reflectance, absorption, simulation_id, forced
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, false)""", (
            float(mu_s), float(mu_a), float(g), float(d), float(T / n), float(R / n), float(A / n), simulation_id
        ))
    conn.commit()


if __name__ == '__main__':
    main()
