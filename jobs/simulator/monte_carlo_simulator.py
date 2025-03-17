import itertools
import sqlite3
import time
import warnings

import my_modules.monte_carlo as mc
from my_modules.monte_carlo.hardware import ID, OD, THETA
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
    mu_s_array = np.arange(0, 51, 1)
    mu_a_array = np.arange(1, 51, 1)
    g_array = [0.8, 0.9]
    d_array = [0.1, float('inf')]
    n = 50000
    tissue_n = 1.33
    surroundings_n = 1.33
    recurse = True

    # Make fixed mediums
    di_water = mc.OpticalMedium(n=1.33, mu_a=0, mu_s=0, g=0, name='water')
    glass = mc.OpticalMedium(n=1.523, mu_a=0, mu_s=0, g=0, name='glass')
    surroundings_n = 1.33

    # Setup illumination
    sampler = mc.monte_carlo.ring_pattern((ID, OD), THETA)
    led = mc.Illumination(pattern=sampler)
    detector = mc.Detector(mc.monte_carlo.cone_of_acceptance(ID))

    # Simulate
    conn = sqlite3.connect('hsdfm_data.db')
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
        simulation_id INTEGER,
        forced BOOLEAN DEFAULT FALSE
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

    try:
        # Set the total number of iterations for each loop level
        start_time = time.time()
        for i, (mu_s, mu_a, g, d) in enumerate(itertools.product(mu_s_array, mu_a_array, g_array, d_array)):
            print(f'Simulating mu_s={mu_s}, mu_a={mu_a}, g={g}, d={d}')

            # Make the system
            tissue = mc.OpticalMedium(n=tissue_n, mu_s=mu_s, mu_a=mu_a, g=g, name='tissue')
            system = mc.System(
                di_water, 0.2,  # 2mm
                glass, 0.017,  # 0.17mm
                tissue, d,
                surrounding_n=surroundings_n,
                illuminator=led,
                detector=(detector, 0)
            )

            T, R, A = 3 * [0]
            detector.reset()
            for i in range(n):
                photon = system.beam(recurse=True,
                                     russian_roulette_constant=20,
                                     tir_limit=10,
                                     recursion_limit=100,
                                     throw_recursion_error=False)
                photon.simulate()
                T += photon.T
                R += photon.R
                A += photon.A

            detected_fraction = detector.n_detected / (n - (R - detector.n_detected))

            # Add results to db
            c.execute(f"""
                        INSERT INTO mclut (
                        mu_s, mu_a, g, depth, transmission, reflectance, absorption, simulation_id, forced
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, false)""", (
                float(mu_s), float(mu_a), float(g), float(d), float(T / n), float(detected_fraction), float(A / n), simulation_id
            ))
            conn.commit()

            total_time = time.time() - start_time
            total_m, total_s = divmod(total_time, 60)
            time_per_it = total_time / i
            est_time_rem = (len(mu_s_array) * len(mu_a_array) * len(g_array) * len(d_array) - i) * time_per_it
            rem_m, rem_s = divmod(est_time_rem, 60)
            print(f'Time -- elapsed: {total_m}:{total_s:0.f} | {time_per_it}s/it | Est. {rem_m}:{rem_s:0.f} remaining')

    except Exception as e:
        warnings.warn(e)
        conn.commit()
    finally:
        conn.close()

if __name__ == '__main__':
    main()
