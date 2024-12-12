import my_modules.monte_carlo.monte_carlo as mc

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
    mu_s_array = np.arange(40, 140, 5)
    mu_a_array = np.arange(0, 100, 5)
    g_array = np.arange(0, 1, 0.1)
    d_array = np.arange(0, 1, 0.05)

    # Make water medium
    di_water = mc.OpticalMedium(n=1, mu_s=0.003, mu_a=0, g=0, type='di water')

    for mu_s in mu_s_array:
        for mu_a in mu_a_array:
            for g in g_array:
                for d in d_array:
                    # Make the system
                    tissue = mc.OpticalMedium(n=1.4, mu_s=mu_s, mu_a=mu_a, g=g, type='tissue')
                    system = mc.System(di_water, 0.01, tissue, d, surrounding_n=1)
                    T, R, A = 3 * [0]
                    n = 50000
                    for i in range(n):
                        photon = mc.Photon(650, system=system)
                        while not photon.is_terminated:
                            photon.absorb()
                            photon.move()
                            photon.scatter()
                        T += photon.T
                        R += photon.R
                        A += photon.A
                    simulation_parameters = (mu_s, mu_a, g, d)
                    simulation_results = (T / n, R / n, A / n)
                    mc.insert_into_mclut_database(simulation_parameters, simulation_results, db_file='1d_mclut_w50000_photons.db')

if __name__ == '__main__':
    main()
