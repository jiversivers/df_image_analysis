import my_modules.monte_carlo.monte_carlo as mc
import numpy as np

from my_modules.image_processing.quantification import get_coefficients, calculate_mus


def main():
    # Init parameter sets
    sO2 = np.linspace(0, 1, num=100)  # %
    tHb = np.linspace(10, 20, num=100)  # mg/mL
    wavelengths = np.arange(400, 725, 5)  # nm
    G = np.linspace(0.5, 1, num=10)

    # Make water medium
    di_water = mc.OpticalMedium(n=1, mu_s=0.003, mu_a=0, g=0, type='di water')

    for so2 in sO2:
        for thb in tHb:
            # Determine coefficients at given parameter set
            wavelengths, hbo2, hb = get_coefficients(wavelengths, db_file='hbo2_hb.db')
            mu_s, mu_a = calculate_mus(a=1, b=1, chb=thb, alpha=so2, wavelength=wavelengths, hb=hb, hbo2=hbo2)
            for wavelength in wavelengths:
                for g in G:
                    for s, a in zip(mu_s, mu_a):
                        # Make the system
                        tissue = mc.OpticalMedium(n=1.4, mu_s=s, mu_a=a, g=g, type='tissue')
                        system = mc.System(di_water, 0.01, tissue, 0.05, surrounding_n=1)
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
                            simulation_parameters=(sO2, tHb, wavelength, g)
                            simulation_results = (T, R, A)

if __name__ == '__main__':
    main()

