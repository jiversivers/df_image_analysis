import my_modules.monte_carlo.monte_carlo as mc


def main():
    # make the system
    di_water = mc.OpticalMedium(1.333, 0, 0.0008, 0, type='di water')
    tissue = mc.OpticalMedium(1.55, 90, 10, 0.75, type='tissue')
    system = mc.System(di_water, 0.1, tissue, 100)
    T, R, A = 3 * [0]
    paths = []
    j = 0
    for i in range(100):
        photon = mc.Photon(650, system=system)
        while not photon.is_terminated:
            photon.move()
            if photon.is_terminated:
                break
            photon.absorb()
            if photon.is_terminated:
                break
            photon.scatter()
            j += 1
        paths.append(photon.location_history)
        T += photon.T
        R += photon.R
        A += photon.A
    print(T, R, A)

if __name__ == '__main__':
    main()

