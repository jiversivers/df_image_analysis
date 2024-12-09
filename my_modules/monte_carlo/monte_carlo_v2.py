import inspect

import numpy as np


class System:
    def __init__(self, *args):
        """
        Args comes in groups of 2:
            0: the medium
            1: the thickness
        The blocks are constructed top down in the order they are received and with a semi-infinite approximation; i.e.
        the top block starts at inf (typically air or water).
        """
        assert len(args) % 2 == 0, "Arguments must be in groups of 2: medium object and thickness."

        # Stack layers
        interface = 0
        self.stack = {}
        self.layer = []
        self.boundaries = []
        for i in range(0, len(args), 2):
            self.stack[(interface, interface + args[i + 1])] = args[i]
            self.layer.append(args[i])
            self.boundaries.append(interface)
            interface += args[i + 1]

        # Last layer finish
        self.boundaries.append(interface)

    def in_medium(self, location):
        z = location[2] if isinstance(location, (tuple, list, np.ndarray)) else location
        if z < self.boundaries[0] or z > self.boundaries[-1]:
            return False
        elif z == self.boundaries[0]:
            in_ = [[], self.layer[0]]
        elif z == self.boundaries[-1]:
            in_ = [self.layer[-1], []]
        else:
            in_ = []
            for boundaries, media in self.stack.items():
                if z == boundaries[0]:
                    in_.append(media)
                elif z == boundaries[1]:
                    in_ = [media]
                elif boundaries[0] < z < boundaries[1]:
                    in_ = media
        return in_

    def interface_crossed(self, location1, location2):
        # Get z coords
        z = []
        for location in [location1, location2]:
            z.append(location[2] if isinstance(location, (tuple, list, np.ndarray)) else location)

        # Get direction of vector
        z_dir = np.sign(z[1] - z[0])

        # Get media at zs
        interface = [self.in_medium(z[0]), self.in_medium(z[1])]

        # Deal with edge cases for individual coordinates
        exited = False
        for i, medium in enumerate(interface):
            # When z is at an interface, find which medium is on the side of the interface of the vector (the one that
            # is fully passed through, not the one that is just being entered/exited)
            if isinstance(medium, (list, tuple)):
                # Bumps the coords towards the other z.
                # When i = 0, coords are bumped forward in the direction of z_dir (toward z[1])
                # When i = 1, coords are bumped backwards of the direction of z_dir (toward z[0])
                bumped_coords = np.nextafter(z[i], z[1 - i])

                # Find the medium at the bumped coords to find which medium is actually passed through
                interface[i] = self.in_medium(bumped_coords)

            # When the location is outside the system, find the medium at the nearest edge of the system
            elif not medium:
                if z[i] < self.boundaries[0]:
                    interface[i] = self.layer[0]
                    plane = self.boundaries[0]
                else:
                    interface[i] = self.layer[-1]
                    plane = self.boundaries[-1]
                exited = exited or not (self.boundaries[0] < z[i] < self.boundaries[-1])

        # Now we are ready to determine if the path actually crosses an interface:
        # First handle no boundary crossing
        # (there is an edge case possible where two boundaries are crossed from one medium back into the same medium...
        # this is mitigated by building the system without copying media)
        if interface[0] == interface[1] and not exited:
            return False, []
        elif interface[0] == interface[1]:
            return (self.in_medium(z[0]), self.in_medium(z[1])), plane
        else:
            # Sort the zs for more straight-forward comparisons
            ends = z
            ends.sort()

            # Determine which boundary(ies) would be crossed by which boundaries are between the ends
            crossed = [ends[0] <= bound <= ends[1] for bound in self.boundaries]

            # Determine distance of start to each boundary,
            # (Will ignore a boundary the photon is currently at)
            dist = abs(self.boundaries - z[0])

            # Determine the rank of the boundaries' distances from the start of the vector
            order = np.argsort(dist)

            # Find which boundary of the crossed boundaries is the closest, it will be crossed first, excluding the
            # boundary a photon is currently sitting on (if it is), which will have a distance of 0
            for idx in order:
                if crossed[idx] and dist[idx] > 0:
                    plane = self.boundaries[idx]
                    break

            # Figure out what the second medium is that makes up the interface
            for boundaries, media in self.stack.items():
                if plane in boundaries and interface[0] != media:
                    interface = (interface[0], media)
                    break

            return interface, plane


class Photon:
    def __init__(self, wavelength,
                 system=None,
                 directional_cosines=(0, 0, 1),
                 location_coordinates=(0, 0, 0),
                 weight=1,
                 russian_roulette_constant=20):
        self.wavelength = wavelength
        self.system = system
        self.directional_cosines = directional_cosines
        self.location_coordinates = location_coordinates
        self.weight = weight
        self.russion_roulette_constant = 20

    def move(self, step=None):
        mu_t = self.medium.mu_t
        step = -np.log(np.random.rand()) / mu_t if step is None else step




class OpticalMedium:

    ## TODO: add metod for mu_s and mu_a to return coefficients as funcitons of photon wavelength
    def __init__(self, n, mu_s, mu_a, g, type='default'):
        self.type = type
        self.n = n
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.g = g

    @property
    def mu_t(self):
        return self.mu_s + self.mu_a

    @property
    def albedo(self):
        return self.mu_a / self.mu_t

# TODO: write function to take data from the MC simulation and insert into a database to be used at fitting. This should
#  include overwrite options for when simulation parameters match what has already been inserted, in the case of
#  updates.
def insert_into_mclut_database(simulation_parameters, simulation_results):
    pass