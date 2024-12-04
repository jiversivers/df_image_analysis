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

        # State
        self.wavelength = wavelength
        self.weight = weight
        self.is_terminated = False
        self.directional_cosines = np.asarray(directional_cosines, dtype=np.float64)
        self._location_coordinates = np.asarray(location_coordinates, dtype=np.float64)
        self.system = system

        # Simulation parameters
        self.russian_roulette_constant = russian_roulette_constant

        # Path tracker
        self.location_history = [self.location_coordinates]

        # Terminal trackers
        self.R = 0
        self.A = 0
        self.T = 0

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        self._system = system
        self.medium = self.system.in_medium(self.location_coordinates) if self.system is not None else None

    def absorb(self):
        # Check that there is an absorber
        if self.medium and not self.is_terminated:
            # For absorbance at interface/boundary
            medium = self.headed_into if isinstance(self.medium, (list, tuple)) else self.medium

            # Simulate absorbance at current location
            self.A += self.weight * medium.albedo
            self.weight = self.weight - self.weight * medium.albedo  # Weight setter handles Russian Roulette and final decision

    def scatter(self):
        # Check if there is still a scatterer
        if self.medium and not isinstance(self.medium, (list, tuple)):
            # Get random scatter angle from media anisotropy
            theta, phi = self.get_solid_angles()

            # Update direction; check if photon is nearly vertical
            self.update_direction(theta, phi)

    def move(self, step=None):
        # Get step size from random distribution
        # If at an interface, hypothetically bump forward slightly to determine upcoming material properties
        mu_t = self.system.in_medium(
            np.nextafter(self.location_coordinates, 1e6 * self.directional_cosines)).mu_t if isinstance(
            self.medium, (list, tuple)) else self.medium.mu_t
        step = -np.log(np.random.rand()) / mu_t if step is None else step

        # Update location by d_location as function of step size and direction
        self.location_coordinates = self.location_coordinates + step * self.directional_cosines

    @property
    def location_coordinates(self):
        return self._location_coordinates

    @location_coordinates.setter
    def location_coordinates(self, new_coordinates):
        """
        The idea with the location_coordinate setter is to ensure the photon is only moved according to the parameters
        of the medium it is currently in, and will not have "carry-over" movement through a medium according to
        parameters of a previous medium. To accomplish this, the first thing we do is check if the new coordinates will
        cause the photon to cross an interface.

        If the photon will cross an interface with the coordinates, we let the photon move only to the interface, but
        store the remainder of the move for after refraction. We calculate the refraction at the interface which updates
        the direction of the photon and the medium it is in, so now we are ready to complete the move. To do this, we
        just recurse into the setter. By recursing (rather than hard-updating) we ensure we don't cross a second
        boundary without refracting. It is extremely unlikely that a photon would ever end up directly on an interface
        without being put there by this logic, but to protect against that, we can check that the photon has already
        "refracted" by ensuring checking that the photon is headed into the medium that it is currently in. If that is
        not true, then we also check if the interface is a system boundary. If it is, we don't bother refracting (as we
        cannot at a system boundary because there is only one index of refraction) and we just update the medium. If it
        is not a system boundary but is an actual interface, then the refraction method has not been called for that
        interface yet, so we handle that (below).

        In the case that the photon will not cross an interface, we can just hard-update the hidden attribute for the
        photon position. This is the case that will break the recursion for refracted photons. This is also where the
        check for photons at an interface needs to occur. That logic is as follows: if the photon is at an interface,
        check that the direction it is moving will move it into the material that it is currently in. If so, move it. If
        not, check if it is at a system boundary. If it is not, refract it and move it with recursion (in case
        refraction causes it to cross a boundary). If it is (at a system boundary), update the medium (without
        refraction) and move it (recursively).

        In short:
        0) Get interface crossing
        1) If interface is crossed
            A) If at an interface (different from that which will be crossed)
                i) If moving into a new medium
                    a) If at a system boundary
                        !) Update current medium to new medium
                    b) If at an actual interface
                        !) Refract
                ii) If not moving into new medium
                    a) pass
            B) Reflect (updates weight)
            C) Move to interface (update hidden attribute)
            D) Refract
            E) Move remaining fraction (recurse)
        2) If not
            A) If at interface
                i) If moving into new medium
                    a) If at a system boundary
                        !) Update current medium to new medium
                        @) Move (recurse)
                    b) If at an actual interface
                        !) Refract
                        @) Move (recurse)
                ii) If not moving into new medium
                    a) Move (update hidden attribute)
            B) If not at an interface
                i) Move (update hidden attribute)

        These shorthand steps will be used to organize the code, with steps and sub-steps in comments preceding
        lines/blocks that accomplish them.
        """
        if not self.is_terminated:
            # 0) Get interface crossing
            # (returns two interfaces and plane of interface if crossed (note: does not return the interface the photon
            # is currently on in cases when it is directly on an interface), returns one interface and empty with plane
            # of boundary crossing a system boundary, else returns false and empty list for plane)
            try:
                interface, plane = self.system.interface_crossed(self.location_coordinates, new_coordinates)
            except UnboundLocalError:
                print(self.location_coordinates, new_coordinates)
                assert False
            # 1) If an interface is crossed
            if interface and all(interface):
                # 1.A) Deal with case of photon currently at interface
                if plane == self.location_coordinates[2]:
                    # 1.A.i) Moving into new medium
                    if not self.has_refracted():
                        # 1.A.i.a) Update current medium
                        if plane == self.system.boundaries[0] or plane == self.system.boundaries[-1]:
                            self.medium = self.headed_into()
                        # 1.a.i.b) Refract
                        else:
                            self.reflect((self.medium, self.headed_into))
                            self.refract((self.medium, self.headed_into))

                # 1.B) Reflect (updates weight)
                self.reflect(interface)

                # 1.C) Move to interface
                # Calculate fraction of step needed to move only to interface and update
                d_location = new_coordinates - self.location_coordinates
                fraction = (plane - self.location_coordinates[2]) / (d_location[2])
                new_coordinates[0] = self.location_coordinates[0] + fraction * d_location[0]
                new_coordinates[1] = self.location_coordinates[1] + fraction * d_location[1]
                new_coordinates[2] = plane
                self._location_coordinates = new_coordinates
                self.location_history.append(new_coordinates)

                # 1.D) Refract (updates direction and medium)
                self.refract(interface)

                # 1.E) Move remaining fraction(recurse)
                d_location = self.directional_cosines * (interface[0].mu_t / interface[1].mu_t) * (
                        1 - fraction) * d_location
                self.location_coordinates = self.location_coordinates + d_location

            elif interface:
                # 1.C) Move to boundary
                # Calculate fraction of step needed to move only to boundary and update
                d_location = new_coordinates - self.location_coordinates
                fraction = (plane - self.location_coordinates[2]) / (d_location[2])
                new_coordinates[0] = self.location_coordinates[0] + fraction * d_location[0]
                new_coordinates[1] = self.location_coordinates[1] + fraction * d_location[1]
                new_coordinates[2] = plane
                self._location_coordinates = new_coordinates
                self.location_history.append(new_coordinates)

            # 2) Not crossing an interface
            else:
                # To make the logic match for the at-interface cases, set plane to current z
                plane = self.location_coordinates[2]

                # 2.A) If at an interface
                if self.location_coordinates[2] in self.system.boundaries:
                    # 2.A.i) Moving into a new medium
                    if not self.has_refracted():
                        # 2.A.i.a) Update current medium and move (recurse)
                        if plane == self.system.boundaries[0] or plane == self.system.boundaries[-1]:
                            self.medium = self.headed_into
                            self.location_coordinates = new_coordinates
                        # 2.A.i.b) Refract and move (recurse)
                        else:
                            self.reflect((self.medium, self.headed_into))
                            self.refract((self.medium, self.headed_into))
                            self.location_coordinates = new_coordinates
                    # 2.A.ii) Move
                    else:
                        self._location_coordinates = new_coordinates
                        self.location_history.append(new_coordinates)
                        self.medium = self.system.in_medium(self.location_coordinates)
                # 2.B) Move
                else:
                    self._location_coordinates = new_coordinates
                    self.location_history.append(new_coordinates)
                    self.medium = self.system.in_medium(self.location_coordinates)

        if self.location_coordinates[2] >= self.system.boundaries[-1]:
            self.T += self.weight
            self.is_terminated = True
        elif self.location_coordinates[2] <= self.system.boundaries[0]:
            self.R += self.weight
            self.is_terminated = True

    def has_refracted(self):
        # Check what direction photon is headed
        headed = self.headed_into

        # Return if it is headed into the same medium it is currently set to
        return headed == self.medium

    @property
    def headed_into(self):
        bumped_coords = np.nextafter(self.location_coordinates, 1e6 * np.sign(self.directional_cosines))
        # bumped_coords[np.isnan(bumped_coords)] = 0
        return self.system.in_medium(bumped_coords)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_weight):
        # Check if photon is finished
        if new_weight <= 0:
            self._weight = 0
        # If it's close, play RR to determine
        elif new_weight > 0 and new_weight < 0.005:
            self.russian_roulette()
        else:
            self._weight = new_weight
        self.is_terminated = self.weight <= 0

    def russian_roulette(self):
        if np.random.rand() < 1 / self.russian_roulette_constant:
            self.weight = 0
        else:
            self.weight *= self.russian_roulette_constant

    def refract(self, interface):
        n1_n2 = interface[0].n / interface[1].n
        sin_theta_t = n1_n2 * np.sqrt(1 - self.directional_cosines[2] ** 2)
        if np.isnan(sin_theta_t):
            assert False

        # Critical angle reflection
        if sin_theta_t > 1:
            self.directional_cosines[2] = -self.directional_cosines[2]
        else:
            # Snell's law
            self.directional_cosines[2] = np.sign(self.directional_cosines[2]) * np.sqrt(1 - sin_theta_t ** 2)
            self.directional_cosines /= np.linalg.norm(self.directional_cosines)
            self.medium = interface[1]

    def get_solid_angles(self):
        [xi, zeta] = np.random.rand(2)
        if self.medium.g != 0:
            lead_coeff = 1 / (2 * self.medium.g)
            term_2 = self.medium.g ** 2
            term_3 = (1 - term_2) / (1 - self.medium.g + (2 * self.medium.g * xi))
            cosine_theta = lead_coeff * (1 + term_2 - term_3)
        else:
            cosine_theta = (2 * xi) - 1
        theta = np.arccos(cosine_theta)
        phi = 2 * np.pi * zeta
        return theta, phi

    def update_direction(self, theta, phi):
        mu_x, mu_y, mu_z = self.directional_cosines

        if np.abs(mu_z) > 0.999:
            self.directional_cosines[0] = np.sin(theta) * np.cos(phi)  # mu_x'
            self.directional_cosines[1] = np.sin(theta) * np.sin(phi)  # mu_y'
            self.directional_cosines[2] = np.sign(mu_z) * np.cos(theta)  # mu_z'
        else:
            deno = np.sqrt(1 - (mu_z ** 2))

            # X
            numr1 = (mu_x * mu_y * np.cos(phi)) - (mu_y * np.sin(phi))
            self.directional_cosines[0] = (np.sin(theta) * (numr1 / deno)) + (mu_x * np.cos(theta))

            # Y
            numr2 = (mu_y * mu_z * np.cos(phi)) + (mu_x * np.sin(phi))
            self.directional_cosines[1] = (np.sin(theta) * (numr2 / deno)) + (mu_y * np.cos(theta))

            # Z
            self.directional_cosines[2] = -(np.sin(theta) * np.cos(phi) * deno) + mu_z * np.cos(theta)

    def reflect(self, interface):
        specular_reflection = abs(
            ((interface[1].n - interface[0].n) ** 2)
            /
            (interface[1].n + interface[0].n)
        )
        self.R += self.weight * specular_reflection
        self.weight = self.weight - self.weight * specular_reflection


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