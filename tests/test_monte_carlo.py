import random
import unittest

import numpy as np

from my_modules.monte_carlo import System, Photon, OpticalMedium, Illumination, Detector
from my_modules.monte_carlo.hardware import ring_pattern, cone_of_acceptance
from random import random


class TestPhoton(unittest.TestCase):
    def setup_method(self, method):
        self.tissue = OpticalMedium(n=1.33, mu_a=5, mu_s=100, g=0.85, name='tissue', display_color='lightpink')
        surroundings_n = 1.33

        OD = 0.3205672345588178
        ID = 0.27206723455881785
        theta = 0.5743788414166319

        sampler = ring_pattern((ID, OD), theta)
        LED = Illumination(pattern=sampler)
        detector = Detector(cone_of_acceptance(ID))

        self.system = System(
            self.tissue, float('inf'),
            surrounding_n=surroundings_n,
            illuminator=LED,
            detector=(detector, 0)
        )
        self.photon = Photon(
            wavelength=500,  # nm
            system=self.system,
            directional_cosines=(0, 0, 1),  # Initially moving along +z
            location_coordinates=(0, 0, 0),
            weight=1.0,
            russian_roulette_constant=20,
            recurse=True,
            recursion_depth=0,
            recursion_limit=10
        )

    def test_initialization(self):
        assert self.photon.wavelength == 500
        assert self.photon.directional_cosines == (0, 0, 1)
        assert self.photon.location_coordinates == (0, 0, 0)
        assert self.photon.weight == 1.0
        assert self.photon.russian_roulette_constant == 20
        assert self.photon.recurse is True
        assert self.photon.recursion_depth == 0
        assert self.photon.recursion_limit == 10

    def test_directional_cosines(self):
        # Test setter normalization
        self.photon.directional_cosines = (1, 1, 2)
        assert np.linalg.norm(self.photon.directional_cosines) == 1
        assert self.photon.directional_cosines[2] == (2 / np.sqrt(5))

        # Test __setitem__ normalization
        self.photon.directional_cosines[2] = 1
        assert np.linalg.norm(self.photon.directional_cosines) == 1

    def test_location_coordinates(self):
        assert self.photon.location_coordinates == (0, 0, 0)
        self.photon.location_coordinates = (0, 0, 1)
        assert self.photon.location_coordinates == (0, 0, 1)
        self.photon.location_coordinates[2] = 2
        assert self.photon.location_coordinates == (0, 0, 2)

    def test_weight_and_russian_roulette(self):
        # Weight is set
        assert self.photon.weight == 1.0

        # Weight can be reset
        self.photon.weight = 0.5
        assert self.photon.weight == 0.5

        # Weight cannot go negative
        self.photon.weight = -1
        assert self.photon.weight == 0

        # 0 weight kills the photon
        assert self.photon.is_terminated

        self.photon.is_terminated = False

        # Weight below threshold roulette's the photon
        self.photon.weight = 0.0001
        assert self.photon.weight == 0.0001 * self.photon.russian_roulette_constant or self.photon.weight == 0

        # Simulate russian roulette enough times that it should "hit" within epsilon of 1/constant.
        epsilon = 0.01
        n = 1 / epsilon ** 2
        i = 0
        for _ in range(n):
            self.photon.is_terminated = False
            self.photon.weight = 0.0001
            if not self.photon.weight == 0:
                i += 1
        assert np.isclose(i / n, 1 / self.photon.russian_roulette_constant, atol=epsilon)

    def test_absorb(self):
        self.photon.weight = 1
        self.photon.absorb()
        absorbed_weight = self.tissue.albedo_at(self.photon.wavelength)
        assert self.photon.weight == (1 - absorbed_weight)

    def test_move(self):
        # Simple move (no crossing)
        self.photon.location_coordinates = (0, 0, 0)
        self.photon.move(1)
        assert self.photon.location_coordinates == (0, 0, 1)
        assert self.photon.medium == self.tissue

        # Move across interfaces
        self.photon.move(-1.5)
        assert self.photon.location_coordinates == (0, 0, -0.5)
        assert self.photon.medium == self.system.surroundings

        # Move to an interfaces
        self.photon.move(0.5)
        assert self.photon.location_coordinates == (0, 0, 0)
        assert self.photon.medium == (self.system.surroundings, self.tissue)

    def test_scatter(self):



class TestOpticalMedium(unittest.TestCase):
    def setUp(self):
        self.properties = {
            'name': 'test',
            'display_color': 'gray',
            'n': random() + 1,  # Random index of refraction in (1, 2)
            'mu_s': 100 * random() + 40,  # Random reduced scatter coeff in (40, 140)
            'mu_a': 100 * random(),  # Random absorb coeff in (0, 100)
            'g': random(),  # Random anisotropy in (0, 1)
        }

        # Make medium test case
        self.medium = OpticalMedium(**self.properties)

    def test_init(self):
        for prop, val in self.properties.items():
            self.assertEqual(self.properties[prop], getattr(self.medium, prop))

    def test_mu_t(self):
        mu_t = self.properties['mu_s'] + self.properties['mu_a']
        self.assertEqual(self.medium.mu_t, mu_t)

    def test_albedo(self):
        albedo = self.properties['mu_a'] / (self.properties['mu_s'] + self.properties['mu_a'])
        self.assertEqual(self.medium.albedo, albedo)


class TestSystem(unittest.TestCase):
    def setUp(self):
        self.air = OpticalMedium(n=1.0, name='air')
        self.tissue = OpticalMedium(n=1.4, name='tissue')
        self.water = OpticalMedium(n=1.33, name='water')
        self.system = System(self.air, 10, self.tissue, 20, self.water, 5, surrounding_n=1.0)
        self.sys = System()

    def test_system_initialization(self):
        interfaces = np.asarray([0, 10, 30, 35])
        self.assertEqual(len(self.system.layer), 4)  # Air, tissue, water, surroundings
        self.assertEqual(self.system.surroundings.n, 1.0)
        np.testing.assert_array_equal(self.system.boundaries, interfaces)

    def test_in_medium(self):
        self.assertEqual(self.system.in_medium(-5), self.system.surroundings)  # Above the first layer
        self.assertEqual(self.system.in_medium(0), (self.system.surroundings, self.air))  # At first boundary
        self.assertEqual(self.system.in_medium(5), self.air)  # Within the first layer
        self.assertEqual(self.system.in_medium(10), (self.air, self.tissue))  # At second interfaces
        self.assertEqual(self.system.in_medium(15), self.tissue)  # Within the second layer
        self.assertEqual(self.system.in_medium(35), (self.water, self.system.surroundings))  # At last boundary
        self.assertEqual(self.system.in_medium(40), self.system.surroundings)  # Below last layer

    def test_interface_crossed(self):
        # Crossing one interfaces cleanly, should return that interfaces
        zs = [5, 25]

        interface, plane = self.system.interface_crossed(*zs)
        self.assertEqual(interface, (self.air, self.tissue))
        self.assertEqual(plane, 10)  # First interfaces at z = 10

        # Crossing two interfaces in positive direction, should return shallowest
        interface, plane = self.system.interface_crossed(5, 25)  # Crossing from air to tissue (interfaces at 10 and )

        # Crossing two interfaces in negative direciton, should return deepest

        # Crossing no interfaces, should return

        # Crossing out of the media

        # Start at interfaces and don't cross

        # Start at interfaces and cross another

    class TestPhoton(unittest.TestCase):
        def setUp(self):
            self.tissue = OpticalMedium(n=1.4, mu_s=2, mu_a=0.5, g=0.8, name='tissue')
            self.water = OpticalMedium(n=1.33, mu_s=1.5, mu_a=0.3, g=0.7, name='water')
            self.system = System(self.tissue, 20, self.water, 30, surrounding_n=1.0)
            self.photon = Photon(wavelength=500, system=self.system, location_coordinates=(0, 0, 10))

        def test_photon_initialization(self):
            self.assertEqual(self.photon.wavelength, 500)
            self.assertEqual(tuple(self.photon.location_coordinates), (0, 0, 10))
            self.assertEqual(tuple(self.photon.directional_cosines), (0, 0, 1))

        def test_photon_weight_behavior(self):
            self.photon.weight = 0.004  # Triggers Russian roulette
            self.assertIn(self.photon.weight, [0, 0.004 * self.photon.russian_roulette_constant])

        def test_photon_medium(self):
            self.assertEqual(self.photon.medium, self.tissue)
            self.photon.location_coordinates = np.array([0, 0, 25])
            self.assertEqual(self.photon.medium, self.water)

        def test_photon_movement(self):
            initial_position = self.photon.location_coordinates.copy()
            self.photon.move()
            self.assertFalse(np.array_equal(initial_position, self.photon.location_coordinates))

        def test_photon_absorption(self):
            initial_weight = self.photon.weight
            self.photon.absorb()
            self.assertLess(self.photon.weight, initial_weight)
            self.assertGreater(self.photon.A, 0)

        def test_photon_scattering(self):
            initial_direction = self.photon.directional_cosines.copy()
            self.photon.scatter()
            self.assertFalse(np.array_equal(initial_direction, self.photon.directional_cosines))


if __name__ == '__main__':
    unittest.main()
