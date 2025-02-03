import random
import unittest

import numpy as np

from my_modules.monte_carlo import System, Photon, OpticalMedium, insert_into_mclut_database, sample_illumination
from random import random


class TestOpticalMedium(unittest.TestCase):
    def setUp(self):
        self.properties = {
            'type': 'test',
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
        self.air = OpticalMedium(n=1.0, type='air')
        self.tissue = OpticalMedium(n=1.4, type='tissue')
        self.water = OpticalMedium(n=1.33, type='water')
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
        self.assertEqual(self.system.in_medium(10), (self.air, self.tissue))  # At second interface
        self.assertEqual(self.system.in_medium(15), self.tissue)  # Within the second layer
        self.assertEqual(self.system.in_medium(35), (self.water, self.system.surroundings))  # At last boundary
        self.assertEqual(self.system.in_medium(40), self.system.surroundings)  # Below last layer

    def test_interface_crossed(self):
        # Crossing one interface cleanly, should return that interface
        zs = [5, 25]

        interface, plane = self.system.interface_crossed(*zs)
        self.assertEqual(interface, (self.air, self.tissue))
        self.assertEqual(plane, 10)  # First interface at z = 10

        # Crossing two interfaces in positive direction, should return shallowest
        interface, plane = self.system.interface_crossed(5, 25)  # Crossing from air to tissue (interface at 10 and )

        # Crossing two interfaces in negative direciton, should return deepest

        # Crossing no interface, should return

        # Crossing out of the media

        # Start at interface and don't cross

        # Start at interface and cross another

    class TestPhoton(unittest.TestCase):
        def setUp(self):
            self.tissue = OpticalMedium(n=1.4, mu_s=2, mu_a=0.5, g=0.8, type='tissue')
            self.water = OpticalMedium(n=1.33, mu_s=1.5, mu_a=0.3, g=0.7, type='water')
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
