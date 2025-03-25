import random
import unittest

import numpy as np

from photon_canon import System, Medium, Illumination, Detector
from photon_canon.optics import Photon, IndexableProperty
from photon_canon.hardware import ring_pattern, cone_of_acceptance, ID, OD, THETA
from random import random


class TestPhoton(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initializes a Photon instance with explicit settings for use in test cases.

        This method sets up a known Photon state, ensuring consistent
        initial conditions for testing initialization and behavior.
        """
        self.water = Medium(n=1.33, desc='water')
        self.tissue = Medium(n=1.53, mu_a=5, mu_s=100, g=0.85, desc='tissue')
        surroundings_n = 1.33

        sampler = ring_pattern((ID, OD), THETA)
        led = Illumination(pattern=sampler)
        detector = Detector(cone_of_acceptance(ID))

        self.system = System(
            self.water, 0.2,
            self.tissue, float('inf'),
            surrounding_n=surroundings_n,
            illuminator=led,
            detector=(detector, 0)
        )
        self.photon = Photon(
            wavelength=500,  # nm
            batch_size=100,
            system=self.system,
            directional_cosines=(0, 0, 1),  # Initially moving along +z
            location_coordinates=(0, 0, 0),
            weights=1.0,
            russian_roulette_constant=20,
            recurse=True,
            recursion_depth=0,
            recursion_limit=10,
            throw_recursion_error=True,
            keep_secondary_photons=True,

        )

    def test_initialization(self):
        """
        Verifies that all Photon attributes are correctly initialized.

        This test ensures that explicitly set attributes, as well as derived
        or automatically assigned attributes, match expected values.
        """

        # Explicit settings
        self.assertEqual(self.photon.wavelength, 500)
        self.assertEqual(self.photon.batch_size, 100)
        self.assertEqual(self.photon.system, self.system)
        self.assertEqual(self.photon.directional_cosines.shape, (100, 3))
        self.assertTrue(np.all(self.photon.directional_cosines == np.array([0, 0, 1])))
        self.assertIsInstance(self.photon.directional_cosines, IndexableProperty)
        self.assertEqual(self.photon.location_coordinates.shape, (100, 3))
        self.assertTrue(np.all(self.photon.location_coordinates == np.array([0, 0, 0])))
        self.assertEqual(self.photon.weights.shape, (100,))
        self.assertTrue(np.all(self.photon.weights == 1.0))
        self.assertEqual(self.photon.russian_roulette_constant, 20)
        self.assertTrue(self.photon.recurse)
        self.assertEqual(self.photon.recursion_depth, 0)
        self.assertEqual(self.photon.recursion_limit, 10)

        # Hidden settings
        self.assertEqual(self.photon.T, 0)
        self.assertEqual(self.photon.R, 0)
        self.assertEqual(self.photon.A, 0)
        self.assertEqual(self.photon.tir_count.shape, (100,))
        self.assertTrue(np.all(self.photon.tir_count == 0))
        self.assertEqual(self.photon.exit_location.shape, (100, 3))
        self.assertTrue(np.all(np.isnan(self.photon.exit_location)))
        self.assertEqual(self.photon.exit_direction.shape, (100, 3))
        self.assertTrue(np.all(np.isnan(self.photon.exit_direction)))
        self.assertEqual(self.photon.exit_weights.shape, (100,))
        self.assertTrue(np.all(np.isnan(self.photon.exit_weights)))
        self.assertEqual(self.photon.location_history.shape, (100, 3, 1))
        self.assertTrue(np.all(self.photon.location_history.squeeze() == self.photon.location_coordinates))
        self.assertEqual(self.photon.weights_history.shape, (100, 1))
        self.assertTrue(np.all(self.photon.weights_history.squeeze() == self.photon.weights))
        self.assertEqual(self.photon.cache_register.shape, (100,))
        self.assertFalse(np.any(self.photon.cache_register))
        self.assertEqual(self.photon._medium.shape, (100,))
        self.assertEqual(self.photon.at_interface.shape, (100,))
        self.assertEqual(len(self.photon.secondary_photons), 0)
        self.assertIsInstance(self.photon.secondary_photons, list)

    def test_directional_cosines(self) -> None:
        """
        Tests the filling and normalization of directional cosines.

        Ensures that the directional cosines setter correctly fills the batch:
        - When only one directional cosine is given
        and correctly normalizes:
        - The entire vector when assigned as a whole.
        - Individual components when updated via indexing.
        """
        # Test setter filling
        self.photon.directional_cosines = (1, 0, 0)
        self.assertEqual(self.photon.directional_cosines.shape, (100, 3))
        self.assertTrue(np.all(self.photon.directional_cosines == np.array([1, 0, 0])))

        # Test setter normalization
        self.photon.directional_cosines = (1, 1, 1)
        self.assertIsInstance(self.photon.directional_cosines, IndexableProperty)
        self.assertTrue(np.all(np.isclose(np.linalg.norm(self.photon.directional_cosines, axis=-1), 1)))
        self.assertTrue(np.all(self.photon.directional_cosines == 1 / np.sqrt(3)))

        # Test __setitem__ normalization (now normalizing by [1/sqrt(3), 1/sqrt(3), 1]
        self.photon.directional_cosines[:, 2] = 1
        self.assertIsInstance(self.photon.directional_cosines, IndexableProperty)
        self.assertTrue(np.all(np.isclose(np.linalg.norm(self.photon.directional_cosines, axis=-1), 1)))
        self.assertTrue(
            np.all(np.isclose(self.photon.directional_cosines, np.sqrt(3 * np.array([1 / 3, 1 / 3, 1]) / 5))))

    def test_location_coordinates(self):
        """
        Tests the setting and filling of location coordinates.

        Ensures that the coordinates setter correctly fills the batch when:
        - Only one coordinate set is given
        - The entire vector when assigned as a whole.
        - Individual components when updated via indexing.
        """
        # Test filling
        self.photon.location_coordinates = (0, 0, 0.1)
        self.assertEqual(self.photon.location_coordinates.shape, (100, 3))
        self.assertIsInstance(self.photon.location_coordinates, np.ndarray)
        self.assertIsInstance(self.photon.location_coordinates[0, 0], np.float64)
        self.assertTrue(np.all(self.photon.location_coordinates == np.array([0, 0, 0.1])))

        # Test batch setting
        self.location_coordinates = np.repeat(np.array([0, 0, 0])[np.newaxis, ...], 100, axis=0)
        self.assertEqual(self.photon.location_coordinates.shape, (100, 3))
        self.assertIsInstance(self.photon.location_coordinates, np.ndarray)
        self.assertIsInstance(self.photon.location_coordinates[0, 0], np.float64)
        self.assertTrue(np.all(self.location_coordinates == np.array([0, 0, 0])))

        # Test indexing
        self.photon.location_coordinates[:, 2] = 0.1
        self.assertIsInstance(self.photon.location_coordinates, np.ndarray)
        self.assertIsInstance(self.photon.location_coordinates[0, 0], np.float64)
        self.assertTrue(np.all(self.photon.location_coordinates == np.array([0, 0, 0.1])))

    def test_weight_and_russian_roulette(self):
        """
        Test the setting, filling, auto-killing, and roulette behavior of the weight setter.

        Ensures that the weight setter correctly fills the batch when:
        - Only one coordinate set is given
        - The entire vector when assigned as a whole.
        - Individual components when updated via indexing.
        Also ensures that the weight setter correctly puts photons below the threshold (0.005) through russian roulette
        with correct augmentation and with proper probability of augmentation.
        """
        # Weight can be reset in batch
        self.photon.weights = 0.5
        self.assertEqual(self.photon.weights.shape, (100,))
        self.assertTrue(np.all(self.photon.weights == 0.5))

        # Weight can be batched manually
        self.photon.weights = np.repeat(1.0, 100)
        self.assertEqual(self.photon.weights.shape, (100,))
        self.assertTrue(np.all(self.photon.weights == 1.0))

        # Weight cannot go negative
        self.photon.weights = -1
        self.assertTrue(np.all(self.photon.weights == 0))

        # 0 weight kills the photon
        self.assertTrue(self.photon.is_terminated)

        # Non-0 weight revives them
        self.photon.weights = 1
        self.assertFalse(self.photon.is_terminated)

        # Weight below threshold roulette's the photon
        self.photon.weights = 0.0001
        self.assertTrue(
            np.all(
                (self.photon.weights == 0.0001 * self.photon.russian_roulette_constant) | (self.photon.weights == 0)
            )
        )

        # Simulate russian roulette enough times that it should "hit" within epsilon of 1/constant.
        epsilon = 0.01
        n = 2 * int((1 / epsilon ** 2) / self.photon.batch_size)
        i = 0
        for _ in range(n):
            self.photon.weights = 0.0001
            i += np.sum(self.photon.weights != 0) / self.photon.batch_size
        self.assertTrue(np.isclose(i / n, 1 / self.photon.russian_roulette_constant, atol=epsilon))

    def test_absorb(self):
        """
        Tests the absorbing behavior. Should be 0 in water and 1 * albedo in tissue.
        """
        self.photon.absorb()
        self.assertTrue(np.all(self.photon.weights == 1))
        self.photon.location_coordinates = (0, 0, 0.3)  # Get into tissue
        self.photon.absorb()
        absorbed_weights = self.tissue.albedo_at(self.photon.wavelength)
        self.assertTrue(np.all(self.photon.weights == (1 - absorbed_weights)))
        self.assertTrue(np.isclose(self.photon.A, self.photon.batch_size * absorbed_weights))

    def test_move(self):
        """
        Tests the expected movement behavior with both auto moves to interfaces when scattering is 0 and movements
        inside scattering media.
        """
        # Simple move
        self.photon.move()  # Should move to first interface with headed_into = tissue
        self.assertTrue(np.all(self.photon.location_coordinates == np.array([0, 0, 0.2])))
        self.assertTrue(np.all(self.photon.headed_into() == self.tissue))
        self.assertTrue(np.all(self.photon.medium == self.tissue))
        self.assertTrue(np.all(self.photon.at_interface))

        # Check reflection at index mismatched interface
        spec_ref = np.abs((self.water.n - self.tissue.n) / (self.water.n + self.tissue.n)) ** 2
        new_weight = 1 - spec_ref
        self.assertTrue(np.all(np.isclose(self.photon.weights, new_weight)))
        self.assertIsNotNone(self.photon.secondary_photons)

        # Move across interfaces
        self.photon.move(0.1)
        self.assertTrue(np.all(np.isclose(self.photon.location_coordinates, np.array([0, 0, 0.3]))))
        self.assertTrue(np.all(self.photon.headed_into() == self.tissue))
        self.assertTrue(np.all(self.photon.medium == self.tissue))
        self.assertFalse(np.any(self.photon.at_interface))

        # Move at an angle
        dir_cos = np.array([1, 2, -2]) / 3  # Aimed towards interface
        self.photon.directional_cosines = dir_cos
        self.photon.move(float('inf'))  # move to interface
        self.assertFalse(np.any(np.isclose(self.photon.directional_cosines, dir_cos)))

        # Check refraction direction update
        n1_n2 = self.tissue.n / self.water.n
        new_dir = n1_n2 * dir_cos  # Scale in-plane components
        new_dir[2] = -np.cos(np.arcsin(n1_n2 * np.sin(np.arccos(-2 / 3))))  # Refract normal component
        self.assertTrue(np.all(np.isclose(self.photon.directional_cosines, new_dir)))

        # Check reflection at angled mismatched boundary
        spec_ref = np.abs(((self.tissue.n * dir_cos[2]) - (self.water.n * new_dir[2])) /
                          ((self.tissue.n * dir_cos[2]) + (self.water.n * new_dir[2]))) ** 2
        new_weight -= new_weight * spec_ref
        self.assertTrue(np.all(np.isclose(self.photon.weights, new_weight)))
        self.assertIsNotNone(self.photon.secondary_photons)

    def test_scatter(self):
        """
        Test that scatter returns new directions when in scattering media, and that it lets directions persis in
        non-scattering media and at interfaces.
        """
        # At non-scattering interface
        dir_cos = np.array([0, 0, 1])
        self.photon.scatter()
        self.assertTrue(np.all(self.photon.directional_cosines == dir_cos))

        # In non-scattering media
        self.photon.move(0.1)
        self.photon.scatter()
        self.assertTrue(np.all(self.photon.directional_cosines == dir_cos))

        # At interface with scatterer
        self.photon.move(0.1)
        self.photon.scatter()
        self.assertTrue(np.all(self.photon.directional_cosines == dir_cos))

        # In scattering media
        self.photon.move(0.1)
        self.photon.scatter()
        self.assertFalse(np.any(self.photon.directional_cosines == dir_cos))

class TestMedium(unittest.TestCase):
    def setUp(self):
        self.properties = {
            'desc': 'test',
            'display_color': 'gray',
            'n': random() + 1,  # Random index of refraction in (1, 2)
            'mu_s': 100 * random() + 40,  # Random reduced scatter coeff in (40, 140)
            'mu_a': 100 * random(),  # Random absorb coeff in (0, 100)
            'g': random(),  # Random anisotropy in (0, 1)
        }

        # Make medium test case
        self.medium = Medium(**self.properties)

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
        self.air = Medium(n=1.0, desc='air')
        self.tissue = Medium(n=1.4, desc='tissue')
        self.water = Medium(n=1.33, desc='water')
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
            self.tissue = Medium(n=1.4, mu_s=2, mu_a=0.5, g=0.8, desc='tissue')
            self.water = Medium(n=1.33, mu_s=1.5, mu_a=0.3, g=0.7, desc='water')
            self.system = System(self.tissue, 20, self.water, 30, surrounding_n=1.0)
            self.photon = Photon(wavelength=500, system=self.system, location_coordinates=(0, 0, 10))

        def test_photon_initialization(self):
            self.assertEqual(self.photon.wavelength, 500)
            self.assertEqual(tuple(self.photon.location_coordinates), (0, 0, 10))
            self.assertEqual(tuple(self.photon.directional_cosines), (0, 0, 1))

        def test_photon_weight_behavior(self):
            self.photon.weights = 0.004  # Triggers Russian roulette
            self.assertIn(self.photon.weights, [0, 0.004 * self.photon.russian_roulette_constant])

        def test_photon_medium(self):
            self.assertEqual(self.photon.medium, self.tissue)
            self.photon.location_coordinates = np.array([0, 0, 25])
            self.assertEqual(self.photon.medium, self.water)

        def test_photon_movement(self):
            initial_position = self.photon.location_coordinates.copy()
            self.photon.move()
            self.assertFalse(np.array_equal(initial_position, self.photon.location_coordinates))

        def test_photon_absorption(self):
            initial_weights = self.photon.weights
            self.photon.absorb()
            self.assertLess(self.photon.weights, initial_weights)
            self.assertGreater(self.photon.A, 0)

        def test_photon_scattering(self):
            initial_direction = self.photon.directional_cosines.copy()
            self.photon.scatter()
            self.assertFalse(np.array_equal(initial_direction, self.photon.directional_cosines))


if __name__ == '__main__':
    unittest.main()
