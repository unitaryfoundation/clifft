import unittest

import numpy as np
from residual_ranks import singular_values


class SchmidtTest(unittest.TestCase):
    def test_bell_pair_and_unentangled_spectator_use_coefficient_bit_order(self):
        state = np.zeros(8, dtype=complex)
        state[[0, 5]] = 1 / np.sqrt(2)
        for bit in [0, 2]:
            np.testing.assert_allclose(singular_values(state, 3, [bit]), [1 / np.sqrt(2)] * 2)
        np.testing.assert_allclose(singular_values(state, 3, [1]), [1, 0], atol=1e-15)

    def test_shared_rotation_has_rank_two_across_every_cut(self):
        state = np.zeros(16, dtype=complex)
        state[0], state[-1] = np.cos(0.3), -1j * np.sin(0.3)
        for cut in range(1, 4):
            values = singular_values(state, 4, list(range(cut)))
            np.testing.assert_allclose(values[:2], [np.cos(0.3), np.sin(0.3)])
            np.testing.assert_allclose(values[2:], 0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
