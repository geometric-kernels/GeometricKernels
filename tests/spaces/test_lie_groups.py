import itertools
import unittest

import lab as B
import numpy as np
from opt_einsum import contract as einsum
from parameterized import parameterized_class

from geometric_kernels.kernels.feature_maps import random_phase_feature_map
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.su import SUGroup

np.set_printoptions(3)


@parameterized_class(
    [
        # {'group': SOGroup, 'n': 3, 'order': 10, 'dtype': np.double},
        # {'group': SOGroup, 'n': 4, 'order': 10, 'dtype': np.double},
        # {'group': SOGroup, 'n': 5, 'order': 10, 'dtype': np.double},
        # {'group': SOGroup, 'n': 6, 'order': 10, 'dtype': np.double},
        # {'group': SOGroup, 'n': 7, 'order': 10, 'dtype': np.double},
        # {'group': SOGroup, 'n': 8, 'order': 10, 'dtype': np.double},
        {"group": SUGroup, "n": 2, "order": 20, "dtype": np.cdouble},
        # {'group': SUGroup, 'n': 3, 'order': 20, 'dtype': np.cdouble},
        # {'group': SUGroup, 'n': 4, 'order': 20, 'dtype': np.cdouble},
        # {'group': SUGroup, 'n': 5, 'order': 20, 'dtype': np.cdouble},
        # {'group': SUGroup, 'n': 6, 'order': 20, 'dtype': np.cdouble},
    ],
    class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["group"].__name__}.'
    f'{params_dict["n"]}.{params_dict["order"]}',
)
class TestCompactLieGroups(unittest.TestCase):
    def setUp(self) -> None:
        self.key = B.create_random_state(self.dtype, seed=0)

        self.group = self.group(n=self.n)
        self.eigenfunctions = self.group.get_eigenfunctions(self.order)
        self.lengthscale, self.nu = 2.0, 5.0

        self.kernel = MaternKarhunenLoeveKernel(self.group, self.order)
        self.param = dict(lengthscale=np.array(1), nu=np.array(1.5))
        _, self.state = self.kernel.init_params_and_state()

        self.feature_order = 50000
        self.feature_map, self.key = random_phase_feature_map(
            self.group,
            self.kernel,
            self.param,
            self.state,
            self.key,
            order=self.feature_order,
        )
        self.key = self.key["key"]

        self.b1, self.b2 = 10, 10
        self.key, self.x = self.group.random(self.key, self.b1)
        self.key, self.y = self.group.random(self.key, self.b2)

    def test_random(self):
        eye_ = np.matmul(self.x, self.group.inverse(self.x))[None, ...]
        diff = eye_ - np.eye(self.n, dtype=self.dtype)
        zeros = np.zeros_like(eye_)
        self.assertTrue(np.allclose(diff, zeros))

    def test_character_conjugation_invariance(self):
        num_samples_x = 20
        num_samples_g = 20
        self.key, xs = self.group.random(self.key, num_samples_x)
        self.key, gs = self.group.random(self.key, num_samples_g)
        conjugates = np.matmul(np.matmul(gs, xs), self.group.inverse(gs))

        conj_gammas = self.eigenfunctions._torus_representative(conjugates)
        xs_gammas = self.eigenfunctions._torus_representative(conjugates)
        for chi in self.eigenfunctions._characters:
            chi_vals_xs = chi(xs_gammas)
            chi_vals_conj = chi(conj_gammas)
            self.assertTrue(np.allclose(chi_vals_xs, chi_vals_conj))

    def test_character_at_identity_equals_dimension(self):
        identity = np.eye(self.n, dtype=self.dtype).reshape(1, self.n, self.n)
        identity_gammas = self.eigenfunctions._torus_representative(identity)
        dimensions = self.eigenfunctions._dimensions
        characters = self.eigenfunctions._characters
        for chi, dim in zip(characters, dimensions):
            chi_val = chi(identity_gammas)
            self.assertTrue(np.allclose(chi_val.real, dim))
            self.assertTrue(np.allclose(chi_val.imag, 0))

    def test_characters_orthogonality(self):
        num_samples_x = 5 * 10**5
        self.key, xs = self.group.random(self.key, num_samples_x)
        gammas = self.eigenfunctions._torus_representative(xs)
        characters = self.eigenfunctions._characters
        scalar_products = np.zeros((self.order, self.order), dtype=self.dtype)
        for a, b in itertools.product(enumerate(characters), repeat=2):
            i, chi1 = a
            j, chi2 = b
            scalar_products[i, j] = np.mean((np.conj(chi1(gammas)) * chi2(gammas)).real)
        print(np.max(np.abs(scalar_products - np.eye(self.order, dtype=self.dtype))))
        self.assertTrue(
            np.allclose(
                scalar_products, np.eye(self.order, dtype=self.dtype), atol=5e-2
            )
        )

    def test_feature_map(self) -> None:
        identity = np.eye(self.n, dtype=self.dtype).reshape(-1, self.n, self.n)
        K_00 = self.kernel.K(self.param, self.state, identity, identity)

        K_xx = (self.kernel.K(self.param, self.state, self.x, self.x) / K_00).real
        embed_x = self.feature_map(self.x)
        F_xx = (
            einsum("ni,mi-> nm", embed_x, embed_x.conj()) / self.feature_order / K_00
        ).real
        print(K_xx)
        print("-------")
        print(F_xx)
        self.assertTrue(np.allclose(K_xx, F_xx, atol=5e-2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
