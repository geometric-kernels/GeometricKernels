import sys
import itertools
import unittest
from parameterized import parameterized_class
import torch
#from torch.autograd.functional import _vmap as vmap
import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.spaces.stiefel import Stiefel
from geometric_kernels.spaces.grassmannian import Grassmannian
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.kernels.feature_maps import random_phase_feature_map

np.set_printoptions(3)

@parameterized_class([
    {'manifold': Stiefel, 'n': 5, 'm': 2, 'order': 20, 'average_order': 100, 'dtype': np.double},
    {'manifold': Stiefel, 'n': 5, 'm': 3, 'order': 20, 'average_order': 100, 'dtype': np.double},
    {'manifold': Stiefel, 'n': 6, 'm': 3, 'order': 20, 'average_order': 100, 'dtype': np.double},

    # {'manifold': Grassmannian, 'n': 5, 'm': 2, 'order': 20, 'average_order': 500, 'dtype': np.double},
    # {'manifold': Grassmannian, 'n': 5, 'm': 3, 'order': 20, 'average_order': 500, 'dtype': np.double},
    # {'manifold': Grassmannian, 'n': 6, 'm': 3, 'order': 20, 'average_order': 500, 'dtype': np.double},

], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["manifold"].__name__}.'
                                                 f'{params_dict["n"]}.{params_dict["order"]}')
class TestCompactLieGroups(unittest.TestCase):

    def setUp(self) -> None:
        self.key = B.create_random_state(self.dtype, seed=0)

        self.key, self.manifold = self.manifold(n=self.n, m=self.m, key=self.key, average_order=self.average_order)
        self.eigenfunctions = self.manifold.get_eigenfunctions(self.order)
        self.lengthscale, self.nu = 0.5, 1.5

        self.kernel = MaternKarhunenLoeveKernel(self.manifold, self.order)
        self.param = dict(lengthscale=np.array(self.lengthscale), nu=np.array(self.nu))
        _, self.state = self.kernel.init_params_and_state()

        self.feature_order = 5000
        self.feature_map, self.key = random_phase_feature_map(self.manifold, self.kernel, self.param, self.state, self.key,
                                                              order=self.feature_order)
        self.key = self.key["key"]

        self.b1, self.b2 = 10, 10
        self.key, self.x = self.manifold.random(self.key, self.b1)
        self.key, self.y = self.manifold.random(self.key, self.b2)

    def test_random(self):
        x_norm = np.linalg.norm(self.x, axis=1)
        self.assertTrue(np.allclose(x_norm, np.ones_like(x_norm)))

    def test_feature_map(self) -> None:
        identity = np.eye(self.n, dtype=self.dtype)[..., :self.m].reshape(-1, self.n, self.m)
        K_00 = self.kernel.K(self.param, self.state, identity, identity)

        K_xx = (self.kernel.K(self.param, self.state, self.x, self.x)/K_00).real
        embed_x = self.feature_map(self.x)
        F_xx = (einsum("ni,mi-> nm", embed_x, embed_x.conj())/self.feature_order/K_00).real
        print(K_xx)
        print('-------')
        print(F_xx)
        self.assertTrue(np.allclose(K_xx, F_xx, atol=5e-2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
