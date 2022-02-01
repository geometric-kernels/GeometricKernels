import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import meshzoo
import numpy as np
import polyscope as ps

from geometric_kernels.backends.jax import SparseGPaxGeometricKernel
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Mesh


class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key


rng = GlobalRNG()

resolution = 40
vertices, faces = meshzoo.icosa_sphere(resolution)
mesh = Mesh(vertices, faces)

nu = 1 / 2.0
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(mesh, truncation_level)
geometric_kernel = SparseGPaxGeometricKernel(base_kernel)

init_params = geometric_kernel.init_params(next(rng))

num_data = 25


def get_data():
    _X = jax.random.randint(
        next(rng), minval=0, maxval=mesh.num_vertices, shape=(num_data, 1)
    )
    _K = geometric_kernel.matrix(init_params, _X, _X)
    _y = jnp.linalg.cholesky(_K + jnp.eye(num_data) * 1e-6) @ jax.random.normal(
        next(rng), (num_data,)
    )
    return _X, _y


X, y = get_data()

X_test = jnp.arange(mesh.num_vertices).reshape(mesh.num_vertices, 1)

Kxx = geometric_kernel.matrix(init_params, X, X) + jnp.eye(num_data) * 1e-6
Lxx, _ = jsp.linalg.cho_factor(Kxx, lower=True)
Lxx_y = jsp.linalg.cho_solve((Lxx, True), y)

Kxt = geometric_kernel.matrix(init_params, X, X_test)
Lxx_Kxt = jsp.linalg.cho_solve((Lxx, True), Kxt)

m = jnp.dot(Lxx_Kxt.T, Lxx_y)  # Ktx @ Kxx^{-1} ^ Y

Ktt = geometric_kernel.matrix(init_params, X_test, X_test)

Ktt_posterior = Ktt - Lxx_Kxt.T @ Lxx_Kxt  # Ktt - Ktx @ Kxx^{-1} @ Kxt
v = jnp.diag(Ktt_posterior)

print("grad", jax.jacfwd(geometric_kernel.matrix)(init_params, X, X))

vertices_X = jnp.take_along_axis(vertices, X.reshape(-1, 1), axis=0)

ps.init()
ps_cloud = ps.register_point_cloud("my points", vertices_X)
ps_cloud.add_scalar_quantity("data", y.flatten())

my_mesh = ps.register_surface_mesh("my mesh", vertices, faces, smooth_shade=True)
my_mesh.add_scalar_quantity("mean", m.squeeze(), enabled=True)
my_mesh.add_scalar_quantity("variance", v.squeeze(), enabled=True)
ps.show()
