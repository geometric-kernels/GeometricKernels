import matplotlib.pyplot as plt
import numpy as np

from geometric_kernels.kernels.geometric_kernels import MaternIntegratedKernel
from geometric_kernels.spaces.sphere import Sphere

_NUM_POINTS = 50
_NU = 2.5


def plot_distance_vs_kernel_sphere():
    hypersphere = Sphere(dim=2)
    lengthscale = 1.

    # Create a set of points along a geodesic
    base = np.r_[1., 0., 0.]
    point = np.r_[-0.9, 0.1, 0.]
    point = point / np.linalg.norm(point)

    geodesic = hypersphere.metric.geodesic(initial_point=base, end_point=point)
    x1 = geodesic(np.linspace(0., 1., 100))
    print('geodesic', x1.shape)

    x2 = x1[-1, None]

    # Compute sphere distance
    distances = hypersphere.distance(x1, x2)

    # Compute heat and Matérn kernels
    heat_kernel_vals = hypersphere.heat_kernel(distances, np.array(0.5*lengthscale**2)[None])  # Lengthscale to heat kernel t parameter
    heat_kernel_vals_normalized = heat_kernel_vals / heat_kernel_vals[-1]
    matern_kernel = MaternIntegratedKernel(hypersphere, _NU, _NUM_POINTS)
    matern_kernel_vals = matern_kernel.K(x1, x2, lengthscale=1.)

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distances, heat_kernel_vals_normalized, color='gold', linewidth=3)
    plt.plot(distances, matern_kernel_vals, color='red', linewidth=1)
    ax.tick_params(labelsize=22)
    ax.set_xlabel(r'distance', fontsize=30)
    ax.set_ylabel(r'k', fontsize=30)
    ax.legend([r'SE', r'Matern'], fontsize=24)
    plt.show()


if __name__ == "__main__":
    plot_distance_vs_kernel_sphere()
