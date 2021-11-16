import geomstats.visualization as visualization
import matplotlib.pyplot as plt
import numpy as np

from geometric_kernels.kernels.geometric_kernels import MaternIntegratedKernel
from geometric_kernels.spaces.hyperbolic import Hyperbolic

_NUM_POINTS = 50
_NU = 2.5


def plot_hyperbolic_matern():
    hyperboloid = Hyperbolic(dim=2)
    kernel = MaternIntegratedKernel(hyperboloid, _NU, _NUM_POINTS)

    # construct a `uniform` grid on hyperbolic space
    s = np.linspace(-5, 5, 25)
    xx, yy = np.meshgrid(s, s)
    points = np.c_[xx.ravel(), yy.ravel()]
    points = hyperboloid.from_coordinates(points, "intrinsic")

    # base point to compute the kernel from
    base_point = hyperboloid.from_coordinates(np.r_[0, 0], "intrinsic")

    kernel_vals = kernel.K(base_point, points, lengthscale=0.5)

    # vizualize
    plt.figure(figsize=(5, 5))
    visualization.plot(points, space="H2_poincare_disk", c=kernel_vals, cmap="plasma")

    # plt.savefig('./test_hyperbolic_matern.png')
    plt.show()


def plot_distance_vs_kernel_hyperbolic():
    hyperboloid = Hyperbolic(dim=2)
    lengthscale = 1.0

    base = np.r_[7.14142843, -5.0, -5.0]
    point = np.r_[14.17744688, 10.0, 10.0]

    geodesic = hyperboloid.metric.geodesic(initial_point=base, end_point=point)

    # Data points
    x1 = geodesic(np.linspace(0.0, 1.0, 100))

    x2 = x1[-1, None]

    # Compute hyperbolic distance
    distances = hyperboloid.distance(x1, x2)
    # Compute heat and Matérn kernels
    heat_kernel_vals = hyperboloid.heat_kernel(
        distances, np.array(0.5 * lengthscale ** 2)[None], num_points=1000
    )  # Lengthscale to heat kernel t parameter
    heat_kernel_vals_normalized = heat_kernel_vals / heat_kernel_vals[-1]
    matern_kernel = MaternIntegratedKernel(hyperboloid, _NU, _NUM_POINTS)
    matern_kernel_vals = matern_kernel.K(x1, x2, lengthscale=1.0)

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distances, heat_kernel_vals_normalized, color="gold", linewidth=3)
    plt.plot(distances, matern_kernel_vals, color="red", linewidth=1)
    ax.tick_params(labelsize=22)
    ax.set_xlabel("distance", fontsize=30)
    ax.set_ylabel("k", fontsize=30)
    ax.legend(["SE", "Matern"], fontsize=24)
    plt.show()
    # plt.savefig("./plot_distance_vs_kernel.png")


if __name__ == "__main__":
    plot_hyperbolic_matern()
    plot_distance_vs_kernel_hyperbolic()
