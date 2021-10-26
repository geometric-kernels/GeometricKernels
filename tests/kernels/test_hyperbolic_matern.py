import geomstats.visualization as visualization
import matplotlib.pyplot as plt
import numpy as np

from geometric_kernels.kernels.geometric_kernels import MaternIntegratedKernel
from geometric_kernels.spaces.hyperbolic import Hyperbolic

_NUM_POINTS = 50
_NU = 0.5


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


if __name__ == "__main__":
    plot_hyperbolic_matern()
