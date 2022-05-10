import numpy as np

from geometric_kernels.spaces.hyperbolic import Hyperbolic


def test_hyperboloid_distance():
    hyperboloid = Hyperbolic(dim=2)
    N = 10

    # Data points
    base = np.r_[7.14142843e00, -5.00000000e00, -5.00000000e00]
    point = np.r_[14.17744688, 10.0, 10.0]
    geodesic = hyperboloid.metric.geodesic(initial_point=base, end_point=point)
    x1 = geodesic(np.linspace(0.0, 1.0, N))  # (N, 3)
    x2 = x1[-1, None]  # (1, 3)

    our_dist_12 = hyperboloid.distance(x2, x1)
    geomstats_dist_12 = hyperboloid.metric.dist(x2, x1)

    our_dist_11 = hyperboloid.distance(x1, x1)  # (N, N)
    geomstats_dist_11 = hyperboloid.metric.dist_pairwise(x1, n_jobs=1)  # (N, N)

    our_dist_11_diag = hyperboloid.distance(x1, x1, diag=True)  # (N, )
    geomstats_dist_11_diag = hyperboloid.metric.dist(x1, x1)  # (N, )

    assert np.allclose(our_dist_12, geomstats_dist_12)
    assert np.allclose(our_dist_11, geomstats_dist_11)
    assert np.allclose(our_dist_11_diag, geomstats_dist_11_diag)
