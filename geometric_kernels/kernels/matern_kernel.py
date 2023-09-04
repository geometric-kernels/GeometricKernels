"""
A wrapper around different kernels and feature maps that dispatches on space.
"""
import lab as B
from plum import dispatch

from geometric_kernels.kernels import MaternFeatureMapKernel, MaternKarhunenLoeveKernel
from geometric_kernels.kernels.feature_maps import (
    deterministic_feature_map_compact,
    random_phase_feature_map_noncompact,
    rejection_sampling_feature_map_hyperbolic,
    rejection_sampling_feature_map_spd,
)
from geometric_kernels.spaces import (
    DiscreteSpectrumSpace,
    Graph,
    Hyperbolic,
    Mesh,
    NoncompactSymmetricSpace,
    Space,
    SymmetricPositiveDefiniteMatrices,
)


@dispatch
def default_feature_map(space: DiscreteSpectrumSpace, *, num, kernel):
    return deterministic_feature_map_compact(space, kernel)


@dispatch
def default_feature_map(space: NoncompactSymmetricSpace, *, num, kernel):
    return random_phase_feature_map_noncompact(space, num)


@dispatch
def default_feature_map(space: Hyperbolic, *, num, kernel):
    return rejection_sampling_feature_map_hyperbolic(space, num)


@dispatch
def default_feature_map(space: Hyperbolic, *, num, kernel):
    return rejection_sampling_feature_map_hyperbolic(space, num)


@dispatch
def default_feature_map(space: SymmetricPositiveDefiniteMatrices, *, num, kernel):
    return rejection_sampling_feature_map_spd(space, num)


@dispatch
def default_num(space: Mesh):
    return min(MaternGeometricKernel._DEFAULT_NUM_EIGENFUNCTIONS, space.num_vertices)


@dispatch
def default_num(space: Graph):
    return min(MaternGeometricKernel._DEFAULT_NUM_EIGENFUNCTIONS, space.num_vertices)


@dispatch
def default_num(space: DiscreteSpectrumSpace):
    return MaternGeometricKernel._DEFAULT_NUM_LEVELS


@dispatch
def default_num(space: NoncompactSymmetricSpace):
    return MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES


class MaternGeometricKernel:
    """
    This class represents a Matérn geometric kernel that "just works".

    Upon creation, this class unpacks into a specific geometric kernel based
    on the provided Space, and the associated feature map.
    """

    _DEFAULT_NUM_EIGENFUNCTIONS = 1000
    _DEFAULT_NUM_LEVELS = 35
    _DEFAULT_NUM_RANDOM_PHASES = 3000

    def __new__(cls, space: Space, num=None, **kwargs):
        r"""
        Construct a kernel and a feature map on `space`.

        :param space: space to construct the kernel on
        :param num: if provided, controls the "order of approximation"
            of the kernel. For the discrete spectrum spaces, this means
            the number of unique eigenvalues that go into the truncated
            series that defines the kernel. For the noncompact symmetric
            spaces, this is the number of random phases to construct the
            kernel.
        :param **kwargs: any additional keyword arguments to be passed to
            the kernel (like `key`).
        """

        if isinstance(space, DiscreteSpectrumSpace):
            num = num or default_num(space)
            kernel = MaternKarhunenLoeveKernel(space, num)
            feature_map = default_feature_map(space, kernel=kernel, num=num)

        elif isinstance(space, NoncompactSymmetricSpace):
            num = num or default_num(space)
            key = kwargs.get("key", B.create_random_state())
            feature_map = default_feature_map(space, kernel=kernel, num=num)
            kernel = MaternFeatureMapKernel(space, feature_map, key)
        else:
            raise NotImplementedError

        return kernel, feature_map
