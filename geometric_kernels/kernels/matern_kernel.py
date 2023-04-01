"""
A wrapper around different kernels and feature maps that dispatches on space.
"""
import lab as B

from geometric_kernels.kernels import MaternFeatureMapKernel, MaternKarhunenLoeveKernel
from geometric_kernels.kernels.feature_maps import (
    deterministic_feature_map_compact,
    random_phase_feature_map_noncompact,
)
from geometric_kernels.spaces import (
    Circle,
    DiscreteSpectrumSpace,
    Graph,
    Hypersphere,
    Mesh,
    NoncompactSymmetricSpace,
    ProductDiscreteSpectrumSpace,
    Space,
)


class MaternGeometricKernel:
    """
    This class represents a Matérn geometric kernel that "just works".

    Upon creation, this class unpacks into a specific geometric kernel based
    on the provided Space, and the associated feature map.
    """

    _MAX_NUM_EIGENFUNCTIONS = 1000
    _MAX_NUM_LEVELS = 10
    _MAX_NUM_RANDOM_PHASES = 3000

    def __new__(cls, space: Space, num=None, **kwargs):
        r"""
        Construct a kernel and a feature map on `space`.

        :param space: space to construct the kernel on
        :param num: if provided, controls the "order of approximation"
            of the kernel. For the discrete spectrum spaces, this means
            the number of unique eigenvalues that go into the truncated
            series that define the kernel. For the noncompact symmetric
            spaces, this is the number of random phases to construct the
            kernel.
        :param **kwargs: any additional keyword arguments to be passed to
            the kernel (like `key`).
        """

        # good ole isinstance
        if isinstance(space, DiscreteSpectrumSpace):
            if num is None:
                if isinstance(space, Mesh):
                    num = min(cls._MAX_NUM_EIGENFUNCTIONS, space.num_vertices)
                elif isinstance(space, Graph):
                    num = min(cls._MAX_NUM_EIGENFUNCTIONS, space.num_vertices)
                elif isinstance(space, Circle):
                    num = cls._MAX_NUM_LEVELS
                elif isinstance(space, Hypersphere):
                    num = cls._MAX_NUM_LEVELS
                else:
                    raise NotImplementedError

            kernel = MaternKarhunenLoeveKernel(space, num)
            feature_map = deterministic_feature_map_compact(space, kernel)

        elif isinstance(space, NoncompactSymmetricSpace):
            key = kwargs.get("key", B.create_random_state())
            feature_map = random_phase_feature_map_noncompact(
                space, cls._MAX_NUM_RANDOM_PHASES
            )
            kernel = MaternFeatureMapKernel(space, feature_map, key)

        elif isinstance(space, ProductDiscreteSpectrumSpace):
            raise NotImplementedError("TODO")

        else:
            raise NotImplementedError

        return kernel, feature_map
