"""
A wrapper around different kernels and feature maps that dispatches on space.
"""
from plum import dispatch

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.feature_maps import (
    deterministic_feature_map_compact,
    random_phase_feature_map_noncompact,
    rejection_sampling_feature_map_hyperbolic,
    rejection_sampling_feature_map_spd,
)
from geometric_kernels.kernels.geometric_kernels import (
    MaternFeatureMapKernel,
    MaternKarhunenLoeveKernel,
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


def default_feature_map(
    *, space: Space = None, num: int = None, kernel: BaseGeometricKernel = None
):
    """
    Constructs the default feature map for the specified space or kernel.

    :param space: space to construct the feature map on. If provided, kernel
                  must either be omitted or set to None.
    :param kernel: kernel to construct the feature map from. If provided,
                   space and num must either be omitted or set to None.
    :param num: controls the number of features (dimensionality of the feature
                map). If omitted or set to None, the default value for each
                respective space is used. Must only be provided when
                constructing a feature map on a space (not from a kernel).

    :return: Callable which is the respective feature map.
    """
    if kernel is not None:
        if space is not None or num is not None:
            raise ValueError(
                "When kernel is provided, space and num must be omitted or set to None"
            )
        return feature_map_from_kernel(kernel)
    elif space is not None:
        if num is None:
            num = default_num(space)
        return feature_map_from_space(space, num)
    else:
        raise ValueError(
            "Either kernel or space must be provided and be different from None"
        )


@dispatch
def feature_map_from_kernel(kernel: MaternKarhunenLoeveKernel):
    return deterministic_feature_map_compact(kernel.space, kernel.num_levels)


@dispatch
def feature_map_from_kernel(kernel: MaternFeatureMapKernel):
    return kernel.feature_map


@dispatch
def feature_map_from_space(space: DiscreteSpectrumSpace, num: int):
    return deterministic_feature_map_compact(space, num)


@dispatch
def feature_map_from_space(space: NoncompactSymmetricSpace, num: int):
    return random_phase_feature_map_noncompact(space, num)


@dispatch
def feature_map_from_space(space: Hyperbolic, num: int):
    return rejection_sampling_feature_map_hyperbolic(space, num)


@dispatch
def feature_map_from_space(space: SymmetricPositiveDefiniteMatrices, num: int):
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
    _DEFAULT_NUM_LEVELS = 25
    _DEFAULT_NUM_RANDOM_PHASES = 3000

    def __new__(
        cls,
        space: Space,
        num=None,
        normalize=True,
        return_feature_map=False,
        **kwargs,
    ):
        r"""
        Construct a kernel and (if `return_feature_map` is `True`) a
        feature map on `space`.

        :param space: space to construct the kernel on.
        :param num: if provided, controls the "order of approximation"
            of the kernel. For the discrete spectrum spaces, this means
            the number of "levels" that go into the truncated series that
            defines the kernel (for example, these are unique eigenvalues
            for the `Hypersphere` or eigenvalues with repetitions for
            the `Graph` or for the `Mesh`). For the noncompact symmetric
            spaces, this is the number of random phases to construct the
            kernel.
        :param normalize: normalize kernel variance. The exact normalization
            technique varies from space to space.
        :param return_feature_map: if `True`, return a feature map (needed
            e.g. for efficient sampling from Gaussian processes) along with
            the kernel. Default is `False`.
        :param ``**kwargs``: any additional keyword arguments to be passed to
            the kernel (like `key`). **Important:** for non-compact symmetric
            spaces (Hyperbolic, SPD) the `key` **must** be provided in kwargs.
        """

        if isinstance(space, DiscreteSpectrumSpace):
            num = num or default_num(space)
            kernel = MaternKarhunenLoeveKernel(space, num, normalize=normalize)
            if return_feature_map:
                feature_map = default_feature_map(kernel=kernel)

        elif isinstance(space, NoncompactSymmetricSpace):
            num = num or default_num(space)
            if "key" in kwargs:
                key = kwargs["key"]
            else:
                raise ValueError(
                    (
                        "MaternGeometricKernel for %s requires mandatory"
                        " keyword argument 'key' which is a random number"
                        " generator specific to the backend used"
                    )
                    % str(type(space))
                )
            feature_map = default_feature_map(space=space, num=num)
            kernel = MaternFeatureMapKernel(
                space, feature_map, key, normalize=normalize
            )
        else:
            raise NotImplementedError

        if return_feature_map:
            return kernel, feature_map
        else:
            return kernel
