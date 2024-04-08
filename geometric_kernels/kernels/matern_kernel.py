"""
A wrapper around different kernels and feature maps that dispatches on space.
"""

from plum import dispatch, overload

from geometric_kernels.feature_maps import (
    DeterministicFeatureMapCompact,
    RandomPhaseFeatureMapCompact,
    RandomPhaseFeatureMapNoncompact,
    RejectionSamplingFeatureMapHyperbolic,
    RejectionSamplingFeatureMapSPD,
)
from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.feature_map import MaternFeatureMapKernel
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import (
    CompactHomogeneousSpace,
    DiscreteSpectrumSpace,
    Graph,
    Hyperbolic,
    MatrixLieGroup,
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


@overload
def feature_map_from_kernel(kernel: MaternKarhunenLoeveKernel):
    if isinstance(kernel.space, MatrixLieGroup) or isinstance(
        kernel.space, CompactHomogeneousSpace
    ):
        # Because `MatrixLieGroup` and `CompactHomogeneousSpace` do not
        # currently support explicit eigenfunction computation (they
        # only support addition theorem).
        return RandomPhaseFeatureMapCompact(
            kernel.space,
            kernel.num_levels,
            MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES,
        )
    else:
        return DeterministicFeatureMapCompact(kernel.space, kernel.num_levels)


@overload
def feature_map_from_kernel(kernel: MaternFeatureMapKernel):
    return kernel.feature_map


@dispatch
def feature_map_from_kernel(kernel: BaseGeometricKernel):
    """
    Return the default feature map for the specified kernel `kernel`.

    .. note::
       This function is organized as an abstract dispatcher plus a set of
       @overload-decorated implementations, one for each type of kernels.

       When followed by an "empty" @dispatch-decorated function of the same
       name, plum-dispatch changes the default behavior of the `@overload`
       decorator, allowing the implementations inside the preceding
       @overload-decorated functions. This is opposed to the standard behavior
       when @overload-decorated functions can only provide type signature,
       while the general implementation should be contained in the function
       of the same name without an `@overload` decorator.

       The trick is taken from https://beartype.github.io/plum/integration.html.

    .. note::
       For dispatching to work, the empty @dispatch-decorated function should
       follow (not precede) the @overload-decorated implementations in the code.
    """
    raise NotImplementedError(
        "feature_map_from_kernel is not implemented for the kernel of type %s."
        % str(type(kernel))
    )


@overload
def feature_map_from_space(space: DiscreteSpectrumSpace, num: int):
    return DeterministicFeatureMapCompact(space, num)


@overload
def feature_map_from_space(space: MatrixLieGroup, num: int):
    return RandomPhaseFeatureMapCompact(
        space, num, MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES
    )


@overload
def feature_map_from_space(space: CompactHomogeneousSpace, num: int):
    return RandomPhaseFeatureMapCompact(
        space, num, MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES
    )


@overload
def feature_map_from_space(space: NoncompactSymmetricSpace, num: int):
    return RandomPhaseFeatureMapNoncompact(space, num)


@overload
def feature_map_from_space(space: Hyperbolic, num: int):
    return RejectionSamplingFeatureMapHyperbolic(space, num)


@overload
def feature_map_from_space(space: SymmetricPositiveDefiniteMatrices, num: int):
    return RejectionSamplingFeatureMapSPD(space, num)


@dispatch
def feature_map_from_space(space: Space, num: int):
    """
    Return the default feature map for the specified space `space` and
    approximation level `num`.

    .. note::
       This function is organized as an abstract dispatcher plus a set of
       @overload-decorated implementations, one for each type of spaces.

       When followed by an "empty" @dispatch-decorated function of the same
       name, plum-dispatch changes the default behavior of the `@overload`
       decorator, allowing the implementations inside the preceding
       @overload-decorated functions. This is opposed to the standard behavior
       when @overload-decorated functions can only provide type signature,
       while the general implementation should be contained in the function
       of the same name without an `@overload` decorator.

       The trick is taken from https://beartype.github.io/plum/integration.html.

    .. note::
       For dispatching to work, the empty @dispatch-decorated function should
       follow (not precede) the @overload-decorated implementations in the code.
    """
    raise NotImplementedError(
        "feature_map_from_space is not implemented for the space of type %s."
        % str(type(space))
    )


@overload
def default_num(space: Mesh):
    return min(MaternGeometricKernel._DEFAULT_NUM_EIGENFUNCTIONS, space.num_vertices)


@overload
def default_num(space: Graph):
    return min(MaternGeometricKernel._DEFAULT_NUM_EIGENFUNCTIONS, space.num_vertices)


@overload
def default_num(space: DiscreteSpectrumSpace):
    return MaternGeometricKernel._DEFAULT_NUM_LEVELS


@overload
def default_num(space: MatrixLieGroup):
    return MaternGeometricKernel._DEFAULT_NUM_LEVELS_LIE_GROUP


@overload
def default_num(space: CompactHomogeneousSpace):
    return MaternGeometricKernel._DEFAULT_NUM_LEVELS_LIE_GROUP


@overload
def default_num(space: NoncompactSymmetricSpace):
    return MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES


@dispatch
def default_num(space: Space):
    """
    Return the default approximation level for the `space`.

    .. note::
       This function is organized as an abstract dispatcher plus a set of
       @overload-decorated implementations, one for each type of spaces.

       When followed by an "empty" @dispatch-decorated function of the same
       name, plum-dispatch changes the default behavior of the `@overload`
       decorator, allowing the implementations inside the preceding
       @overload-decorated functions. This is opposed to the standard behavior
       when @overload-decorated functions can only provide type signature,
       while the general implementation should be contained in the function
       of the same name without an `@overload` decorator.

       The trick is taken from https://beartype.github.io/plum/integration.html.

    .. note::
       For dispatching to work, the empty @dispatch-decorated function should
       follow (not precede) the @overload-decorated implementations in the code.
    """
    raise NotImplementedError(
        "default_num is not implemented for the space of type %s." % str(type(space))
    )


class MaternGeometricKernel:
    """
    This class represents a Matérn geometric kernel that "just works". Unless
    you really know what you are doing, you should always use this kernel class.

    Upon creation, this class unpacks into a specific geometric kernel based on
    the provided space, and, optionally, the associated (approximate) feature map.
    """

    _DEFAULT_NUM_EIGENFUNCTIONS = 1000
    _DEFAULT_NUM_LEVELS = 25
    _DEFAULT_NUM_LEVELS_LIE_GROUP = 20
    _DEFAULT_NUM_RANDOM_PHASES = 3000

    def __new__(
        cls,
        space: Space,
        num: int = None,
        normalize: bool = True,
        return_feature_map: bool = False,
        **kwargs,
    ):
        r"""
        Construct a kernel and (if `return_feature_map` is `True`) a
        feature map on `space`.

        :param space: space to construct the kernel on.
        :param num:
            If provided, controls the "order of approximation" of the kernel.
            For the discrete spectrum spaces, this means the number of "levels"
            that go into the truncated series that defines the kernel (for
            example, these are unique eigenvalues for the `Hypersphere` or
            eigenvalues with repetitions for the `Graph` or for the `Mesh`).
            For the noncompact symmetric spaces, this is the number of random
            phases to construct the kernel.

            The default is space-dependent.
        :param normalize:
            Normalize kernel variance. The exact normalization technique
            varies from space to space.

            Defaults to True.
        :param return_feature_map:
            If `True`, return a feature map (needed e.g. for efficient sampling
            from Gaussian processes) along with the kernel.

            Default is False.
        :param ``**kwargs``:
            Any additional keyword arguments to be passed to the kernel (like `key`).

        .. note::
           For non-compact symmetric spaces (Hyperbolic, SPD) the `key`
           **must** be provided in kwargs.
        """

        kernel: BaseGeometricKernel
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
