"""
Provides :class:`MaternGeometricKernel`,the geometric Matérn kernel---with
the heat kernel as a special case---that just works.

It wraps around different kernels and feature maps, dispatching on the space.

Unless you know exactly what you are doing, use :class:`MaternGeometricKernel`.
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
    CompactMatrixLieGroup,
    DiscreteSpectrumSpace,
    Graph,
    Hyperbolic,
    HypercubeGraph,
    Hypersphere,
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

    :param space:
        A space to construct the feature map on. If provided, kernel must
        either be omitted or set to None.
    :param kernel:
        A kernel to construct the feature map from. If provided, `space` and
        `num` must either be omitted or set to None.
    :param num:
        Controls the number of features (dimensionality of the feature
        map). If omitted or set to None, the default value for each
        respective space is used. Must only be provided when
        constructing a feature map on a space (not from a kernel).

    :return:
        Callable which is the respective feature map.
    """
    if kernel is not None:
        if space is not None or num is not None:
            raise ValueError(
                "When kernel is provided, space and num must be omitted or set to None"
            )
        return feature_map_from_kernel(kernel)  # type: ignore[call-overload]
    elif space is not None:
        if num is None:
            num = default_num(space)  # type: ignore[call-overload]
        return feature_map_from_space(space, num)  # type: ignore[call-overload]
    else:
        raise ValueError(
            "Either kernel or space must be provided and be different from None"
        )


@overload
def feature_map_from_kernel(kernel: MaternKarhunenLoeveKernel):
    if isinstance(kernel.space, CompactMatrixLieGroup):
        # Because `CompactMatrixLieGroup` does not currently support explicit
        # eigenfunction computation (they only support addition theorem).
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

    :param kernel:
        A kernel to construct the feature map from.

    :return:
        A feature map.
    :rtype: feature_maps.FeatureMap

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
    if isinstance(space, CompactMatrixLieGroup):
        return RandomPhaseFeatureMapCompact(
            space, num, MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES
        )
    elif isinstance(space, Hypersphere):
        num_computed_levels = space.num_computed_levels
        if num_computed_levels > 0:
            return DeterministicFeatureMapCompact(space, min(num, num_computed_levels))
        else:
            return RandomPhaseFeatureMapCompact(
                space, num, MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES
            )
    else:
        return DeterministicFeatureMapCompact(space, num)


@overload
def feature_map_from_space(space: NoncompactSymmetricSpace, num: int):
    if isinstance(space, Hyperbolic):
        return RejectionSamplingFeatureMapHyperbolic(space, num)
    elif isinstance(space, SymmetricPositiveDefiniteMatrices):
        return RejectionSamplingFeatureMapSPD(space, num)
    else:
        return RandomPhaseFeatureMapNoncompact(space, num)


@dispatch
def feature_map_from_space(space: Space, num: int):
    """
    Return the default feature map for the specified space `space` and
    approximation level `num`.

    :param space:
        A space to construct the feature map on.
    :param num:
        Approximation level.

    :return:
        A feature map.
    :rtype: feature_maps.FeatureMap

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
def default_num(space: DiscreteSpectrumSpace) -> int:
    if isinstance(space, CompactMatrixLieGroup):
        return MaternGeometricKernel._DEFAULT_NUM_LEVELS_LIE_GROUP
    elif isinstance(space, (Graph, Mesh)):
        return min(
            MaternGeometricKernel._DEFAULT_NUM_EIGENFUNCTIONS, space.num_vertices
        )
    elif isinstance(space, HypercubeGraph):
        return min(MaternGeometricKernel._DEFAULT_NUM_LEVELS, space.dim + 1)
    else:
        return MaternGeometricKernel._DEFAULT_NUM_LEVELS


@overload
def default_num(space: NoncompactSymmetricSpace) -> int:
    return MaternGeometricKernel._DEFAULT_NUM_RANDOM_PHASES


@dispatch
def default_num(space: Space) -> int:
    """
    Return the default approximation level for the `space`.

    :param space:
        A space.

    :return:
        The default approximation level.

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

    Upon creation, unpacks into a specific geometric kernel based on the
    provided space, and, optionally, the associated (approximate) feature map.
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

        .. note::
           See :doc:`this page </theory/feature_maps>` for a brief
           introduction into feature maps.

        :param space:
            Space to construct the kernel on.
        :param num:
            If provided, controls the "order of approximation" of the kernel.
            For the discrete spectrum spaces, this means the number of "levels"
            that go into the truncated series that defines the kernel (for
            example, these are unique eigenvalues for the
            :class:`~.spaces.Hypersphere` or eigenvalues with repetitions for
            the :class:`~.spaces.Graph` or for the :class:`~.spaces.Mesh`).
            For the non-compact symmetric spaces
            (:class:`~.spaces.NoncompactSymmetricSpace`), this is the number
            of random phases used to construct the kernel.

            If num=None, we use a (hopefully) reasonable default, which is
            space-dependent.
        :param normalize:
            Normalize the kernel (and the feature map). If normalize=True,
            then either $k(x, x) = 1$ for all $x \in X$, where $X$ is the
            `space`, or $\int_X k(x, x) d x = 1$, depending on the space.

            Defaults to True.

            .. note::
                For many kernel methods, $k(\cdot, \cdot)$ and
                $a k(\cdot, \cdot)$ are indistinguishable, whatever the
                positive constant $a$ is. For these, it makes sense to use
                normalize=False to save up some computational overhead. For
                others, like for the Gaussian process regression, the
                normalization of the kernel might be important. In
                these cases, you will typically want to set normalize=True.

        :param return_feature_map:
            If `True`, return a feature map (needed e.g. for efficient sampling
            from Gaussian processes) along with the kernel.

            Default is False.
        :param ``**kwargs``:
            Any additional keyword arguments to be passed to the kernel
            (like `key`).

        .. note::
           For non-compact symmetric spaces, like :class:`~.spaces.Hyperbolic`
           or :class:`~.spaces.SymmetricPositiveDefiniteMatrices`, the `key`
           **must** be provided in ``**kwargs``.
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
