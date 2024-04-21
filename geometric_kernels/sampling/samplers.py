"""
Samplers.
"""

from __future__ import annotations  # By https://stackoverflow.com/a/62136491

from functools import partial

import lab as B
from beartype.typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

# By https://stackoverflow.com/a/62136491
if TYPE_CHECKING:
    from geometric_kernels.feature_maps import FeatureMap


def sample_at(
    feature_map: FeatureMap,
    s: int,
    X: B.Numeric,
    params: Dict[str, B.Numeric],
    key: B.RandomState = None,
    normalize: bool = None,
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    Given a `feature_map` $\phi_{\nu, \kappa}: X \to \mathbb{R}^n$, where
    $\nu, \kappa$ are determined by `params["nu"]` and `params["lengthscale"]`,
    respectively, compute `s` samples of the Gaussian process with kernel

    .. math :: k_{\nu, \kappa}(x, y) = \langle \phi_{\nu, \kappa}(x), \phi_{\nu, \kappa}(y) \rangle_{\mathbb{R}^n}

    at input locations `X` and using the random state `key`.

    Generating a sample from $GP(0, k_{\nu, \kappa})$ is as simple as computing

    .. math :: \sum_{j=1}^n w_j \cdot (\phi_{\nu, \kappa}(x))_j \qquad w_j \stackrel{IID}{\sim} N(0, 1).

    .. note::
        Fixing $w_j$, and treating $x \to (\phi_{\nu, \kappa}(x))_j$ as basis
        functions, while letting $x$ vary, you get an actual *function* as a
        sample, meaning something that can be evaluated at any $x \in X$.
        The way to fix $w_j$ in code is to apply :func:`~.make_deterministic`
        utility function.

    :param feature_map:
        The feature map $\phi_{\nu, \kappa}$ that defines the Gaussian process
        $GP(0, k_{\nu, \kappa})$ to sample from.
    :param s:
        The number of samples to generate.
    :param X:
        An [N, <axis>]-shaped array containing N elements of the space
        `feature_map` is defined on. <axis> is the shape of these elements.
        These are the points to evaluate the samples at.
    :param params:
        Parameters of the kernel (length scale and smoothness).
    :param key: random state, either `np.random.RandomState`,
        `tf.random.Generator`, `torch.Generator` or `jax.tensor` (which
        represents a random state).
    :param normalize:
        Passed down to `feature_map` directly. Controls whether to force the
        average variance of the Gaussian process to be around 1 or not. If None,
        follows the standard behavior, typically same as normalize=True.

        Defaults to None.

    :return:
        [N, s]-shaped array containing s samples of the $GP(0, k_{\nu, \kappa})$
        evaluated at X.
    """

    if key is None:
        key = B.global_random_state(B.dtype(X))

    _context, features = feature_map(X, params, key=key, normalize=normalize)  # [N, M]

    if _context is not None:
        key = _context

    num_features = B.shape(features)[-1]

    key, random_weights = B.randn(
        key, B.dtype(params["lengthscale"]), num_features, s
    )  # [M, S]

    random_sample = B.matmul(features, random_weights)  # [N, S]

    return key, random_sample


def sampler(
    feature_map: FeatureMap, s: Optional[int] = 1, **kwargs
) -> Callable[[Any], Any]:
    """
    A helper wrapper around `sample_at` that fixes `feature_map`, `s` and the
    keyword arguments in ``**kwargs`` but leaves `X`, `params` and the other
    keyword arguments vary.

    :param feature_map:
        The feature map to fix.
    :param s:
        The number of samples parameter to fix.

        Defaults to 1.
    :param ``**kwargs``:
        Keyword arguments to fix.

    :return:
        The version of :func:`sample_at` with parameters `feature_map` and `s`
        fixed, together with the keyword arguments in ``**kwargs``.
    """

    sample_f = partial(sample_at, feature_map, s, **kwargs)
    new_docstring = f"""
        This is a version of the `{sample_at.__name__}` function with
        - feature_map={feature_map},
        - s={s},
        - and additional keyword arguments {kwargs}.

        The original docstring follows.

        {sample_at.__doc__}
        """
    sample_f.__name__ = sample_at.__name__  # type: ignore[attr-defined]
    sample_f.__doc__ = new_docstring

    return sample_f
