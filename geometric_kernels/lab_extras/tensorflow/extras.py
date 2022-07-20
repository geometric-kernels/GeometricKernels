from typing import Any, List, Optional

import lab as B
import tensorflow as tf
import tensorflow_probability as tfp
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TFNumeric, B.NPNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return tf.gather(a, B.flatten(index), axis=axis)


@dispatch
def from_numpy(_: B.TFNumeric, b: Union[List, B.Numeric, B.NPNumeric, B.TFNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return tf.convert_to_tensor(b)


@dispatch
def trapz(y: _Numeric, x: _Numeric, dx=None, axis=-1):  # type: ignore
    """
    Integrate along the given axis using the composite trapezoidal rule.
    """
    return tfp.math.trapz(y, x, dx, axis)


@dispatch
def norm(x: _Numeric, ord: Optional[Any] = None, axis: Optional[int] = None):  # type: ignore
    """
    Matrix or vector norm.
    """
    return tf.norm(x, ord=ord, axis=axis)


@dispatch
def logspace(start: _Numeric, stop: _Numeric, num: int = 50, base: _Numeric = 50.0):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    y = tf.linspace(start, stop, num)
    return tf.math.pow(base, y)


@dispatch
def degree(a: B.TFNumeric):  # type: ignore
    """
    Given a vector a, return a diagonal matrix with a as main diagonal.
    """
    degrees = tf.reduce_sum(a, axis=0)  # type: ignore
    return tf.linalg.diag(degrees)


@dispatch
def eigenpairs(L: B.TFNumeric, k: int):
    """
    Obtain the k highest eigenpairs of a symmetric PSD matrix L.
    """
    l, u = tf.linalg.eigh(L)
    return l[:k], u[:, :k]


@dispatch
def set_value(a: B.TFNumeric, index: int, value: B.Numeric):
    """
    Set a[index] = value.
    """
    a = tf.where(tf.range(len(a)) == index, value, a)
    return a
