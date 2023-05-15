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
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
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
def set_value(a: B.TFNumeric, index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = tf.where(tf.range(len(a)) == index, value, a)
    return a


@dispatch
def dtype_double(reference: B.TFRandomState):  # type: ignore
    """
    Return `double` dtype of a backend based on the reference.
    """
    return tf.float64


@dispatch
def dtype_integer(reference: B.TFRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return tf.int32


@dispatch
def get_random_state(key: B.TFRandomState):
    """
    Return the random state of a random generator.

    Parameters
    ----------
    key : B.TFRandomState
        The key used to generate the random state.

    Returns
    -------
    Any
        The random state of the random generator.
    """
    return key.state, key.algorithm


@dispatch
def restore_random_state(key: B.TFRandomState, state):
    """
    Set the random state of a random generator.

    Parameters
    ----------
    key : B.TFRandomState
        The random generator.
    state : Any
        The new random state of the random generator.

    Returns
    -------
    Any
       The new random generator with state `state`.
    """
    gen = tf.random.Generator.from_state(state=state[0], alg=state[1])
    return gen


@dispatch
def create_complex(real: _Numeric, imag: B.TFNumeric):
    """
    Returns a complex number with the given real and imaginary parts using tensorflow.

    Args:
    - real: float, real part of the complex number.
    - imag: float, imaginary part of the complex number.

    Returns:
    - complex_num: complex, a complex number with the given real and imaginary parts.
    """
    complex_num = tf.complex(B.cast(B.dtype(imag), from_numpy(imag, real)), imag)
    return complex_num


@dispatch
def dtype_complex(reference: B.TFNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    if B.dtype(reference) == tf.float32:
        return tf.complex64
    else:
        return tf.complex128


@dispatch
def cumsum(x: B.TFNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return tf.math.cumsum(x, axis=axis)
