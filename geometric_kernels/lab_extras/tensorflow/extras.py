import sys

import lab as B
import tensorflow as tf
import tensorflow_probability as tfp
from beartype.typing import Any, List, Optional
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TFNumeric, B.NPNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    if sys.version_info[:2] <= (3, 9):
        index = tf.cast(index, tf.int32)
    return tf.experimental.numpy.take_along_axis(
        a, index, axis=axis
    )  # the absence of explicit cast to int64 causes an error for Python 3.9 and below


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
    Obtain the eigenpairs that correspond to the `k` lowest eigenvalues
    of a symmetric positive semi-definite matrix `L`.
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
def float_like(reference: B.TFNumeric):
    """
    Return the type of the reference if it is a floating point type.
    Otherwise return `double` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if reference_dtype.is_floating:
        return reference_dtype
    else:
        return tf.float64


@dispatch
def dtype_integer(reference: B.TFRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return tf.int32


@dispatch
def int_like(reference: B.TFNumeric):
    reference_dtype = reference.dtype
    if reference_dtype.is_integer:
        return reference_dtype
    else:
        return tf.int32


@dispatch
def get_random_state(key: B.TFRandomState):
    """
    Return the random state of a random generator.

    :param key:
        The random generator of type `B.TFRandomState`.
    """
    return tf.identity(key.state), key.algorithm


@dispatch
def restore_random_state(key: B.TFRandomState, state):
    """
    Set the random state of a random generator. Return the new random
    generator with state `state`.

    :param key:
        The random generator.
    :param state:
        The new random state of the random generator of type `B.TFRandomState`.
    """
    gen = tf.random.Generator.from_state(state=tf.identity(state[0]), alg=state[1])
    return gen


@dispatch
def create_complex(real: _Numeric, imag: B.TFNumeric):
    """
    Return a complex number with the given real and imaginary parts using tensorflow.

    :param real:
        float, real part of the complex number.
    :param imag:
        float, imaginary part of the complex number.
    """
    complex_num = tf.complex(B.cast(B.dtype(imag), from_numpy(imag, real)), imag)
    return complex_num


@dispatch
def complex_like(reference: B.TFNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    return B.promote_dtypes(tf.complex64, reference.dtype)


@dispatch
def is_complex(reference: B.TFNumeric):
    """
    Return True if reference of `complex` dtype.
    """
    return (B.dtype(reference) == tf.complex64) or (B.dtype(reference) == tf.complex128)


@dispatch
def cumsum(x: B.TFNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return tf.math.cumsum(x, axis=axis)


@dispatch
def qr(x: B.TFNumeric, mode="reduced"):
    """
    Return a QR decomposition of a matrix x.
    """
    full_matrices = mode == "complete"
    Q, R = tf.linalg.qr(x, full_matrices=full_matrices)
    return Q, R


@dispatch
def slogdet(x: B.TFNumeric):
    """
    Return the sign and log-determinant of a matrix x.
    """
    sign, logdet = tf.linalg.slogdet(x)
    return sign, logdet


@dispatch
def eigvalsh(x: B.TFNumeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.
    """
    return tf.linalg.eigvalsh(x)


@dispatch
def reciprocal_no_nan(x: B.TFNumeric):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.
    """
    return tf.math.reciprocal_no_nan(x)


@dispatch
def complex_conj(x: B.TFNumeric):
    """
    Return complex conjugate
    """
    return tf.math.conj(x)


@dispatch
def logical_xor(x1: B.TFNumeric, x2: B.TFNumeric):
    """
    Return logical XOR of two arrays.
    """
    return tf.math.logical_xor(x1, x2)


@dispatch
def count_nonzero(x: B.TFNumeric, axis=None):
    """
    Count non-zero elements in an array.
    """
    return tf.math.count_nonzero(x, axis=axis)


@dispatch
def dtype_bool(reference: B.TFRandomState):  # type: ignore
    """
    Return `bool` dtype of a backend based on the reference.
    """
    return tf.bool


@dispatch
def bool_like(reference: B.NPNumeric):
    """
    Return the type of the reference if it is of boolean type.
    Otherwise return `bool` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if reference_dtype.is_bool:
        return reference_dtype
    else:
        return tf.bool
