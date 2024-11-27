import lab as B
from beartype.typing import List
from lab import dispatch
from lab.util import abstract
from plum import Union
from scipy.sparse import spmatrix


@dispatch
@abstract()
def take_along_axis(a: B.Numeric, index: B.Numeric, axis: int = 0):
    """
    Gathers elements of `a` along `axis` at `index` locations.

    :param a:
        Array of any backend, as in `numpy.take_along_axis`.
    :param index:
        Array of any backend, as in `numpy.take_along_axis`.
    :param axis:
        As in `numpy.take_along_axis`.
    """


@dispatch
@abstract()
def from_numpy(_: B.Numeric, b: Union[List, B.Numeric]):
    """
    Converts the array `b` to a tensor of the same backend as `_`.

    :param _:
        Array of any backend used to determine the backend.
    :param b:
        Array of any backend or list to be converted to the backend of _.
    """


@dispatch
@abstract()
def trapz(y: B.Numeric, x: B.Numeric, dx: B.Numeric = 1.0, axis: int = -1):
    """
    Integrate along the given axis using the trapezoidal rule.

    :param y:
        Array of any backend, as in `numpy.trapz`.
    :param x:
        Array of any backend, as in `numpy.trapz`.
    :param dx:
        Array of any backend, as in `numpy.trapz`.
    :param axis:
        As in `numpy.trapz`.
    """


@dispatch
@abstract()
def logspace(start: B.Numeric, stop: B.Numeric, num: int = 50):
    """
    Return numbers spaced evenly on a log scale.

    :param start:
        Array of any backend, as in `numpy.logspace`.
    :param stop:
        Array of any backend, as in `numpy.logspace`.
    :param num:
        As in `numpy.logspace`.
    """


def cosh(x: B.Numeric) -> B.Numeric:
    r"""
    Compute hyperbolic cosine using the formula

    .. math:: \textrm{cosh}(x) = \frac{\exp(x) + \exp(-x)}{2}.

    :param x:
        Array of any backend.
    """
    return 0.5 * (B.exp(x) + B.exp(-x))


def sinh(x: B.Numeric) -> B.Numeric:
    r"""
    Compute hyperbolic sine using the formula

    .. math:: \textrm{sinh}(x) = \frac{\exp(x) - \exp(-x)}{2}.

    :param x:
        Array of any backend.
    """
    return 0.5 * (B.exp(x) - B.exp(-x))


@dispatch
@abstract()
def degree(a):
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.

    :param a:
        Array of any backend or `scipy.sparse` array.
    """


@dispatch
@abstract()
def eigenpairs(L, k: int):
    """
    Obtain the eigenpairs that correspond to the `k` lowest eigenvalues
    of a symmetric positive semi-definite matrix `L`.

    :param a:
        Array of any backend or `scipy.sparse` array.
    :param k:
        The number of eigenpairs to compute.
    """


@dispatch
@abstract()
def set_value(a, index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.

    :param a:
        Array of any backend or `scipy.sparse` array.
    :param index:
        The index.
    :param value:
        The value to set at the given index.
    """


@dispatch
@abstract()
def dtype_double(reference: B.RandomState):
    """
    Return `double` dtype of a backend based on the reference.

    :param reference:
        A random state to infer the backend from.
    """


@dispatch
@abstract()
def float_like(reference: B.Numeric):
    """
    Return the type of the reference if it is a floating point type.
    Otherwise return `double` dtype of a backend based on the reference.

    :param reference:
        Array of any backend.
    """


@dispatch
@abstract()
def dtype_integer(reference: B.RandomState):
    """
    Return `int` dtype of a backend based on the reference.

    :param reference:
        A random state to infer the backend from.
    """


@dispatch
@abstract()
def int_like(reference: B.Numeric):
    """
    Return the type of the reference if it is integer type.
    Otherwise return `int32` dtype of a backend based on the reference.

    :param reference:
        Array of any backend.
    """


@dispatch
@abstract()
def get_random_state(key: B.RandomState):
    """
    Return the random state of a random generator.

    :param key:
        The random generator.
    """


@dispatch
@abstract()
def restore_random_state(key: B.RandomState, state):
    """
    Set the random state of a random generator. Return the new random
    generator with state `state`.

    :param key:
        The random generator.
    :param state:
        The new random state of the random generator.
    """


@dispatch
@abstract()
def create_complex(real: B.Numeric, imag: B.Numeric):
    """
    Return a complex number with the given real and imaginary parts.

    :param real:
        Array of any backend, real part of the complex number.
    :param imag:
        Array of any backend, imaginary part of the complex number.
    """


@dispatch
@abstract()
def complex_like(reference: B.Numeric):
    """
    Return `complex` dtype of a backend based on the reference.

    :param reference:
        Array of any backend.
    """


@dispatch
@abstract()
def is_complex(reference: B.Numeric):
    """
    Return True if reference of `complex` dtype.

    :param reference:
        Array of any backend.
    """


@dispatch
@abstract()
def cumsum(a: B.Numeric, axis=None):
    """
    Return cumulative sum (optionally along axis).

    :param a:
        Array of any backend.
    :param axis:
        As in `numpy.cumsum`.
    """


@dispatch
@abstract()
def qr(x: B.Numeric, mode="reduced"):
    """
    Return a QR decomposition of a matrix x.

    :param x:
        Array of any backend.
    :param mode:
        As in `numpy.linalg.qr`.
    """


@dispatch
@abstract()
def slogdet(x: B.Numeric):
    """
    Return the sign and log-determinant of a matrix x.

    :param x:
        Array of any backend.
    """


@dispatch
@abstract()
def eigvalsh(x: B.Numeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.

    :param x:
        Array of any backend.
    """


@dispatch
@abstract()
def reciprocal_no_nan(x: Union[B.Numeric, spmatrix]):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.

    :param x:
        Array of any backend or `scipy.sparse.spmatrix`.
    """


@dispatch
@abstract()
def complex_conj(x: B.Numeric):
    """
    Return complex conjugate.

    :param x:
        Array of any backend.
    """


@dispatch
@abstract()
def logical_xor(x1: B.Bool, x2: B.Bool):
    """
    Return logical XOR of two arrays.

    :param x1:
        Array of any backend.
    :param x2:
        Array of any backend.
    """


@dispatch
@abstract()
def count_nonzero(x: B.Numeric, axis=None):
    """
    Count non-zero elements in an array.

    :param x:
        Array of any backend and of any shape.
    """


@dispatch
@abstract()
def dtype_bool(reference: B.RandomState):
    """
    Return `bool` dtype of a backend based on the reference.

    :param reference:
        A random state to infer the backend from.
    """


@dispatch
@abstract()
def bool_like(reference: B.Numeric):
    """
    Return the type of the reference if it is of boolean type.
    Otherwise return `bool` dtype of a backend based on the reference.

    :param reference:
        Array of any backend.
    """


def smart_cast(
    dtype: Union[B.Bool, B.Int, B.Float, B.Complex, B.Numeric], x: B.Numeric
):
    """
    Return `x` cast to the `dtype` abstract data type.

    :param dtype:
        An abstract DType of lab, one of `B.Bool`, `B.Int`, `B.Float`,
        `B.Complex`, `B.Numeric`.
    :param x:
        Array of any backend.
    """
    if dtype == B.Bool:
        return B.cast(bool_like(x), x)
    elif dtype == B.Int:
        return B.cast(int_like(x), x)
    elif dtype == B.Float:
        return B.cast(float_like(x), x)
    elif dtype == B.Complex:
        return B.cast(complex_like(x), x)
