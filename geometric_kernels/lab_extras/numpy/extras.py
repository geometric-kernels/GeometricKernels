import lab as B
import numpy as np
from beartype.typing import Any, List, Optional
from lab import dispatch
from plum import Union
from scipy.sparse import spmatrix

_Numeric = Union[B.Number, B.NPNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return np.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.NPNumeric, b: Union[List, B.NPNumeric, B.Number]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return np.array(b)


@dispatch
def trapz(y: _Numeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the composite trapezoidal rule.
    """
    return np.trapz(y, x, dx, axis)


@dispatch
def norm(x: _Numeric, ord: Optional[Any] = None, axis: Optional[int] = None):  # type: ignore
    """
    Matrix or vector norm.
    """
    return np.linalg.norm(x, ord=ord, axis=axis)


@dispatch
def logspace(start: _Numeric, stop: _Numeric, num: int = 50, base: _Numeric = 50.0):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return np.logspace(start, stop, num, base=base)


@dispatch
def degree(a: _Numeric):  # type: ignore
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
    """
    degrees = a.sum(axis=0)  # type: ignore
    return np.diag(degrees)


@dispatch
def dtype_double(reference: B.NPRandomState):  # type: ignore
    """
    Return `double` dtype of a backend based on the reference.
    """
    return np.float64


@dispatch
def float_like(reference: B.NPNumeric):
    """
    Return the type of the reference if it is a floating point type.
    Otherwise return `double` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if np.issubdtype(reference_dtype, np.floating):
        return reference_dtype
    else:
        return np.float64


@dispatch
def dtype_integer(reference: B.NPRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return np.int32


@dispatch
def int_like(reference: B.NPNumeric):
    reference_dtype = reference.dtype
    if np.issubdtype(reference_dtype, np.integer):
        return reference_dtype
    else:
        return np.int32


@dispatch
def get_random_state(key: B.NPRandomState):
    """
    Return the random state of a random generator.

    :param key:
        The random generator of type `B.NPRandomState`.
    """
    return key.get_state()


@dispatch
def restore_random_state(key: B.NPRandomState, state):
    """
    Set the random state of a random generator. Return the new random
    generator with state `state`.

    :param key:
        The random generator of type `B.NPRandomState`.
    :param state:
        The new random state of the random generator.
    """
    gen = np.random.RandomState()
    gen.set_state(state)
    return gen


@dispatch
def create_complex(real: _Numeric, imag: _Numeric):
    """
    Return a complex number with the given real and imaginary parts using numpy.

    :param real:
        float, real part of the complex number.
    :param imag:
        float, imaginary part of the complex number.
    """
    complex_num = real + 1j * imag
    return complex_num


@dispatch
def complex_like(reference: B.NPNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    return B.promote_dtypes(np.complex64, reference.dtype)


@dispatch
def is_complex(reference: B.NPNumeric):
    """
    Return True if reference of `complex` dtype.
    """
    return (B.dtype(reference) == np.complex64) or (B.dtype(reference) == np.complex128)


@dispatch
def cumsum(a: _Numeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return np.cumsum(a, axis=axis)


@dispatch
def qr(x: _Numeric, mode="reduced"):
    """
    Return a QR decomposition of a matrix x.
    """
    Q, R = np.linalg.qr(x, mode=mode)
    return Q, R


@dispatch
def slogdet(x: _Numeric):
    """
    Return the sign and log-determinant of a matrix x.
    """
    sign, logdet = np.linalg.slogdet(x)
    return sign, logdet


@dispatch
def eigvalsh(x: _Numeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.
    """
    return np.linalg.eigvalsh(x, UPLO="U")


@dispatch
def reciprocal_no_nan(x: B.NPNumeric):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.
    """
    x_is_zero = np.equal(x, 0.0)
    safe_x = np.where(x_is_zero, 1.0, x)
    return np.where(x_is_zero, 0.0, np.reciprocal(safe_x))


@dispatch  # type: ignore[no-redef]
def reciprocal_no_nan(x: spmatrix):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.
    """
    return x._with_data(reciprocal_no_nan(x._deduped_data().copy()), copy=True)


@dispatch
def complex_conj(x: B.NPNumeric):
    """
    Return complex conjugate
    """
    return np.conjugate(x)


@dispatch
def logical_xor(x1: B.NPNumeric, x2: B.NPNumeric):
    """
    Return logical XOR of two arrays.
    """
    return np.logical_xor(x1, x2)


@dispatch
def count_nonzero(x: B.NPNumeric, axis=None):
    """
    Count non-zero elements in an array.
    """
    return np.count_nonzero(x, axis=axis)


@dispatch
def dtype_bool(reference: B.NPRandomState):  # type: ignore
    """
    Return `bool` dtype of a backend based on the reference.
    """
    return np.bool_


@dispatch
def bool_like(reference: B.NPNumeric):
    """
    Return the type of the reference if it is of boolean type.
    Otherwise return `bool` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if np.issubdtype(reference_dtype, np.bool_):
        return reference_dtype
    else:
        return np.bool_
