import jax.numpy as jnp
import lab as B
from beartype.typing import List
from lab import dispatch
from plum import Union, convert

_Numeric = Union[B.Number, B.JAXNumeric]


@dispatch
def take_along_axis(a: Union[_Numeric, B.Numeric], index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    if not isinstance(a, jnp.ndarray):
        a = jnp.array(a)
    return jnp.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.JAXNumeric, b: Union[List, B.NPNumeric, B.Number, B.JAXNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return jnp.array(b)


@dispatch
def trapz(y: B.JAXNumeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the trapezoidal rule.
    """
    return jnp.trapz(y, x, dx, axis)


@dispatch
def logspace(start: B.JAXNumeric, stop: _Numeric, num: int = 50):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return jnp.logspace(start, stop, num)


@dispatch
def degree(a: B.JAXNumeric):  # type: ignore
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
    """
    degrees = a.sum(axis=0)  # type: ignore
    return jnp.diag(degrees)


@dispatch
def eigenpairs(L: B.JAXNumeric, k: int):
    """
    Obtain the eigenpairs that correspond to the `k` lowest eigenvalues
    of a symmetric positive semi-definite matrix `L`.
    """
    l, u = jnp.linalg.eigh(L)
    return l[:k], u[:, :k]


@dispatch
def set_value(a: B.JAXNumeric, index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = a.at[index].set(value)
    return a


@dispatch
def dtype_double(reference: B.JAXRandomState):  # type: ignore
    """
    Return `double` dtype of a backend based on the reference.
    """
    return jnp.float64


@dispatch
def float_like(reference: B.JAXNumeric):
    """
    Return the type of the reference if it is a floating point type.
    Otherwise return `double` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if jnp.issubdtype(reference_dtype, jnp.floating):
        return convert(
            reference_dtype, B.JAXDType
        )  # JAX .dtype returns a NumPy data type. This converts it to a JAX one.
    else:
        return jnp.float64


@dispatch
def dtype_integer(reference: B.JAXRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return jnp.int32


@dispatch
def int_like(reference: B.JAXNumeric):
    reference_dtype = reference.dtype
    if jnp.issubdtype(reference_dtype, jnp.integer):
        return convert(
            reference_dtype, B.JAXDType
        )  # JAX .dtype returns a NumPy data type. This converts it to a JAX one.
    else:
        return jnp.int32


@dispatch
def get_random_state(key: B.JAXRandomState):
    """
    Return the random state of a random generator.

    :param key:
        The random generator of type `B.JAXRandomState`.
    """
    return key


@dispatch
def restore_random_state(key: B.JAXRandomState, state):
    """
    Set the random state of a random generator. Return the new random
    generator with state `state`.

    :param key:
        The random generator of type `B.JAXRandomState`.
    :param state:
        The new random state of the random generator.
    """
    return state


@dispatch
def create_complex(real: _Numeric, imag: B.JAXNumeric):
    """
    Return a complex number with the given real and imaginary parts using jax.

    :param real:
        float, real part of the complex number.
    :param imag:
        float, imaginary part of the complex number.
    """
    complex_num = real + 1j * imag
    return complex_num


@dispatch
def complex_like(reference: B.JAXNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    return B.promote_dtypes(jnp.complex64, reference.dtype)


@dispatch
def is_complex(reference: B.JAXNumeric):
    """
    Return True if reference of `complex` dtype.
    """
    return (B.dtype(reference) == jnp.complex64) or (
        B.dtype(reference) == jnp.complex128
    )


@dispatch
def cumsum(x: B.JAXNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return jnp.cumsum(x, axis=axis)


@dispatch
def qr(x: B.JAXNumeric, mode="reduced"):
    """
    Return a QR decomposition of a matrix x.
    """
    Q, R = jnp.linalg.qr(x, mode=mode)
    return Q, R


@dispatch
def slogdet(x: B.JAXNumeric):
    """
    Return the sign and log-determinant of a matrix x.
    """
    sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


@dispatch
def eigvalsh(x: B.JAXNumeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.
    """
    return jnp.linalg.eigvalsh(x)


@dispatch
def reciprocal_no_nan(x: B.JAXNumeric):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.
    """
    x_is_zero = jnp.equal(x, 0.0)
    safe_x = jnp.where(x_is_zero, 1.0, x)
    return jnp.where(x_is_zero, 0.0, jnp.reciprocal(safe_x))


@dispatch
def complex_conj(x: B.JAXNumeric):
    """
    Return complex conjugate
    """
    return jnp.conj(x)


@dispatch
def logical_xor(x1: B.JAXNumeric, x2: B.JAXNumeric):
    """
    Return logical XOR of two arrays.
    """
    return jnp.logical_xor(x1, x2)


@dispatch
def count_nonzero(x: B.JAXNumeric, axis=None):
    """
    Count non-zero elements in an array.
    """
    return jnp.count_nonzero(x, axis=axis)


@dispatch
def dtype_bool(reference: B.JAXRandomState):  # type: ignore
    """
    Return `bool` dtype of a backend based on the reference.
    """
    return jnp.bool_


@dispatch
def bool_like(reference: B.JAXRandomState):
    """
    Return the type of the reference if it is of boolean type.
    Otherwise return `bool` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if jnp.issubdtype(reference_dtype, jnp.bool_):
        return convert(
            reference_dtype, B.JAXDType
        )  # JAX .dtype returns a NumPy data type. This converts it to a JAX one.
    else:
        return jnp.bool_
