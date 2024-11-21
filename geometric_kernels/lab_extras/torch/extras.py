import lab as B
import torch
from beartype.typing import Any, List, Optional
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TorchNumeric]


@dispatch
def take_along_axis(a: Union[_Numeric, B.Numeric], index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    if not torch.is_tensor(a):
        a = torch.tensor(a).to(index.device)  # type: ignore
    return torch.take_along_dim(a, index.long(), axis)  # long is required by torch


@dispatch
def from_numpy(
    a: B.TorchNumeric, b: Union[List, B.Number, B.NPNumeric, B.TorchNumeric]
):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    if not torch.is_tensor(b):
        b = torch.tensor(b.copy()).to(a.device)  # type: ignore
    return b


@dispatch
def trapz(y: B.TorchNumeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the trapezoidal rule.
    """
    return torch.trapz(y, x, dim=axis)


@dispatch
def norm(x: _Numeric, ord: Optional[Any] = None, axis: Optional[int] = None):  # type: ignore
    """
    Matrix or vector norm.
    """
    return torch.linalg.norm(x, ord=ord, dim=axis)


@dispatch
def logspace(start: B.TorchNumeric, stop: B.TorchNumeric, num: int = 50, base: _Numeric = 10.0):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return torch.logspace(start.item(), stop.item(), num, base)


@dispatch
def degree(a: B.TorchNumeric):  # type: ignore
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
    """
    degrees = a.sum(axis=0)  # type: ignore
    return torch.diag(degrees)


@dispatch
def eigenpairs(L: B.TorchNumeric, k: int):
    """
    Obtain the eigenpairs that correspond to the `k` lowest eigenvalues
    of a symmetric positive semi-definite matrix `L`.

    TODO(AR): Replace with torch.lobpcg after sparse matrices are supported
    by torch.
    """
    l, u = torch.linalg.eigh(L)
    return l[:k], u[:, :k]


@dispatch
def set_value(a: B.TorchNumeric, index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = a.clone()
    a[index] = value
    return a


@dispatch
def dtype_double(reference: B.TorchRandomState):  # type: ignore
    """
    Return `double` dtype of a backend based on the reference.
    """
    return torch.double


@dispatch
def float_like(reference: B.TorchNumeric):
    """
    Return the type of the reference if it is a floating point type.
    Otherwise return `double` dtype of a backend based on the reference.
    """
    if torch.is_floating_point(reference):
        return B.dtype(reference)
    else:
        return torch.float64


@dispatch
def dtype_integer(reference: B.TorchRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return torch.int


@dispatch
def int_like(reference: B.TorchNumeric):
    reference_dtype = reference.dtype
    if reference_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return reference_dtype
    else:
        return torch.int32


@dispatch
def get_random_state(key: B.TorchRandomState):
    """
    Return the random state of a random generator.

    :param key: the random generator of type `B.TorchRandomState`.
    """
    return key.get_state()


@dispatch
def restore_random_state(key: B.TorchRandomState, state):
    """
    Set the random state of a random generator. Return the new random
    generator with state `state`.

    :param key:
        The random generator of type `B.TorchRandomState`.
    :param state:
        The new random state of the random generator.
    """
    gen = torch.Generator()
    gen.set_state(state)
    return gen


@dispatch
def create_complex(real: _Numeric, imag: B.TorchNumeric):
    """
    Return a complex number with the given real and imaginary parts using pytorch.

    :param real:
        float, real part of the complex number.
    :param imag:
        float, imaginary part of the complex number.
    """
    complex_num = real + 1j * imag
    return complex_num


@dispatch
def complex_like(reference: B.TorchNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    return B.promote_dtypes(torch.cfloat, reference.dtype)


@dispatch
def is_complex(reference: B.TorchNumeric):
    """
    Return True if reference of `complex` dtype.
    """
    return (B.dtype(reference) == torch.cfloat) or (B.dtype(reference) == torch.cdouble)


@dispatch
def cumsum(x: B.TorchNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return torch.cumsum(x, dim=axis)


@dispatch
def qr(x: B.TorchNumeric, mode="reduced"):
    """
    Return a QR decomposition of a matrix x.
    """
    Q, R = torch.linalg.qr(x, mode=mode)
    return Q, R


@dispatch
def slogdet(x: B.TorchNumeric):
    """
    Return the sign and log-determinant of a matrix x.
    """
    sign, logdet = torch.slogdet(x)
    return sign, logdet


@dispatch
def eigvalsh(x: B.TorchNumeric):
    """
    Compute the eigenvalues of a Hermitian or real symmetric matrix x.
    """
    return torch.linalg.eigvalsh(x)


@dispatch
def reciprocal_no_nan(x: B.TorchNumeric):
    """
    Return element-wise reciprocal (1/x). Whenever x = 0 puts 1/x = 0.
    """
    safe_x = torch.where(x == 0.0, 1.0, x)
    return torch.where(x == 0.0, 0.0, torch.reciprocal(safe_x))


@dispatch
def complex_conj(x: B.TorchNumeric):
    """
    Return complex conjugate
    """
    return torch.conj(x)


@dispatch
def logical_xor(x1: B.TorchNumeric, x2: B.TorchNumeric):
    """
    Return logical XOR of two arrays.
    """
    return torch.logical_xor(x1, x2)


@dispatch
def count_nonzero(x: B.TorchNumeric, axis=None):
    """
    Count non-zero elements in an array.
    """
    return torch.count_nonzero(x, dim=axis)


@dispatch
def dtype_bool(reference: B.TorchRandomState):  # type: ignore
    """
    Return `bool` dtype of a backend based on the reference.
    """
    return torch.bool


@dispatch
def bool_like(reference: B.NPNumeric):
    """
    Return the type of the reference if it is of boolean type.
    Otherwise return `bool` dtype of a backend based on the reference.
    """
    reference_dtype = reference.dtype
    if reference_dtype is torch.bool:
        return reference_dtype
    else:
        return torch.bool
