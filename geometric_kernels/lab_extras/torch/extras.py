from typing import Any, List, Optional

import lab as B
import torch
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
    return torch.index_select(a, axis, B.flatten(index))


@dispatch
def from_numpy(
    _: B.TorchNumeric, b: Union[List, B.Number, B.NPNumeric, B.TorchNumeric]
):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    if not torch.is_tensor(b):
        b = torch.tensor(b.copy())
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
    Obtain the k highest eigenpairs of a symmetric PSD matrix L.
    TODO(AR): Replace with torch.lobpcg after sparse matrices are supported by torch.
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
def dtype_integer(reference: B.TorchRandomState):  # type: ignore
    """
    Return `int` dtype of a backend based on the reference.
    """
    return torch.int


@dispatch
def get_random_state(key: B.TorchRandomState):
    """
    Return the random state of a random generator.

    Parameters
    ----------
    key : B.TorchRandomState
        The key used to generate the random state.

    Returns
    -------
    Any
        The random state of the random generator.
    """
    return key.get_state()


@dispatch
def restore_random_state(key: B.TorchRandomState, state):
    """
    Set the random state of a random generator.

    Parameters
    ----------
    key : B.TorchRandomState
        The random generator.
    state : Any
        The new random state of the random generator.

    Returns
    -------
    Any
       The new random generator with state `state`.
    """
    gen = torch.Generator()
    gen.set_state(state)
    return gen


@dispatch
def create_complex(real: _Numeric, imag: B.TorchNumeric):
    """
    Returns a complex number with the given real and imaginary parts using pytorch.

    Args:
    - real: float, real part of the complex number.
    - imag: float, imaginary part of the complex number.

    Returns:
    - complex_num: complex, a complex number with the given real and imaginary parts.
    """
    complex_num = real + 1j * imag
    return complex_num


@dispatch
def dtype_complex(reference: B.TorchNumeric):
    """
    Return `complex` dtype of a backend based on the reference.
    """
    if B.dtype(reference) == torch.float:
        return torch.cfloat
    else:
        return torch.cdouble


@dispatch
def cumsum(x: B.TorchNumeric, axis=None):
    """
    Return cumulative sum (optionally along axis)
    """
    return torch.cumsum(x, dim=axis)


@dispatch
def qr(x: B.TorchNumeric):
    """
    Return a QR decomposition of a matrix x.
    """
    Q, R = torch.qr(x)
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
