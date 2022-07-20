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
        b = torch.tensor(b)
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
    Given a vector a, return a diagonal matrix with a as main diagonal.
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
