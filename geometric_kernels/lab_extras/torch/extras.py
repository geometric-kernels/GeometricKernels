from typing import List

import lab as B
import torch
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TorchNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return torch.index_select(a, axis, B.flatten(index))


@dispatch
def from_numpy(
    _: B.TorchNumeric, b: Union[List, B.Number, B.NPNumeric, B.TorchNumeric]
):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    print("HERE!!!")
    return torch.tensor(b)


@dispatch
def swapaxes(a: B.TorchNumeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """
    return torch.transpose(a, axis1, axis2)


@dispatch
def copysign(a: B.TorchNumeric, b: _Numeric) -> B.Numeric:  # type: ignore
    """
    Change the sign of `a` to that of `b`, element-wise.
    """
    return torch.copysign(a, b)


@dispatch
def take_along_last_axis(a: B.TorchNumeric, indices: _Numeric):  # type: ignore
    """
    Takes elements of `a` along the last axis. `indices` must be the same rank (ndim) as
    `a`. Useful in e.g. argsorting and then recovering the sorted array.
    ```

    E.g. for 3d case:
    output[i, j, k] = a[i, j, indices[i, j, k]]
    """
    return torch.gather(a, -1, indices)
