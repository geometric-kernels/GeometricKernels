from typing import List

import lab as B
import tensorflow as tf
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
def swapaxes(a: B.TFNumeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """
    perm = list(range(tf.rank(a)))
    perm[axis1] = axis2
    perm[axis2] = axis1

    return tf.transpose(a, perm=perm)


@dispatch
def copysign(a: B.TFNumeric, b: _Numeric) -> B.Numeric:  # type: ignore
    """
    Change the sign of `a` to that of `b`, element-wise.
    """

    return tf.where(tf.equal(tf.math.sign(a), tf.math.sign(b)), -a, a)


@dispatch
def take_along_last_axis(a: B.TFNumeric, indices: _Numeric):  # type: ignore
    """
    Takes elements of `a` along the last axis. `indices` must be the same rank (ndim) as
    `a`. Useful in e.g. argsorting and then recovering the sorted array.
    ```

    E.g. for 3d case:
    output[i, j, k] = a[i, j, indices[i, j, k]]
    """
    return tf.gather(a, indices, batch_dims=-1)
