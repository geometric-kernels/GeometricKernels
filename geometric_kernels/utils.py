"""
Convenience utilities.
"""
from typing import Any, List

from eagerpy.norms import l2

from geometric_kernels.types import TensorLike


def l2norm(X: TensorLike) -> TensorLike:
    """
    Returns the norm of the vectors in `X`. The vectors are
    D-dimensional and  stored in the last dimension of `X`.

    :param X: [..., D]
    :return: norm for each element in `X`, [N, 1]

    """
    return l2(X, axis=-1, keepdims=True)
    return tf.reduce_sum(X ** 2, keepdims=True, axis=-1) ** 0.5


def chain(elements: List[Any], repetitions: List[int]) -> TensorLike:
    """
    Repeats each element in `elements` by a certain number of repetitions as
    specified in `repetitions`.  The length of `elements` and `repetitions`
    should match.

    .. code:
        elements = ['a', 'b', 'c']
        repetitions = [2, 1, 3]
        out = chain(elements, repetitions)
        print(out)  # ['a', 'a', 'b', 'c', 'c', 'c']
    """
    return tf.concat(
        values=[tf.repeat(elements[i], r) for i, r in enumerate(repetitions)],
        axis=0,
    )
