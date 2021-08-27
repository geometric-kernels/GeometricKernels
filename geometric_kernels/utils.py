from typing import Any, List

import tensorflow as tf

from geometric_kernels.types import TensorLike


def l2norm(X: TensorLike) -> TensorLike:
    """
    Returns the norm of the vectors in `X`. The vectors are
    D-dimensional and  stored in the last dimension of `X`.

    :param X: [..., D]
    :return: norm for each element in `X`, [N, 1]

    """
    return tf.reduce_sum(X ** 2, keepdims=True, axis=-1) ** 0.5


def chain(elements: List[Any], repetitions: List[int]) -> List[Any]:
    return tf.concat(
        values=[tf.repeat(elements[i], r) for i, r in enumerate(repetitions)],
        axis=0,
    )
