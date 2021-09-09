"""
Convenience utilities.
"""
from typing import Any, List

import eagerpy as ep

from geometric_kernels.eagerpy_extras import repeat
from geometric_kernels.types import TensorLike


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
    values = [
        repeat(elements[i : i + 1], "j -> (tile j)", tile=repetitions[i])
        for i in range(len(repetitions))
    ]
    return ep.concatenate(values, axis=0)
