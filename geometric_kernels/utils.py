"""
Convenience utilities.
"""
from typing import Any, List, Type

import einops
import lab as B
from plum import Union


class OptionalMeta(type):
    def __getitem__(cls, args: Type):
        return Union[(None,) + (args,)]


class Optional(metaclass=OptionalMeta):
    pass


def chain(elements: List[Any], repetitions: List[int]) -> B.Numeric:
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
        einops.repeat(elements[i : i + 1], "j -> (tile j)", tile=repetitions[i])
        for i in range(len(repetitions))
    ]
    return B.concat(*values, axis=0)
