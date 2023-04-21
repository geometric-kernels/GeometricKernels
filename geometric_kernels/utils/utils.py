"""
Convenience utilities.
"""
import inspect
from typing import List, Type

import einops
import lab as B
from plum import Union

from geometric_kernels.lab_extras import get_random_state, restore_random_state


class OptionalMeta(type):
    def __getitem__(cls, args: Type):
        return Union[(None,) + (args,)]


class Optional(metaclass=OptionalMeta):
    pass


def chain(elements: B.Numeric, repetitions: List[int]) -> B.Numeric:
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


def make_deterministic(f, key):
    """
    Returns a deterministic version of a function that uses a random number generator.

    Parameters
    ----------
    f : function
        The function to make deterministic.
    key : Any
        The key used to generate the random state.

    Returns
    -------
    function
        A deterministic version of the input function.

    Notes
    -----
    This function assumes that the input function has a 'key' argument or keyword-only argument that is used to generate random numbers. Otherwise, the function is returned as is.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax import random

    >>> def f(x, y, key=random.PRNGKey(0)):
            return jnp.dot(x, y) + random.normal(key=key)

    >>> f_deterministic = make_deterministic(f, random.PRNGKey(42))
    >>> x = jnp.array([1., 2., 3.])
    >>> y = jnp.array([4., 5., 6.])
    >>> f(x, y)
    DeviceArray(10.832865, dtype=float32)
    >>> f_deterministic(x, y)
    DeviceArray(10.832865, dtype=float32)
    """
    f_argspec = inspect.getfullargspec(f)
    f_varnames = f_argspec.args
    key_argtype = None
    if "key" in f_varnames:
        key_argtype = "pos"
        key_position = f_varnames.index("key")
    elif "key" in f_argspec.kwonlyargs:
        key_argtype = "kwonly"

    if key_argtype is None:
        return f  # already deterministic

    saved_random_state = get_random_state(key)

    def deterministic_f(*args, **kwargs):
        restored_key = restore_random_state(key, saved_random_state)
        if key_argtype == "kwonly":
            kwargs["key"] = restored_key
            new_args = args
        elif key_argtype == "pos":
            new_args = args[:key_position] + (restored_key,) + args[key_position:]
        else:
            raise ValueError("Unknown key_argtype %s" % key_argtype)
        return f(*new_args, **kwargs)

    return deterministic_f
