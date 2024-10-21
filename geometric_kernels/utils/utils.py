"""
Convenience utilities.
"""

import inspect
import sys
from contextlib import contextmanager
from importlib import resources as impresources
from itertools import combinations

import einops
import lab as B
import numpy as np
from beartype.typing import Callable, Generator, List, Set, Tuple

from geometric_kernels import resources
from geometric_kernels.lab_extras import (
    count_nonzero,
    get_random_state,
    logical_xor,
    restore_random_state,
)


def chain(elements: B.Numeric, repetitions: List[int]) -> B.Numeric:
    """
    Repeats each element in `elements` by a certain number of repetitions as
    specified in `repetitions`.  The length of `elements` and `repetitions`
    should match.

    :param elements:
        An [N,]-shaped array of elements to repeat.
    :param repetitions:
        A list specifying the number of types to repeat each of the elements in
        `elements`. The length of `repetitions` should be equal to N.

    :return:
        An [M,]-shaped array.

    EXAMPLE:

    .. code-block:: python

        elements = np.array([1, 2, 3])
        repetitions = [2, 1, 3]
        out = chain(elements, repetitions)
        print(out)  # [1, 1, 2, 3, 3, 3]
    """
    values = [
        einops.repeat(elements[i : i + 1], "j -> (tile j)", tile=repetitions[i])
        for i in range(len(repetitions))
    ]
    return B.concat(*values, axis=0)


def make_deterministic(f: Callable, key: B.RandomState) -> Callable:
    """
    Returns a deterministic version of a function that uses a random
    number generator.

    :param f:
        The function to make deterministic.
    :param key:
        The key used to generate the random state.

    :return:
        A function representing the deterministic version of the input function.

        .. note::
           This function assumes that the input function has a 'key' argument
           or keyword-only argument that is used to generate random numbers.
           Otherwise, the function is returned as is.

    EXAMPLE:

    .. code-block:: python

        key = tf.random.Generator.from_seed(1234)
        feature_map = default_feature_map(kernel=base_kernel)
        sample_paths = make_deterministic(sampler(feature_map), key)
        _, ys_train  = sample_paths(xs_train, params)
        key, ys_test = sample_paths(xs_test,  params)
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

    if hasattr(f, "__name__"):
        f_name = f.__name__
    elif hasattr(f, "__class__") and hasattr(f.__class__, "__name__"):
        f_name = f.__class__.__name__
    else:
        f_name = "<anonymous>"

    if hasattr(f, "__doc__"):
        f_doc = f.__doc__
    else:
        f_doc = ""

    new_docstring = f"""
        This is a deterministic version of the function {f_name}.

        The original docstring follows.

        {f_doc}
        """
    deterministic_f.__doc__ = new_docstring

    return deterministic_f


def ordered_pairwise_differences(X: B.Numeric) -> B.Numeric:
    """
    Compute the ordered pairwise differences between elements of a vector.

    :param X:
        A [..., D]-shaped array, a batch of D-dimensional vectors.

    :return:
        A [..., C]-shaped array, where C = D*(D-1)//2, containing the ordered
        pairwise differences between the elements of X. That is, the array
        containing differences X[...,i] - X[...,j] for all i < j.
    """
    diffX = B.expand_dims(X, -2) - B.expand_dims(X, -1)  # [B, D, D]
    # diffX[i, j] = X[j] - X[i]
    # lower triangle is i > j
    # so, lower triangle is X[k] - X[l] with k < l

    diffX = B.tril_to_vec(diffX, offset=-1)  # don't take the diagonal

    return diffX


def fixed_length_partitions(  # noqa: C901
    n: int, L: int
) -> Generator[List[int], None, None]:
    """
    A generator for integer partitions of n into L parts, in colex order.

    :param n:
        The number to partition.
    :param L:
        Size of partitions.

    Developed by D. Eppstein in 2005, taken from
    https://www.ics.uci.edu/~eppstein/PADS/IntegerPartitions.py
    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it to Hindenburg, 1779.
    """

    # guard against special cases
    if L == 0:
        if n == 0:
            yield []
        return
    if L == 1:
        if n > 0:
            yield [n]
        return
    if n < L:
        return

    partition = [n - L + 1] + (L - 1) * [1]
    while True:
        yield partition.copy()
        if partition[0] - 1 > partition[1]:
            partition[0] -= 1
            partition[1] += 1
            continue
        j = 2
        s = partition[0] + partition[1] - 1
        while j < L and partition[j] >= partition[0] - 1:
            s += partition[j]
            j += 1
        if j >= L:
            return
        partition[j] = x = partition[j] + 1
        j -= 1
        while j > 0:
            partition[j] = x
            s -= x
            j -= 1
        partition[0] = s


def partition_dominance_cone(partition: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
    """
    Calculates partitions dominated by a given one and having the same number
    of parts (including the zero parts of the original).

    :param partition:
        A partition.

    :return:
        A set of partitions.
    """
    cone = {partition}
    new_partitions: Set[Tuple[int, ...]] = {(0,)}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    for j in range(i + 1, len(partition)):
                        if (
                            partition[i] > partition[j] + 1
                            and partition[j] < partition[j - 1]
                        ):
                            new_partition = list(partition)
                            new_partition[i] -= 1
                            new_partition[j] += 1
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


def partition_dominance_or_subpartition_cone(
    partition: Tuple[int, ...]
) -> Set[Tuple[int, ...]]:
    """
    Calculates subpartitions and partitions dominated by a given one and having
    the same number of parts (including zero parts of the original).

    :param partition:
        A partition.

    :return:
        A set of partitions.
    """
    cone = {partition}
    new_partitions: Set[Tuple[int, ...]] = {(0,)}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    new_partition = list(partition)
                    new_partition[i] -= 1
                    new_partition = tuple(new_partition)
                    if new_partition not in cone:
                        new_partitions.add(new_partition)
                    for j in range(i + 1, len(partition)):
                        if (
                            partition[i] > partition[j] + 1
                            and partition[j] < partition[j - 1]
                        ):
                            new_partition = list(partition)  # type: ignore[assignment]
                            new_partition[i] -= 1  # type: ignore[index]
                            new_partition[j] += 1  # type: ignore[index]
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


@contextmanager
def get_resource_file_path(filename: str):
    """
    A contextmanager wrapper around `impresources` that supports both
    Python>=3.9 and Python==3.8 with a unified interface.

    :param filename:
        The name of the file.
    """
    if sys.version_info >= (3, 9):
        with impresources.as_file(impresources.files(resources) / filename) as path:
            yield path
    else:
        with impresources.path(resources, filename) as path:
            yield path


def hamming_distance(x1: B.Bool, x2: B.Bool):
    """
    Hamming distance between two batches of boolean vectors.

    :param x1:
        Array of any backend, of shape [N, D].
    :param x2:
        Array of any backend, of shape [M, D].

    :return:
        An array of shape [N, M] whose entry n, m contains the Hamming distance
        between x1[n, :] and x2[m, :]. It is of the same backend as x1 and x2.
    """
    return count_nonzero(logical_xor(x1[:, None, :], x2[None, :, :]), axis=-1)


def log_binomial(n: B.Int, k: B.Int) -> B.Float:
    """
    Compute the logarithm of the binomial coefficient.

    :param n:
        The number of elements in the set.
    :param k:
        The number of elements to choose.

    :return:
        The logarithm of the binomial coefficient binom(n, k).
    """
    assert B.all(0 <= k <= n)

    return B.loggamma(n + 1) - B.loggamma(k + 1) - B.loggamma(n - k + 1)


def binary_vectors_and_subsets(d: int):
    r"""
    Generates all possible binary vectors of size d and all possible subsets of
    the set $\{0, .., d-1\}$ as a byproduct.

    :param d:
        The dimension of binary vectors and the size of the set to take subsets of.

    :return:
        A tuple (x, combs), where x is a matrix of size (2**d, d) whose rows
        are all possible binary vectors of size d, and combs is a list of all
        possible subsets of the set $\{0, .., d-1\}$, each subset being
        represented by a list of integers itself.
    """
    x = np.zeros((2**d, d), dtype=bool)
    combs = []
    i = 0
    for level in range(d + 1):
        for cur_combination in combinations(range(d), level):
            indices = list(cur_combination)
            combs.append(indices)
            x[i, indices] = 1
            i += 1

    return x, combs
