import sys

import lab as B
import scipy
import scipy.sparse as sp
from beartype.typing import Union
from lab import dispatch
from plum import Signature

from .extras import _Numeric

"""
SparseArray defines a lab data type that covers all possible sparse
scipy arrays, so that multiple dispatch works with such arrays.
"""
if sys.version_info[:2] <= (3, 8):
    SparseArray = Union[
        sp.bsr_matrix,
        sp.coo_matrix,
        sp.csc_matrix,
        sp.csr_matrix,
        sp.dia_matrix,
        sp.dok_matrix,
        sp.lil_matrix,
    ]
else:
    SparseArray = Union[
        sp.sparray,
        sp.spmatrix,
    ]


@dispatch
def degree(a: SparseArray):  # type: ignore
    """
    Given an adjacency matrix `a`, return a diagonal matrix
    with the col-sums of `a` as main diagonal - this is the
    degree matrix representing the number of nodes each node
    is connected to.
    """
    d = a.sum(axis=0)  # type: ignore
    return sp.spdiags(d, 0, d.size, d.size)


@dispatch
def eigenpairs(L: Union[SparseArray, _Numeric], k: int):
    """
    Obtain the eigenpairs that correspond to the `k` lowest eigenvalues
    of a symmetric positive semi-definite matrix `L`.
    """
    if sp.issparse(L) and (k == L.shape[0]):
        L = L.toarray()
    if sp.issparse(L):
        return sp.linalg.eigsh(L, k, sigma=1e-8)
    else:
        eigenvalues, eigenvectors = scipy.linalg.eigh(L)
        return (eigenvalues[:k], eigenvectors[:, :k])


@dispatch
def set_value(a: Union[SparseArray, _Numeric], index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = a.copy()
    a[index] = value
    return a


""" Register methods for simple ops for a sparse array. """


def pinv(a: Union[SparseArray]):
    i, j = a.nonzero()
    if not (i == j).all():
        raise NotImplementedError(
            "pinv is not supported for non-diagonal sparse arrays."
        )
    else:
        a = sp.csr_matrix(a.copy())
        a[i, i] = 1 / a[i, i]
        return a


# putting "ignore" here for now, seems like some plum/typing issue
_SparseArray = Signature(SparseArray)  # type: ignore

B.T.register(lambda a: a.T, _SparseArray)
B.shape.register(lambda a: a.shape, _SparseArray)
B.sqrt.register(lambda a: a.sqrt(), _SparseArray)
B.any.register(lambda a: bool((a == True).sum()), _SparseArray)  # noqa

B.linear_algebra.pinv.register(pinv, _SparseArray)
