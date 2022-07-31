import lab as B
import scipy.sparse as sp
from lab import dispatch
from plum import Signature, Union

from .extras import _Numeric

"""
SparseArray defines a lab data type that covers all possible sparse
scipy arrays, so that multiple dispatch works with such arrays.
"""
SparseArray = Union(
    sp.bsr.bsr_matrix,
    sp.coo.coo_matrix,
    sp.csc.csc_matrix,
    sp.csr.csr_matrix,
    sp.dia.dia_matrix,
    sp.dok.dok_matrix,
    sp.lil.lil_matrix,
    alias="SparseArray",
)

_SparseArraySign = Signature(SparseArray)


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
    Obtain the k highest eigenpairs of a symmetric PSD matrix L.
    """
    if sp.issparse(L) and (k == L.shape[0]):
        L = L.toarray()
    return sp.linalg.eigsh(L, k, sigma=1e-8)


@dispatch
def set_value(a: Union[SparseArray, _Numeric], index: int, value: float):
    """
    Set a[index] = value.
    This operation is not done in place and a new array is returned.
    """
    a = a.copy()
    a[index] = value
    return a


def sparse_transpose(a):
    return a.T


def sparse_shape(a):
    return a.shape


def sparse_any(a):
    return bool((a == True).sum())  # noqa


""" Register methods for the shape, transpose and any of a sparse array. """
B.T.register(_SparseArraySign, sparse_transpose)
B.shape.register(_SparseArraySign, sparse_shape)
B.any.register(_SparseArraySign, sparse_any)
