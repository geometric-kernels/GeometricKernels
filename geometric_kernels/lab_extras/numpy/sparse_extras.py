import scipy.sparse as sp
from lab import dispatch
from plum import Union

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


@dispatch
def degree(a: SparseArray):  # type: ignore
    """
    Diagonal matrix with x as main diagonal.
    """
    d = a.sum(axis=0)  # type: ignore
    return sp.spdiags(d, 0, d.size, d.size)
