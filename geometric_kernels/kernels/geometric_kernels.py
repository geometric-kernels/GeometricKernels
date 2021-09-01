"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Optional

import eagerpy as ep
import numpy as np

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces import Mesh
from geometric_kernels.types import Parameter, TensorLike
from geometric_kernels.utils import cast_to_int, take_along_axis


class MeshKernel(BaseGeometricKernel):
    """
    Geometric kernel on a Mesh
    """

    def __init__(
        self,
        space: Mesh,
        nu: float,
        truncation_level: int,
    ):
        super().__init__(space)
        self.truncation_level = truncation_level
        self.nu = nu
        self._eigenfunctions: Optional[Callable[[TensorLike], TensorLike]] = None

    def spectrum(self, s: TensorLike, lengthscale: Parameter):
        """
        Matern or RBF spectrum evaluated as `s`. Depends on the
        `lengthscale` parameters.
        """

        # cast `lengthscale` to eagerpy
        lengthscale = ep.astensor(lengthscale)
        # cast `s` to the same backend as `lengthscale`
        s = ep.from_numpy(lengthscale, s).astype(lengthscale.dtype)

        def spectrum_rbf():

            # the backend should be the same as `lengthscale`
            return ep.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))

        def spectrum_matern():
            power = -self.nu - self.space.dim / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            # the backend should be the same as `lengthscale`
            return ep.astensor(base ** power)

        if self.nu == np.inf:
            return spectrum_rbf()
        elif self.nu > 0:
            return spectrum_matern()
        else:
            raise NotImplementedError

    def eigenfunctions(self, **__parameters) -> Callable:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        assert "lengthscale" in __parameters
        lengthscale = ep.astensor(__parameters["lengthscale"])

        class _EigenFunctions:
            """
            Converts the array of eigenvectors to a callable objects,
            The inputs are given by the indices.
            """

            def __init__(self, eigenvectors):
                # cast eigenvectors to the same backend as lengthscale
                self.eigenvectors = ep.from_numpy(lengthscale, eigenvectors).astype(
                    lengthscale.dtype
                )

            def __call__(self, indices: TensorLike) -> TensorLike:
                """
                Selects N locations from the  eigenvectors.

                :param indices: indices [N, 1]
                :return: [N, L]
                """
                assert len(indices.shape) == 2
                assert indices.shape[-1] == 1

                # cast indices to whatever eigenvectors have as a backend
                indices = cast_to_int(ep.astensor(indices))  # [I, 1]

                # This is a very hacky way of taking along 0'th axis.
                # For some reason eagerpy does not take along axis other than last.
                Phi = take_along_axis(self.eigenvectors, indices, axis=0)
                return Phi

        if self._eigenfunctions is None:
            eigenvectors = self.space.get_eigenfunctions(self.truncation_level)  # [Nv, L]
            self._eigenfunctions = _EigenFunctions(eigenvectors)

        return self._eigenfunctions

    def eigenvalues(self, **parameters) -> TensorLike:
        """
        Eigenvalues of the kernel.

        :return: [L, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(self.truncation_level)  # [L, 1]
        return self.spectrum(eigenvalues_laplacian ** 0.5, lengthscale=parameters["lengthscale"])

    def K(self, X, X2=None, **parameters):
        """Compute the mesh kernel via Laplace eigendecomposition"""
        Phi_X = self.eigenfunctions(**parameters)(X)  # [N, L]
        if X2 is None:
            Phi_X2 = Phi_X
        else:
            Phi_X2 = self.eigenfunctions(**parameters)(X2)  # [N2, L]

        coeffs = self.eigenvalues(**parameters)  # [L, 1]

        Kxx = ep.matmul(coeffs.T * Phi_X, Phi_X2.T)
        return Kxx.raw

    def K_diag(self, X, **parameters):
        Phi_X = self.eigenfunctions(**parameters)(X)  # [N, L]
        coeffs = self.eigenvalues(**parameters)  # [L, 1]

        Kx = ep.sum(coeffs.T * Phi_X ** 2, axis=1)  # [N, ]
        return Kx.raw
