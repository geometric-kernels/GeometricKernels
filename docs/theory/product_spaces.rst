################################################
  Kernels on Product Spaces
################################################

To work on a product space (i.e. a space which is itself a product of spaces), one can use either *product Matérn kernels* or *Matérn kernels on product spaces*.
These are generally not the same.
On this page we discuss the Matérn kernels on product spaces.
For a discussion on product Matérn kernels, see :doc:`the respective page <product_kernels>`.
For a brief demonstration of the difference, see the example notebook `on the torus <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Torus.ipynb>`_.

**Warning:** this is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.
you can get by fine without reading this page for almost all use cases involving product spaces, either

* by using the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>` with :class:`ProductDiscreteSpectrumSpace <geometric_kernels.spaces.ProductDiscreteSpectrumSpace>` (only works for discrete spectrum spaces),

* or by using :class:`ProductGeometricKernel <geometric_kernels.kernels.ProductGeometricKernel>`, which can combine any types of kernels while also maintaining a separate lengthscale for each of them, much like ARD kernels in the Euclidean case).

Both ways are described in the example notebook `on the torus <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Torus.ipynb>`_, which is a product of circles. 

=======
Theory
=======

This builds on the general :doc:`theory on compact manifolds <compact>`.

Assume that $\mathcal{M}$ is a product of compact Riemannian manifolds $\mathcal{M}_j$, i.e. $\mathcal{M} = \mathcal{M}_1 \times \ldots \times \mathcal{M}_m$.
You can consider other discrete spectrum spaces in place of the manifolds, like graphs or meshes, just as well.
Here we concentrate on manifolds for simplicity.

Matérn kernel on $\mathcal{M}$ is determined by the *eigenvalues* $\lambda_n \geq 0$ and *eigenfunctions* $f_n(\cdot)$ of the *Laplace–Beltrami operator* $\Delta_{\mathcal{M}}$ on $\mathcal{M}$.

The **key idea** is that $\lambda_n, f_n$ can be obtained from the eigenvalues and eigenfunctions on $\mathcal{M}_j$ therefore allowing to build Matérn kernels on the product space $\mathcal{M}$ from the components you would use to build Matérn kernels on the separate factors $\mathcal{M}_j$.

In fact, all eigenfunctions on $\mathcal{M}$ have form
$$
f_n(x_1, \ldots, x_m)
=
f^{(1)}_{n_1(n)}(x_1) \cdot \ldots \cdot f^{(m)}_{n_m(n)}(x_m)
$$
where $f^{(j)}_{i}(\cdot)$ is the $i$-th eigenfunction on $\mathcal{M}_j$.
What is more,
$$
\Delta_{\mathcal{M}} f_n = \lambda_n f_n
\qquad
\text{for}
\qquad
\lambda_n = \lambda^{(1)}_{n_1(n)} + \ldots + \lambda^{(m)}_{n_m(n)}
$$
where $\lambda^{(j)}_{i}$ is the $i$-th eigenvalue on $\mathcal{M}_j$.
See e.g. page 48 of the `"Analysis on Manifolds via the Laplacian" by Yaiza Canzani <https://www.math.mcgill.ca/toth/spectral%20geometry.pdf>`_.
