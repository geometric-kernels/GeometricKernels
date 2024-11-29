################################################
  Kernels on Product Spaces
################################################

To work on a product space (i.e. a space which is itself a product of spaces), one can use either *product Matérn kernels* or *Matérn kernels on product spaces*.
These are generally not the same.
On this page we discuss the Matérn kernels on product spaces.
For a discussion on product Matérn kernels, see :doc:`the respective page <product_kernels>`.
For a brief demonstration of the difference, see the example notebook :doc:`on the torus </examples/Torus>`.

.. warning::
    This is optional material meant to explain the basic theory and based mainly on :cite:t:`borovitskiy2020`.

    You can get by fine without reading this page for almost all use cases involving product spaces, either

    * by using the standard :class:`~.kernels.MaternGeometricKernel` with :class:`~.spaces.ProductDiscreteSpectrumSpace` (only works for discrete spectrum spaces),

    * or by using :class:`~.kernels.ProductGeometricKernel`, which can combine any types of kernels while also maintaining a separate lengthscale for each of them, much like ARD kernels in the Euclidean case.

    Both ways are described in the example notebook :doc:`on the torus </examples/Torus>`, which is a product of circles.

=======
Theory
=======

This builds on the general :doc:`theory on compact manifolds <compact>`.

Assume that $\mathcal{M}$ is a product of compact Riemannian manifolds $\mathcal{M}_s$, i.e. $\mathcal{M} = \mathcal{M}_1 \times \ldots \times \mathcal{M}_S$.
You can consider other discrete spectrum spaces in place of the manifolds, like graphs or meshes, just as well.
Here we concentrate on manifolds for simplicity of exposition.

Matérn kernels on $\mathcal{M}$ are determined by the *eigenvalues* $\lambda_j \geq 0$ and *eigenfunctions* $f_j(\cdot)$ of the minus *Laplacian* $-\Delta_{\mathcal{M}}$ on $\mathcal{M}$.

The **key idea** is that $\lambda_j, f_j$ can be obtained from the eigenvalues and eigenfunctions on $\mathcal{M}_s$ therefore allowing to build Matérn kernels on the product space $\mathcal{M}$ from the components you would use to build Matérn kernels on the separate factors $\mathcal{M}_s$.

In fact, all eigenfunctions on $\mathcal{M}$ have form
$$
f_j(x_1, \ldots, x_S)
=
f^{(1)}_{j_1(j)}(x_1) \cdot \ldots \cdot f^{(S)}_{j_S(j)}(x_S)
$$
where $f^{(s)}_{j}(\cdot)$ is the $j$-th eigenfunction on $\mathcal{M}_s$.
What is more,
$$
\Delta_{\mathcal{M}} f_j = \lambda_j f_j
\qquad
\text{for}
\qquad
\lambda_j = \lambda^{(1)}_{j_1(j)} + \ldots + \lambda^{(S)}_{j_S(j)}
$$
where $\lambda^{(s)}_{j}$ is the $j$-th eigenvalue on $\mathcal{M}_s$.
See, e.g., page 48 of the :cite:t:`canzani2013`.

.. note::
    The *levels* (see :class:`here <.kernels.MaternKarhunenLoeveKernel>` and :class:`here <.eigenfunctions.Eigenfunctions>`) on factors define levels on the product space, in the same fashion as individual eigenfunctions and eigenvalues on the factors define their counterparts on the product space. In practice we operate on levels rather than on individual eigenpairs.
