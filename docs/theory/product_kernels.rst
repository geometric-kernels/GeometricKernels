################################################
  Product Kernels
################################################

To work on a product space (i.e. a space which is itself a product of spaces), one can use either *product Matérn kernels* or *Matérn kernels on product spaces*.
These are generally not the same.
On this page we discuss the product Matérn kernels.
For a discussion on Matérn kernels on product spaces, see :doc:`the respective page <product_spaces>`.
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

In GeometricKernels, there is a concept of product Matérn kernel (:class:`~.kernels.ProductGeometricKernel`).
It allows you to define a kernel $k$ on a product $\mathcal{M} = \mathcal{M}_1 \times \ldots \times \mathcal{M}_m$ of some other spaces $\mathcal{M}_i$ by taking a product of some kernels $k_i: \mathcal{M}_i \times \mathcal{M}_i \to \mathbb{R}$:
$$
k((x_1, \ldots, x_m), (x_1', \ldots, x_m'))
=
k_1(x_1, x_1') \cdot \ldots \cdot k_m(x_m, x_m')
.
$$
Each $k_i$ would usually be :class:`~.kernels.MaternGeometricKernel` on spaces $\mathcal{M}_i$, which can be anything: compact manifolds, graphs, meshes, non-compact symmetric spaces, etc.

**Importantly**, this allows you to have a separate length scale parameter for each of the factors, enabling, e.g. *automatic relevance determination* (ARD, cf. :cite:t:`rasmussen2006`).

For Matérn kernels, even if $\nu$ and $\kappa$ are the same for all $k_j$, the product kernel turns out to be different from the Matérn kernel on the product space whenever $\nu < \infty$.
If $\nu = \infty$, i.e. in the case of the heat kernel (a.k.a. diffusion kernel, or squared exponential kernel, or RBF kernel), the product of kernels with same values of $\kappa$ coincides with the kernel on the product space with this same $\kappa$.
This mirrors the standard Euclidean case.
