################################################
  Product Kernels
################################################

Product kernels are not the same as kernels on product spaces, for the latter see :doc:`the respective page <product_spaces>`.
For a brief demonstration of the difference, see the example notebook `on the torus <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Torus.ipynb>`_.


**Warning:** this is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.
You can get by fine without reading this page for almost all use cases, either

* by using the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>` with :class:`ProductDiscreteSpectrumSpace <geometric_kernels.spaces.ProductDiscreteSpectrumSpace>` (only works for discrete spectrum spaces),

* or by using :class:`ProductGeometricKernel <geometric_kernels.kernels.ProductGeometricKernel>`, which can combine any types of kernels while also maintaining a separate lengthscale for each of them, much like ARD kernels in the Euclidean case).

Both ways are described in the example notebook `on the torus <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Torus.ipynb>`_, which is a product of circles. 

=======
Theory
=======

In GeometricKernels, there is a concept of product Matérn kernel (:class:`ProductGeometricKernel <geometric_kernels.kernels.ProductGeometricKernel>`).
It allows you to define a kernel $k$ on a product $\mathcal{M} = \mathcal{M}_1 \times \ldots \times \mathcal{M}_m$ of some other spaces $\mathcal{M}_j$ by taking a product of some kernels $k_j: \mathcal{M}_j \times \mathcal{M}_j \to \mathbb{R}$:
$$
k(x_1, \ldots, x_m)
=
k_1(x_1) \cdot \ldots \cdot k_m(x_m)
.
$$
Each $k_j$ would usually be :class:`ProductGeometricKernel <geometric_kernels.kernels.ProductGeometricKernel>` on spaces $\mathcal{M}_j$, which can be anything: compact manifold, graph, mesh, non-compact symmetric space, even a :doc:`product space <product_kernels>`.

**Importantly**, this allows you to have a separate length scale parameter for each of the factors, enabling, e.g. *automatic relevance determination* (ARD, cf. `Rasmussen and Williams (2006) <https://gaussianprocess.org/gpml/chapters/RW.pdf>`_).

For Matérn kernels, even if $\nu$ and $\kappa$ are the same for all $k_j$, the product kernel turns out to be different from the Matérn kernel on the product space whenever $\nu < \infty$.
If $\nu = \infty$, i.e. in the case of the heat (a.k.a. diffusion, squared exponential, RBF) kernel, the product of kernels with same values of $\kappa$ and the kernel on the product space with this $\kappa$ coincide.
This mirrors the standard Euclidean case.
