# CHANGELOG

## v0.2 - 21.04.2024
 New geometric kernel that *just works*, `kernels.MaternGeometricKernel`. Relies on *(hopefully)* sensible defaults we defined. Mostly by @stoprightthere.

New spaces, based on Azangulov et al. ([2022](https://arxiv.org/abs/2208.14960), [2023](https://arxiv.org/abs/2301.13088)), mostly by @imbirik and @stoprightthere:
- hyperbolic spaces $\mathbb{H}_n$ in `spaces.Hyperbolic`,
- manifolds of symmetric positive definite matrices $\mathrm{SPD}(n)$ endowed with the affine-invariant Riemannian metric in `spaces.SymmetricPositiveDefiniteMatrices`,
- special orthogonal groups $\mathrm{SO}(n)$ in `spaces.SpecialOrthogonal`.
- special unitary groups $\mathrm{SU}(n)$ in `spaces.SpecialUnitary`.

New package `geometric_kernels.feature_maps` for (approximate) finite-dimensional feature maps. Mostly by @stoprightthere.

New small package `geometric_kernels.sampling` for efficient sampling from geometric Gaussian process priors. Based on the (approximate) finite-dimensional feature maps. Mostly by @stoprightthere.

Examples/Tutorials improvements, mostly by @vabor112:
- new Jupyter notebooks `Graph.ipynb`, `Hyperbolic.ipynb`, `Hypersphere.ipynb`, `Mesh.ipynb`, `SPD.ipynb`, `SpecialOrthogonal.ipynb`, `SpecialUnitary.ipynb`, `Torus.ipynb` featuring tutorials on all the spaces in the library,
- new Jupyter notebooks `backends/JAX_Graph.ipynb`, `backends/PyTorch_Graph.ipynb`, `backends/TensorFlow_Graph.ipynb` showcasing how to use all the backends supported by the library,
- new Jupyter notebooks `frontends/GPflow.ipynb`, `frontends/GPJax.ipynb`, `frontends/GPyTorch.ipynb` showcasing how to use all the frontends supported by the library,
- other notebooks updated and grouped together in `other/` folder.


Documentation improvements, mostly by @vabor112:
- all docstrings throughout the library revised,
- added new documentation pages describing the basic theoretical concepts, in `docs/theory`,
- notebooks are now rendered as part of the documentation, you can refer to them from the docstrings and other documentation pages,
- introduced a more or less unified style for docstrings.

Other:
- refactoring and bug fixes,
- added type hints throughout the library and enabled `mypy`,
- updated frontends (with limited suppot for GPJax due to conflicting dependencies),
- improved `spaces.ProductDiscreteSpectrumSpace` and `kernels.ProductGeometricKernel`,
- filtered out or fixed some annoying external warnings,
- added a new banner for `README.md` and for our [landing page](https://geometric-kernels.github.io/), courtesy of @aterenin,
- example notebooks are now run as tests,
- we now support Python 3.8, 3.9, 3.10, 3.11 and have test workflows for all the supported Python versions,
- we now provide a PyPI package,
- [LAB](https://github.com/wesselb/lab) is now a lightweight dependency, thanks to @wesselb,
- kernels are now normalized to have unit outputscale by default.

## v0.1-alpha - 20.10.2022
Alpha release.
