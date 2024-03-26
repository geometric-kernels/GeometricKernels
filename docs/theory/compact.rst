###############################
Kernels on Compact Manifolds
###############################

**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>`, following the example notebook `on hypersheres <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Hypersphere.ipynb>`_. This is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.

=======
Theory
=======

For compact Riemannian manifolds, :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>` is an alias to :class:`MaternKarhunenLoeveKernel <geometric_kernels.kernels.MaternKarhunenLoeveKernel>`.
For such a manifold $\mathcal{M}$ the latter is given by the formula
$$
k_{\nu, \kappa}(x,x')
\!=\!
\frac{1}{C_{\nu, \kappa}} \sum_{n=1}^N \Phi_{\nu, \kappa}(\lambda_n) f_n(x) f_n(x')
\quad
\Phi_{\nu, \kappa}(\lambda)
\!=\!
\begin{cases}
\left(\frac{2\nu}{\kappa^2} + \lambda\right)^{-\nu-\frac{d}{2}}
&
\nu < \infty \text{ — Matérn}
\\
e^{-\frac{\kappa^2}{2} \lambda}
&
\nu = \infty \text{ — Heat (RBF)}
\end{cases}
$$
The notation here is as follows.

* The values $\lambda_n \geq 0$ and the functions $f_n(\cdot)$ are *eigenvalues* and *eigenfunctions* of the *Laplace–Beltrami operator* $\Delta_{\mathcal{M}}$ on $\mathcal{M}$ such that
  $$
  \Delta_{\mathcal{M}} f_n = \lambda_n f_n
  .
  $$
  The functions $\left\{f_n\right\}_{n=1}^{\infty}$ constitute an orthonormal basis of the space $L^2(\mathcal{M})$ of square integrable functions on the manifold $\mathcal{M}$ with respect to the inner product $\langle f, g \rangle_{L^2(\mathcal{M})} = \frac{1}{\lvert\mathcal{M}\rvert} \int_{\mathcal{M}} f(x) g(x) \mathrm{d} x$.

* $d$ is the dimension of the manifold.

* The summation limit $1 \leq N < \infty$ controls the quality of approximation of the kernel.
  For some manifolds, e.g. manifolds represented by discrete :class:`meshes <geometric_kernels.spaces.Mesh>`, this corresponds to the *number of levels* parameter of the :class:`MaternKarhunenLoeveKernel <geometric_kernels.kernels.MaternKarhunenLoeveKernel>`. For others, for which the *Addition theorem* holds (:doc:`see the respective page <addition_theorem>`), the *number of levels* parameter has a different meaning.

* $C_{\nu, \kappa}$ is the constant which ensures that average variance is equal to $1$, i.e. $\frac{1}{\lvert\mathcal{M}\rvert}\int_{\mathcal{M}} k(x, x) \mathrm{d} x = 1$ where $\lvert\mathcal{M}\rvert$ denotes the volume of the manifold $\mathcal{M}$.
  It is easy to show that $C_{\nu, \kappa} = \sum_{n=1}^N \Phi_{\nu, \kappa}(\lambda_n)$.

**Note:** For general manifolds, $k(x, x)$ can vary from point to point.
You usually observe this for manifolds represented by meshes, the ones which do not have a lot of symmetries.
On the other hand, for the hyperspheres $k(x, x)$ is a constant, as it is for all *homogeneous spaces* which hyperspheres are instances of, as well as for Lie groups (which are also instances of homogeneous spaces).
