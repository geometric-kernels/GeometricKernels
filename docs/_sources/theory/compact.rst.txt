###############################
Kernels on Compact Manifolds
###############################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the :doc:`example notebook on hypersheres </examples/Hypersphere>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`borovitskiy2020`. [#]_

=======
Theory
=======

For compact Riemannian manifolds, :class:`~.kernels.MaternGeometricKernel` is an alias to :class:`~.kernels.MaternKarhunenLoeveKernel`.
For such a manifold $\mathcal{M}$ the latter is given by the formula
$$
k_{\nu, \kappa}(x,x')
\!=\!
\frac{1}{C_{\nu, \kappa}} \sum_{j=0}^{J-1} \Phi_{\nu, \kappa}(\lambda_j) f_j(x) f_j(x')
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

* The values $\lambda_j \geq 0$ and the functions $f_j(\cdot)$ are *eigenvalues* and *eigenfunctions* of the minus *Laplace–Beltrami operator* $-\Delta_{\mathcal{M}}$ on $\mathcal{M}$ such that
  $$
  \Delta_{\mathcal{M}} f_j = - \lambda_j f_j
  .
  $$
  The functions $\left\{f_j\right\}_{j=0}^{\infty}$ constitute an orthonormal basis of the space $L^2(\mathcal{M})$ of square integrable functions on the manifold $\mathcal{M}$ with respect to the inner product $\langle f, g \rangle_{L^2(\mathcal{M})} = \frac{1}{\lvert\mathcal{M}\rvert} \int_{\mathcal{M}} f(x) g(x) \mathrm{d} x$, where $\lvert\mathcal{M}\rvert$ denotes the volume of the manifold $\mathcal{M}$.

* $d$ is the dimension of the manifold.

* The number of eigenpairs $1 \leq J < \infty$ controls the quality of approximation of the kernel.
  For some manifolds, e.g. manifolds represented by discrete :class:`meshes <.spaces.Mesh>`, this corresponds to the *number of levels* parameter of the :class:`~.kernels.MaternKarhunenLoeveKernel`. For others, for which the *addition theorem* holds (:doc:`see the respective page <addition_theorem>`), the *number of levels* parameter has a different meaning [#]_.

* $C_{\nu, \kappa}$ is the constant which ensures that average variance is equal to $1$, i.e. $\frac{1}{\lvert\mathcal{M}\rvert}\int_{\mathcal{M}} k(x, x) \mathrm{d} x = 1$.
  It is easy to show that $C_{\nu, \kappa} = \sum_{j=0}^{J-1} \Phi_{\nu, \kappa}(\lambda_j)$.

**Note:** For general manifolds, $k(x, x)$ can vary from point to point.
You usually observe this for manifolds represented by meshes, the ones which do not have a lot of symmetries.
On the other hand, for the hyperspheres $k(x, x)$ is a constant, as it is for all *homogeneous spaces* which hyperspheres are instances of, as well as for Lie groups (which are also instances of homogeneous spaces).

.. rubric:: Footnotes

.. [#] Similar ideas have also appeared in :cite:t:`solin2020` and :cite:t:`coveney2020`.

.. [#] The notion of *levels* is discussed in the documentation of the :class:`~.kernels.MaternKarhunenLoeveKernel` and :class:`~.Eigenfunctions` classes.