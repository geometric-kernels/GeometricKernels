####################
  Kernels on Meshes
####################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the :doc:`example notebook on meshes </examples/Mesh>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`borovitskiy2020`. [#]_

One one hand, this is similar to the first section of the :doc:`theory on compact manifolds <compact>` because meshes are discretizations of compact 2-dimensional manifolds.
On the other hand, this is similar to the :doc:`theory on compact graphs <graphs>` because meshes are graphs with additional structure.

=======
Theory
=======

Consider a mesh $M$ with $N$ nodes.
There are a few notions of *Laplacian* $\mathbf{\Delta}$ for $M$, which is always a positive semidefinite matrix of size $N \times N$. We use the *robust Laplacian* by :cite:t:`sharp2020` implemented in the `robust_laplacian <https://github.com/nmwsharp/robust-laplacians-py>`_  package.

Since $\mathbf{\Delta}$ is positive semidefinite, there is an orthonormal basis $\{\boldsymbol f_l\}_{l=0}^{N-1}$ in $\mathbb{R}^N$ of eigenvectors such that $\mathbf{\Delta} \boldsymbol f_l = \lambda_l \boldsymbol f_l$ for $0 = \lambda_0 \leq \lambda_2 \leq \ldots \leq \lambda_{N-1}$.

The eigenvectors $f_l$ can be regarded as functions on the mesh nodes: $f_l(j) = (f_l)_j$.
For meshes, :class:`~.kernels.MaternGeometricKernel` is an alias to :class:`~.kernels.MaternKarhunenLoeveKernel`.
The latter is given by the formula
$$
k_{\nu, \kappa}(i,j)
=
\frac{1}{C_{\nu, \kappa}} \sum_{l=0}^{L-1} \Phi_{\nu, \kappa}(\lambda_l) f_l(i) f_l(j)
\quad
\Phi_{\nu, \kappa}(\lambda)
=
\begin{cases}
\left(\frac{2\nu}{\kappa^2} + \lambda\right)^{-\nu - d/2}
&
\nu < \infty \text{ — Matérn}
\\
e^{-\frac{\kappa^2}{2} \lambda}
&
\nu = \infty \text{ — Heat (RBF)}
\end{cases}
$$
The notation here is as follows.

* $d$ is the dimension of the mesh (i.e. the dimension of the implied manifold the mesh approximates). In our implementation $d = 2$ as we only handle 2-dimensional meshes in $\mathbb{R}^3$.

* $1 \leq L \leq N$ controls the quality of approximation of the kernel.

  * Setting $L = N$ gives you the exact kernel but usually requires $O(N^3)$ to compute the eigenpairs.

  * Setting $L \ll N$ can in principle allow much faster eigenpair computation because the Laplacian is usually sparse for meshes.
    Such techniques are, however, not (yet) implemented in GeometricKernels.

* The constant $C_{\nu, \kappa}$ above ensures that the average variance is equal to $1$, i.e. $\frac{1}{N} \sum_{n=1}^N k(n, n) = 1$.
  It is easy to show that $C_{\nu, \kappa} = \sum_{n=1}^L \Phi_{\nu, \kappa}(\lambda_n)$.

.. note::
  The "variance" $k(x, x)$ can vary from point to point.

.. rubric:: Footnotes

.. [#] Similar ideas have also appeared in :cite:t:`solin2020` and :cite:t:`coveney2020`.
