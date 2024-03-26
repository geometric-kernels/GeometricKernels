####################
  Kernels on Meshes
####################

**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>`, following the example notebook `on meshes <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Mesh.ipynb>`_. This is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.

**Note:** one one hand, this is similar to the first section of the :doc:`theory on compact manifolds <compact>` because meshes are discretizations of compact 2-dimensional manifolds.
On the other hand, this is similar to the :doc:`theory on compact graphs <graphs>` because meshes are graphs with additional structure.

=======
Theory
=======

Consider a mesh $M$ with $N$ nodes.
There are a few notions of *Laplacian* $\mathbf{\Delta}$ for $M$, which is always a positive semidefinite matrix of size $N \times N$. We use the *robust Laplacian* by Sharp and Crane (2020) implemented in the `robust_laplacian <https://github.com/nmwsharp/robust-laplacians-py>`_  package.

Since $\mathbf{\Delta}$ is positive semidefinite, there is an orthonormal basis $\{\boldsymbol f_n\}_{n=1}^N$ in $\mathbb{R}^N$ of eigenvectors such that $\mathbf{\Delta} \boldsymbol f_n = \lambda_n \boldsymbol f_n$ for $0 = \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_N$.

The eigenvectors $f_n$ can be regarded as functions on the mesh nodes: $f_n(j) = (f_n)_j$.
For meshes, :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>` is an alias to :class:`MaternKarhunenLoeveKernel <geometric_kernels.kernels.MaternKarhunenLoeveKernel>`.
The latter is given by the formula
$$
k_{\nu, \kappa}(i,j)
=
\frac{1}{C_{\nu, \kappa}} \sum_{n=1}^L \Phi_{\nu, \kappa}(\lambda_n) f_n(i) f_n(j)
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

**Note:** the "variance" $k(x, x)$ can vary from point to point.
