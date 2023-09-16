####################
  Kernels on Graphs
####################

**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>`, following the respective `example notebook <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Graph.ipynb>`_. This is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2021) <https://arxiv.org/abs/2010.15538>`_.

==========================
Node Set of a Graph
==========================

Consider a (weighted) undirected graph with $N$ nodes determined by a (weighted) adjacency matrix $\mathbf{A}$ of size $N \times N$.
The weights are nonnegative: $\mathbf{A}_{i j} \geq 0$.

Let $\mathbf{D}$ denote the degree matrix, i.e. the diagonal matrix with $\mathbf{D}_{j j} = \sum_{n=1}^N \mathbf{A}_{n j}$.
Then there are two notions of the *graph Laplacian* you can use to define kernels: the *unnormalized graph Laplacian* $\mathbf{\Delta}_{un} = \mathbf{D} - \mathbf{A}$ and the *symmetric normalized graph Laplacian* $\mathbf{\Delta}_{no} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ where $\mathbf{I}$ denotes the identity matrix.
Which one to use is up to the user, the performance will depend on the task at hand, in practice it makes sense to try both.

If $\mathbf{\Delta}$ is either $\mathbf{\Delta}_{un}$ or $\mathbf{\Delta}_{no}$, it is a symmetric positive semidefinite matrix.
Because of this, there is an orthonormal basis $\{\boldsymbol f_n\}_{n=1}^N$ of $\mathbb{R}^N$ consisting of eigenvectors such that
$$
\mathbf{\Delta} \boldsymbol f_n
=
\lambda_n \boldsymbol f_n
\qquad
\text{for}
\qquad
0 = \lambda_1 \leq \lambda_2 \leq \ldots \leq \lambda_N
.
$$

The eigenvectors $\boldsymbol f_n$ can be regarded as functions on the graph nodes: $f_n(j) = (\boldsymbol f_n)_j$.
For graphs, :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>` is an alias to :class:`MaternKarhunenLoeveKernel <geometric_kernels.kernels.MaternKarhunenLoeveKernel>`.
The latter is given by the formula
$$
k_{\nu, \kappa}(i,j)
\!=\!
\frac{1}{C_{\nu, \kappa}} \sum_{n=1}^L \Phi_{\nu, \kappa}(\lambda_n) f_n(i) f_n(j)
\quad
\Phi_{\nu, \kappa}(\lambda)
=
\begin{cases}
\left(\frac{2\nu}{\kappa^2} + \lambda\right)^{-\nu}
&
\nu < \infty \text{ — Matérn}
\\
e^{-\frac{\kappa^2}{2} \lambda}
&
\nu = \infty \text{ — Heat (RBF)}
\end{cases}
$$
The notation here is as follows.

* $1 \leq L \leq N$ controls the quality of approximation of the kernel.

  * Throughout the package, $L$ is referred to as the *number of levels* (though this term may have different meaning for other spaces).

  * Setting $L = N$ gives you the exact kernel but usually requires $O(N^3)$ to compute the eigenpairs.

  * Setting $L \ll N$ can in principle allow much faster eigenpair computation for some graphs, such techniques are, however, not (yet) implemented in GeometricKernels.

* The constant $C_{\nu, \kappa}$ ensures that average variance is equal to $1$, i.e. $\frac{1}{N} \sum_{n=1}^N k_{\nu, \kappa}(n, n) = 1$.

  * It is easy to show that $C_{\nu, \kappa} = \sum_{n=1}^L \Phi_{\nu, \kappa}(\lambda_n)$.

**Notes:**

#. The "variance" $k(x, x)$ can vary from point to point.

#. Unlike the Euclidean or the manifold case, the $1/2, 3/2, 5/2$ may fail to be the reasonable values of $\nu$.
   On the other hand, the parameter $\nu$ is optimizable in the same way in which the lengthscale is.
   Keep in mind though, that the optimization problem may require finding some trial and error to find good a initialization and that reasonable $\kappa$ and $\nu$ will heavily depend on the specific graph in a way that is hard to predict.

#. Consider $\mathbf{A}' = \alpha^2 \mathbf{A}$ for some $\alpha > 0$.
   Then, for the normalized graph Laplacian $\mathbf{\Delta}_{no}$, we have $k_{\nu, \kappa}' (i, j) = k_{\nu, \kappa} (i, j)$ where $k_{\nu, \kappa}'$ is the kernel corresponding to $\mathbf{A}'$ instead of $\mathbf{A}$.
   On the other hand, for the unnormalized graph Laplacian $\mathbf{\Delta}_{un}$, we have $k_{\nu, \kappa}' (i, j) = k_{\nu, \alpha \cdot \kappa} (i, j)$, i.e. the lengthscale changes.

==========================
Edge Set of a Graph
==========================

if you want to model a **signal on the edges** of a graph $G$, you can consider modeling the signal on the nodes of the `line graph <https://en.wikipedia.org/wiki/Line_graph>`_. The line graph of $G$ has a node for each edge of $G$ and its two nodes are connected by an edge if the corresponding edges in $G$ used to share a common node. To build the line graph, you can use the `line_graph <https://networkx.org/documentation/stable/reference/generated/networkx.generators.line.line_graph.html#line-graph>`_ function of `networkx <https://networkx.org>`_.
