####################
  Kernels on Graphs
####################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the respective :doc:`example notebook </examples/Graph>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`borovitskiy2021`.

==========================
Node Set of a Graph
==========================

Consider a (weighted) undirected graph with $N$ nodes determined by a (weighted) adjacency matrix $\mathbf{A}$ of size $N \times N$.
The weights are nonnegative: $\mathbf{A}_{i j} \geq 0$.

Let $\mathbf{D}$ denote the degree matrix, i.e. the diagonal matrix with $\mathbf{D}_{j j} = \sum_{n=1}^N \mathbf{A}_{n j}$.
Then there are two notions of the *graph Laplacian* you can use to define kernels: the *unnormalized graph Laplacian* $\mathbf{\Delta}_{un} = \mathbf{D} - \mathbf{A}$ and the *symmetric normalized graph Laplacian* $\mathbf{\Delta}_{no} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ where $\mathbf{I}$ denotes the identity matrix.
Which one to use is up to you. The performance will depend on the task at hand. In practice, it makes sense to try both.

If $\mathbf{\Delta}$ is either $\mathbf{\Delta}_{un}$ or $\mathbf{\Delta}_{no}$, it is a symmetric positive semidefinite matrix.
Because of this, there is an orthonormal basis $\{\boldsymbol f_l\}_{l=0}^{N-1}$ of $\mathbb{R}^N$ consisting of eigenvectors such that
$$
\mathbf{\Delta} \boldsymbol f_l
=
\lambda_l \boldsymbol f_l
\qquad
\text{for}
\qquad
0 = \lambda_0 \leq \lambda_2 \leq \ldots \leq \lambda_{N-1}
.
$$

The eigenvectors $\boldsymbol f_l$ can be regarded as functions on the graph nodes: $f_l(j) = (\boldsymbol f_l)_j$.
For graphs, :class:`~.kernels.MaternGeometricKernel` is an alias to :class:`~.kernels.MaternKarhunenLoeveKernel`.
The latter is given by the formula
$$
k_{\nu, \kappa}(i,j)
\!=\!
\frac{1}{C_{\nu, \kappa}} \sum_{l=0}^{L-1} \Phi_{\nu, \kappa}(\lambda_l) f_l(i) f_l(j)
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

  * Throughout the package, $L$ is referred to as the *number of levels* (though this term may have different meaning for other spaces [#]_).

  * Setting $L = N$ gives you the exact kernel but usually requires $O(N^3)$ to compute the eigenpairs.

  * Setting $L \ll N$ can in principle allow much faster eigenpair computation for some graphs. Such techniques are, however, not (yet) implemented in the library.

* The constant $C_{\nu, \kappa}$ ensures that the average variance is equal to $1$, i.e. $\frac{1}{N} \sum_{n=1}^N k_{\nu, \kappa}(n, n) = 1$.

  * It is easy to show that $C_{\nu, \kappa} = \sum_{l=0}^{L-1} \Phi_{\nu, \kappa}(\lambda_l)$.

**Notes:**

#. The "variance" $k(n, n)$ can vary from point to point.

#. Unlike the Euclidean or the manifold case, the $1/2, 3/2, 5/2$ may fail to be the reasonable values of $\nu$.
   On the other hand, the parameter $\nu$ is optimizable in the same way in which the lengthscale is.
   Keep in mind though that the optimization problem may require some trial and error to find a good initialization and that reasonable $\kappa$ and $\nu$ will heavily depend on the specific graph in a way that is hard to predict.

#. Consider $\mathbf{A}' = \alpha^2 \mathbf{A}$ for some $\alpha > 0$.
   Then, for the normalized graph Laplacian $\mathbf{\Delta}_{no}$, we have $k_{\nu, \kappa}' (i, j) = k_{\nu, \kappa} (i, j)$ where $k_{\nu, \kappa}'$ is the kernel corresponding to $\mathbf{A}'$ instead of $\mathbf{A}$.
   On the other hand, for the unnormalized graph Laplacian $\mathbf{\Delta}_{un}$, we have $k_{\nu, \kappa}' (i, j) = k_{\nu, \alpha \cdot \kappa} (i, j)$, i.e. the lengthscale changes.

==========================
Edge Set of a Graph
==========================

if you want to model a **signal on the edges** of a graph $G$, you can consider modeling the signal on the nodes of the `line graph <https://en.wikipedia.org/wiki/Line_graph>`_. The line graph of $G$ has a node for each edge of $G$ and its two nodes are connected by an edge if the corresponding edges in $G$ used to share a common node. To build the line graph, you can use the `line_graph <https://networkx.org/documentation/stable/reference/generated/networkx.generators.line.line_graph.html#line-graph>`_ function from `networkx <https://networkx.org>`_.

Alternatively, especially for the flow-type data, you might want to use specialized edge kernels, see :cite:t:`yang2024` and :cite:t:`alain2023`.
These are, however, not implemented in GeometricKernels at the moment.

.. rubric:: Footnotes

.. [#] The notion of *levels* is discussed in the documentation of the :class:`~.kernels.MaternKarhunenLoeveKernel` and :class:`~.Eigenfunctions` classes.