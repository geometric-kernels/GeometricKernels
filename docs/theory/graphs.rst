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


Consider a simplicial 2-complex with $N_0$ nodes, $N_1$ edges and $N_2$ triangular faces (triangles). 
We additionally assign reference orientations [#]_ to the edges and triangles according to the increaseing order of their node labels. 
An oriented edge, denoted as $e=[i,j]$ is an ordering of $\{i,j\}$. 
**Note that** this is not a directed edge allowing flow only from $i$ to $j$, but rather an assignment of the sign of the flow: from $i$ to $j$ it is positive and the reverse is negative. 
Same goes for oriented triangles, denoted as $t=[i,j,k]$ and we have $t=[i,j,k] = [j,k,i] = [k,i,j] = -[i,k,j] = -[k,j,i] = -[j,i,k]$.

In a simplicial 2-complex, the functions, $f_1:E\to\mathbb{R}$, on its edges $E$ are required to be _alternating_ :cite:p:`lim2020hodge`. 
By collecting the edge functions on $E$ into a vector $\mathbf{f}_1=[f_1(e_1),\dots,f_1(e_{N_1}]^\top\in\mathbb{R}^{N_1}$, we then obtain an **edge flow**. 
 
Given a simplicial 2-complex, we can define the discrete **Hodge Laplacian**, which operates on the space of edge flows, as 
$$
\mathbf{L} = \mathbf{B}_1^\top \mathbf{B}_1 + \mathbf{B}_2 \mathbf{B}_2^\top := \mathbf{L}_{\text{d}} + \mathbf{L}_{\text{u}},
$$
where $\mathbf{B}_1$ is the *oriented* node-to-edge incidence matrix of dimension $N_0\times N_1$, and $\mathbf{B}_2$ is the *oriented* edge-to-triangle incidence matrix of dimension $N_1\times N_2$. 
For $\mathbf{B}_2$, its entries are $[ \mathbf{B}_2 ]_{e t} = 1$, for $e = [i, j]$ or $e = [j, k]$, and $[ \mathbf{B}_2 ]_{e t} = -1$ for $e = [i, k]$, if a triangle $t = [i, j, k]$ exists, and zero otherwise.

Thus, the Hodge Laplacian $\mathbf{L}$ describes the connectivity of edges where the *down* part $\mathbf{L}_d$ and the *up* part $\mathbf{L}_u$ encode how edges are adjacent, respectively, through nodes and via triangles.
Matrix $\mathbf{L}$ is positive semi-definite, admitting an eigendecomposition $\mathbf{L} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^{T}$ where diagonal matrix $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_N)$ collects the eigenvalues and $\mathbf{U}$ is the eigenvector matrix. 
Here, we consider the unweighted $\mathbf{L}$ but it also holds for the weighted variants.

The eigenvectors provide an orthonormal basis for the space of edge flows. 
Furthermore, the  :doc:`Hodge decomposition </theory/hodge>` says that the space of edge flows can be decomposed into harmonic, gradient and curl subspaces. 
Moreover, :cite:t:`yang2022simplicial` showed that the eigenspace of the Hodge Laplacian can be reorganized in terms of the three Hodge subspaces as 
$$
\mathbf{U} = \begin{bmatrix} \mathbf{U}_{H} & \mathbf{U}_{G} & \mathbf{U}_{C} \end{bmatrix},
$$
where $\mathbf{U}_H$ is the eigenvector matrix associated to zero eigenvalues $\boldsymbol{\Lambda}_H = 0$ of $\mathbf{L}_1$, $\mathbf{U}_G$ is associated to the nonzero eigenvalues $\boldsymbol{\Lambda}_G$ of $\mathbf{L}_d$, and $\mathbf{U}_C$ is associated to the nonzero eigenvalues $\boldsymbol{\Lambda}_C$ of $\mathbf{L}_u$. 
That is, they span the Hodge subspaces:
$$
\mathrm{span}(\mathbf{U}_H) = \ker(\mathbf{L}_1), \quad \mathrm{span}(\mathbf{U}_G) = \mathrm{im}(\mathbf{B}_1^{\top}), \quad
\mathrm{span}(\mathbf{U}_C) = \mathrm{im}(\mathbf{B}_2)
$$
where $\mathrm{span}(\bullet)$ denotes all possible linear combinations of columns of $\bullet$.

The Hodge-compositional edge kernel is built to enable separable control on the different Hodge subspaces. 
Specifically, the kernel is defined as 
$$
\mathbf{K}_{\nu,\kappa} = \mathbf{K}_{H} + \mathbf{K}_{G} + \mathbf{K}_{C}, 
\quad 
\text{where}
\quad
\mathbf{K}_{\Box} = \mathbf{U}_{\Box} \Phi_{\Box}(\boldsymbol{\Lambda}_{\Box}) \mathbf{U}_{\Box}^\top
$$ 
for $\Box = H,G,C$, with $\Phi_{\Box}(\boldsymbol{\Lambda}_{\Box})$ having diagonal entries 
$$
\Phi_{\Box}({\lambda}_{\Box}) 
= 
\begin{cases}
\sigma_{\Box}^2
\left(\frac{2\nu_{\Box}}{\kappa_{\Box}^2} + \lambda_{\Box}\right)^{-\nu_{\Box}}
&
\text{ — Matérn}
\\
\sigma_{\Box}^2
e^{-\frac{\kappa_{\Box}^2}{2} \lambda_{\Box}}
&
\text{ — Heat (RBF)}
\end{cases}
$$

That is, each $\mathbf{K}_{\Box}$ encodes the covariance between edge functions *individually* for the three Hodge subspaces and the three sub-kernels do not share hyperparameters.


.. rubric:: Footnotes

.. [#] The notion of *levels* is discussed in the documentation of the :class:`~.kernels.MaternKarhunenLoeveKernel` and :class:`~.Eigenfunctions` classes.
.. [#] The orientation of a general simplex is an equivalence class of permutations of its labels. Two orientations are equivalent (respectively, opposite) if they differ by an even (respectively, odd) permutation :cite:p:`lim2020hodge`.