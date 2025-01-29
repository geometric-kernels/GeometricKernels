####################
  Kernels on Graphs
####################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following :doc:`this example notebook </examples/Graph>` if you are modeling a function of nodes, or :doc:`this example notebook </examples/GraphEdges>` if you are modeling a function of edges.

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

If you want to model a signal on the edges of a graph $G$, you can consider modeling the signal on the nodes of the `line graph <https://en.wikipedia.org/wiki/Line_graph>`_.
The line graph of $G$ has a node for each edge of $G$ and its two nodes are connected by an edge if the corresponding edges in $G$ used to share a common node.
To build the line graph, you can use the `line_graph <https://networkx.org/documentation/stable/reference/generated/networkx.generators.line.line_graph.html#line-graph>`_ function from `networkx <https://networkx.org>`_.
Alternatively, especially for the **flow-type data**, you might want to use **specialized edge kernels that we briefly describe below**.

To define kernels for flow-type data on graph edges, the graph is extended to a *simplicial 2-complex*.  
You can ignore the concept of a simplicial 2-complex and treat the space as a graph because GeometricKernels can automatically extend your graph to a simplicial 2-complex in a sensible way.
However, if you prefer to ignore this concept, you may want to disregard the rest of this section too.

Simplicial 2-complexes
----------------------

A *simplicial 2-complex* is a collection of $N_0$ nodes, $N_1$ edges, and $N_2$ triangles such that each triangle is a triplet of nodes and each edge is a pair of nodes.
**The edges are not directed, but they are oriented** [#]_.
An oriented edge, denoted as $e=[i,j]$ is an ordering of $\{i,j\}$. 
This is not a directed edge allowing flow only from $i$ to $j$, but rather an assignment of the sign of the flow: from $i$ to $j$ it is positive and the reverse is negative. 
For $e=[i,j]$, we denote the same edge with reverse orientation by $-e=[j,i]$.
Same goes for oriented triangles, denoted as $t=[i,j,k]$, where $i, j, k$ are nodes, with
$$
t=[i,j,k] = [j,k,i] = [k,i,j] = -[i,k,j] = -[k,j,i] = -[j,i,k]
.
$$

A function $f : E \to \mathbb{R}$ on the edge set $E$ of a simplicial 2-complex is required to be **alternating**, i.e. $f(-e) = -f(e)$ for all $e \in E$.
Such a function may be identified with the vector
$$
\mathbf{f}=[f(e_1),\dots,f(e_{N_1})]^\top\in\mathbb{R}^{N_1}
$$
of its values on all positively oriented edges $e_i$.
We call this vector an **edge flow**. 

Hodge Laplacian
---------------

Given a simplicial 2-complex, we can define the discrete **Hodge Laplacian**, which operates on the space of edge flows, as 
$$
\mathbf{L} = \mathbf{B}_1^\top \mathbf{B}_1 + \mathbf{B}_2 \mathbf{B}_2^\top := \mathbf{L}_{\text{d}} + \mathbf{L}_{\text{u}},
$$
where $\mathbf{B}_1$ is the *oriented* node-to-edge incidence matrix of dimension $N_0\times N_1$, and $\mathbf{B}_2$ is the *oriented* edge-to-triangle incidence matrix of dimension $N_1\times N_2$. 
For every positively oriented edge $e=[i, j]$, we have $[ \mathbf{B}_1 ]_{i e} = -1$ and $[ \mathbf{B}_1 ]_{j e} = 1$.
All the other entries of $\mathbf{B}_1$ are zero.
If an edge $e$ is aligned with the triangle $t$, we have $[ \mathbf{B}_2 ]_{e t} = 1$, if $-e$ is aligned with $t$, we have $[ \mathbf{B}_2 ]_{e t} = -1$.
All the other entries of $\mathbf{B}_2$ are zero.

The Hodge Laplacian $\mathbf{L}$ describes the connectivity of edges where the *down* part $\mathbf{L}_d$ and the *up* part $\mathbf{L}_u$ encode how edges are adjacent, respectively, through nodes and via triangles.
Matrix $\mathbf{L}$ is positive semi-definite, admitting an orthonormal basis of eigenvectors $\mathbf{f}_1, \ldots, \mathbf{f}_{N_1}$ with eigenvalues $\lambda_l$:
$$
\mathbf{L} \mathbf{f}_l = \lambda_l \mathbf{f}_l
\qquad
\lambda_l \geq 0
.
$$
The eigenvalues are assumed to be in ascending order: $\lambda_1 \leq \ldots \leq \lambda_{N_1}$.
Each $\mathbf{f}_l$ defines a function $f_l: E \to \mathbb{R}$ with $f_l(e) = \left(\mathbf{f}_l\right)_{e}$ if $e$ is positively oriented and $f_l(e) = -f_l(-e)$ otherwise.

Matérn Karhunen–Loève Kernel
-----------------------------

The eigenpairs $\lambda_l, f_l$ of the Hodge Laplacian can be used to define the :class:`~.kernels.MaternKarhunenLoeveKernel` on the set $E$ of graph edges.
Much like for the node set of a graph, this kernel is given by the formula
$$
k_{\nu, \kappa}(e,e')
\!=\!
\frac{1}{C_{\nu, \kappa}} \sum_{l=1}^{L} \Phi_{\nu, \kappa}(\lambda_l) f_l(e) f_l(e')
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

Where $L$ is the user-defined truncation parameter, and $C_{\nu, \kappa}$ is the normalizing constant making sure that $1/N_1 \sum k_{\nu, \kappa}(e, e) = 1$ where the summation is over all edges $e$ in positive orientation.

Matérn Hodge-compositional Kernel
---------------------------------

Edge flows can be thought of as discrete analogs of vector fields.
Much like for vector fields, you can define the gradient, divergence and curl of an edge flow:
$$
\begin{aligned}
\operatorname{div} \mathbf{f} &= \mathbf{B}_1 \mathbf{f},
&&
\mathbf{f} \in \mathbb{R}^{N_1},
&&
\operatorname{div} \mathbf{f} \in \mathbb{R}^{N_0},
\\
\operatorname{curl} \mathbf{f} &= \mathbf{B}_2^{\top} \mathbf{f},
&&
\mathbf{f} \in \mathbb{R}^{N_1},
&&
\operatorname{curl} \mathbf{f} \in \mathbb{R}^{N_2}.
\end{aligned}
$$
We can also define the gradient $\operatorname{grad}$, that takes a node function ($\mathbb{R}^{N_0}$ vector) returning an edge flow, and the curl-adjoint $\operatorname{curl}^*$, that takes a triangle function ($\mathbb{R}^{N_2}$ vector) returning an edge flow:
$$
\begin{aligned}
\operatorname{grad} \mathbf{g} &= \mathbf{B}_1^{\top} \mathbf{g},
&&
\mathbf{g} \in \mathbb{R}^{N_0},
&&
\operatorname{grad} \mathbf{g} \in \mathbb{R}^{N_1},
\\
\operatorname{curl}^* \mathbf{h} &= \mathbf{B}_2 \mathbf{h},
&&
\mathbf{h} \in \mathbb{R}^{N_2},
&&
\operatorname{curl}^* \mathbf{h} \in \mathbb{R}^{N_1}.
\end{aligned}
$$
In some applications, only divergence-free or curl-free flows are of interest.
The **Hodge decomposition** provides a way to decompose edge flows into gradient (curl-free), curl-adjoint (divergence-free), and harmonic (curl-free and divergence-free at the same time) components.
The curl-adjoint component we typically just call the curl component.
It motivates the :class:`~.kernels.MaternHodgeCompositionalKernel` kernel, which is the underlying kernel for the :class:`~.kernels.MaternGeometricKernel` on the :class:`~.spaces.GraphEdges` space.

Hodge decomposition implies that eigenvectors of the Hodge Laplacian can be chosen in such a way that they form three groups.

- The eigenvectors from the first group are denoted by $f_l^H$ and called harmonic.
  They satisfy $\mathbf{L} f_l^H = \operatorname{div} f_l^H = \operatorname{curl} f_l^H = 0$, corresponding to $\lambda_l^H = 0$ eigenvalues of the Hodge Laplacian $\mathbf{L}$.
- The eigenvectors from the second group are denoted by $f_l^G$ and called gradient.
  They are curl-free: $\operatorname{curl} f_l^G = 0$, corresponding to nonzero eigenvalues $\lambda_l^G$ of the down Laplacian $\mathbf{L}_d$.
- The eigenvectors from the third group are denoted by $f_l^C$ and called curl(-adjoint).
  They are divergence-free: $\operatorname{div} f_l^C = 0$, corresponding to nonzero eigenvalues $\lambda_l^C$ of the up Laplacian $\mathbf{L}_u$.

The Hodge-compositional kernel is built to enable separable control on the different Hodge subspaces.
For the truncation parameter $L$, one obtains $L$ eigenpairs of the Hodge Laplacian corresponding to the lowest eigenvalues.
Of them, $L_H$ are associated with zero eigenvalues, $L_G$ with nonzero eigenvalues of the down Laplacian, and $L_C$ with nonzero eigenvalues of the up Laplacian.
The kernel has three vector hyperparameters: $\mathbf{\nu} = (\nu_H, \nu_G, \nu_C)$; $\mathbf{\kappa} = (\kappa_H, \kappa_G, \kappa_C)$; and $\mathbf{\alpha} = (\alpha_H, \alpha_G, \alpha_C)$.
It is given by the formula
$$
k_{\mathbf{\nu}, \mathbf{\kappa}, \mathbf{\alpha}}(e,e')
\propto
\sum_{\Box \in {H,G,C}}
\alpha_{\Box}
\sum_{l=1}^{L_{\Box}} \Phi_{\nu_{\Box}, \kappa_{\Box}}(\lambda_l^{\Box}) f_l^{\Box}(e) f_l^{\Box}(e')
$$
where the proportionality constant is chosen to ensure that the average variance is equal to $1$.
We have
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
Each of the inner sums is a kernel whose range lies only in one Hodge subspace.

- By setting $\alpha_{\Box} = 0$, you can effectively "turn off" the corresponding subspace.
- By using equal $\nu_{\Box}$ and $\kappa_{\Box}$ for all $\Box$ and choosing appropriate $\alpha_{\Box}$, you can recover the Matérn Karhunen–Loève kernel, which is a special case of the Hodge-compositional Matérn kernel. [#]_
- You can also automatically infer $\mathbf{\alpha}$ from data.

.. rubric:: Footnotes

.. [#] The notion of *levels* is discussed in the documentation of the :class:`~.kernels.MaternKarhunenLoeveKernel` and :class:`~.Eigenfunctions` classes.
.. [#] The orientation of a general simplex is an equivalence class of permutations of its labels. Two orientations are equivalent (respectively, opposite) if they differ by an even (respectively, odd) permutation.
.. [#] Although the formula on this page recovers the Matérn Karhunen–Loève kernel for $\alpha_H = \alpha_G = \alpha_C = 1$, the implementation is a bit different because of normalization considerations.
