################################
  Kernels on the Hypercube Graph
################################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the respective :doc:`example notebook </examples/HypercubeGraph>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`borovitskiy2023`.

==========================
Motivation
==========================

The :class:`~.spaces.HypercubeGraph` space $C^d$ can be used to model $d$-dimensional *binary vector* inputs.

There are many settings where inputs are binary vectors or can be represented as such. For instance, upon flattening, binary vectors represent adjacency matrices of *unweighted labeled graphs* [#]_. 

==========================
Structure of the Space
==========================

The elements of this space—given its `dim` is $d \in \mathbb{Z}_{>0}$—are exactly the binary vectors of length $d$.

The geometry of this space is simple: it is a graph such that $x, x' \in C^d$ are connected by an edge if and only if they differ in exactly one coordinate (i.e. there is exactly *Hamming distance* $1$ between).

Being a graph, $C^d$ could also be represented using the general :class:`~.spaces.Graph` space.
However, the number of nodes in $C^d$ is $2^d$, which is exponential in $d$, rendering the general techniques infeasible.

==========================
Eigenfunctions
==========================

On graphs, kernels are computed using the eigenfunctions and eigenvalues of the Laplacian.

The eigenfunctions of the Laplacian on the hypercube graph are the *Walsh functions* [#]_ given analytically by the simple formula
$$
w_T(x_0, .., x_{d-1}) = (-1)^{\sum_{j \in T} x_j}
$$
where $x = (x_0, .., x_{d-1}) \in C^d$ and the index $T$ is an arbitrary subset of the set $\{0, .., d-1\}$.

The corresponding eigenvalues are $\lambda_T = \lambda_{\lvert T \rvert} = 2 \lvert T \rvert / d$, where $\lvert T \rvert$ is the cardinality of $T$.

However, the problem is that the number of eigenfunctions is $2^d$.
Hence naive truncation of the sum in the kernel formula to a few hundred terms leads to a poor approximation of the kernel for larger $d$.

==========================
Addition Theorem
==========================

Much like for the hyperspheres and unlike for the general graphs, there is an :doc:`addition theorem <addition_theorem>` for the hypercube graph:

$$
\sum_{T \subseteq \{0, .., d-1\}, \lvert T \rvert = j} w_T(x) w_T(x')
=
\sum_{T \subseteq \{0, .., d-1\}, \lvert T \rvert = j} w_T(x \oplus x')
=
\binom{d}{j}
\widetilde{G}_{d, j, m}
$$
where $\oplus$ is the elementwise XOR operation, $m$ is the Hamming distance between $x$ and $x'$, and $\widetilde{G}_{d, j, m}$ is the Kravchuk polynomial of degree $d$ and order $j$ normalized such that $\widetilde{G}_{d, j, 0} = 1$, evaluated at $m$.

Normalized Kravchuk polynomials $\widetilde{G}_{d, j, m}$ satisfy the following three-term recurrence relation
$$
\widetilde{G}_{d, j, m}
=
\frac{d - 2 m}{d - j + 1} \widetilde{G}_{d, j - 1, m}
-\frac{j-1}{d - j + 1} \widetilde{G}_{d, j - 2, m},
\quad
\widetilde{G}_{d, 0, m} = 1,
\quad
\widetilde{G}_{d, 1, m} = 1 - \frac{2}{d} m,
$$
which allows for their efficient computation without the need to compute large sums of the individual Walsh functions.

With that, the kernels on the hypercube graph can be computed efficiently using the formula
$$
k_{\nu, \kappa}(x, x')
=
\frac{1}{C_{\nu, \kappa}}
\sum_{l=0}^{L-1}
\Phi_{\nu, \kappa}(\lambda_l)
\binom{d}{j} \widetilde{G}_{d, j, m}
\qquad
\Phi_{\nu, \kappa}(\lambda)
=
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
where $m$ is the Hamming distance between $x$ and $x'$, and $L \leq d + 1$ is the user-controlled number of levels parameters.

**Notes:**

#. We define the dimension of the :class:`~.spaces.HypercubeGraph` space $C^d$ to be $d$, in contrast to the graphs represented by the :class:`~.spaces.Graph` space, whose dimension is defined to be $0$.

   Because of this, much like in the Euclidean or the manifold case, the $1/2, 3/2, 5/2$ *are* in fact reasonable values of for the smoothness parameter $\nu$.

.. rubric:: Footnotes

.. [#] Every node of a labeled graph is associated with a unique label. Functions on labeled graphs do *not* have to be invariant to permutations of nodes.

.. [#] Since the hypercube graph $C^d$ is $d$-regular, the unnormalized Laplacian and the symmetric normalized Laplacian coincide up to a multiplication by $d$. Thus their eigenfunctions are the same and eigenvalues coincide up to a multiplication by $d$. For better numerical stability, we use symmetric normalized Laplacian in the implementation and assume its use throughout this page.