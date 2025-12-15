################################
  Kernels on the Hamming Graph
################################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the respective :doc:`example notebook </examples/HammingGraph>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`doumont2025` and :cite:t:`borovitskiy2023`.

==========================
Motivation
==========================

The :class:`~.spaces.HammingGraph` space $H(d,q)$ can be used to model $d$-dimensional *categorical vector* inputs, where each component takes values in $\{0, 1, ..., q-1\}$.

There are many settings where inputs are categorical vectors or can be represented as such. For instance, DNA sequences (with $q=4$ for A, C, G, T), protein sequences (with $q=20$ for the standard amino acids), or error-correcting codes over finite alphabets. When $q=2$, the Hamming graph reduces to the hypercube graph $C^d$ (see :doc:`HypercubeGraph </theory/hypercube_graph>`), which represents binary vectors.

==========================
Structure of the Space
==========================

The elements of this space—given its `dim` is $d \in \mathbb{Z}_{>0}$ and `n_cat` is $q \in \mathbb{Z}_{>1}$—are exactly the categorical vectors of length $d$ where each component is in $\{0, 1, ..., q-1\}$.

The geometry of this space is simple: it is a graph such that $x, x' \in H(d,q)$ are connected by an edge if and only if they differ in exactly one coordinate (i.e. *Hamming distance* $1$ between them is exactly $1$).

Being a graph, $H(d,q)$ could also be represented using the general :class:`~.spaces.Graph` space.
However, the number of nodes in $H(d,q)$ is $q^d$, which is exponential in $d$ (and grows rapidly with $q$), rendering the general graph representation intractable in most cases.

==========================
Eigenfunctions
==========================

On graphs, kernels are computed using the eigenfunctions and eigenvalues of the Laplacian.

The eigenfunctions of the Laplacian on the Hamming graph are the *Vilenkin functions* (also known as *generalized Walsh functions*) [#]_. These are characters of the Abelian group $\mathbb{Z}_q^d$ and can be expressed using $q$-th roots of unity.

The Vilenkin functions $\psi_{\mathbf{s}}$ are indexed by frequency vectors $\mathbf{s} = (s_0, \ldots, s_{d-1})$ where each $s_i \in \{0, 1, \ldots, q-1\}$, and take the form
$$
\psi_{\mathbf{s}}(x_0, \ldots, x_{d-1}) = \prod_{i=0}^{d-1} \omega_q^{s_i x_i}
$$
where $\omega_q = e^{2\pi i/q}$ is a primitive $q$-th root of unity and $x = (x_0, \ldots, x_{d-1}) \in H(d,q)$.

The eigenfunctions with the same number of non-zero frequency components $j = \lvert\{i : s_i \neq 0\}\rvert$ share the same eigenvalue $\lambda_j = \frac{q j}{(q-1)d}$ and form a *level* $j$. The dimension of level $j$ is $\binom{d}{j}(q-1)^j$.

However, the problem is that the number of eigenfunctions is $q^d$.
Hence naive truncation of the sum in the kernel formula to a few hundred terms leads to a poor approximation of the kernel for larger $d$ or $q$.

**Note:** Direct evaluation of Vilenkin functions is not currently implemented for $q > 2$, but this is not required for kernel computation thanks to the addition theorem below.

==========================
Addition Theorem
==========================

Much like for hyperspheres and hypercube graphs, there is an :doc:`addition theorem <addition_theorem>` for the Hamming graph:

$$
\sum_{\substack{T \subseteq \{0, .., d-1\} \\ \lvert T \rvert = j}} \sum_{\alpha \in \{1,..,q-1\}^T} \psi_{T,\alpha}(x) \overline{\psi_{T,\alpha}(x')}
=
\binom{d}{j} (q-1)^j
\widetilde{K}_{d, q, j}(m)
$$

where $\psi_{T,\alpha}$ denotes the Vilenkin function indexed by the subset $T$ with frequency values $\alpha = (\alpha_i)_{i \in T}$, $m$ is the Hamming distance between $x$ and $x'$, and $\widetilde{K}_{d, q, j}(m)$ is the generalized Kravchuk polynomial of degree $d$, alphabet size $q$, and order $j$, normalized such that $\widetilde{K}_{d, q, j}(0) = 1$.

Normalized generalized Kravchuk polynomials $\widetilde{K}_{d, q, j}(m)$ satisfy the following three-term recurrence relation:
$$
\widetilde{K}_{d, q, j}(m)
=
\frac{d(q-1) - qm + q - j(q-1)}{(d - j + 1)(q-1)} \widetilde{K}_{d, q, j-1}(m)
- \frac{j-1}{d - j + 1} \widetilde{K}_{d, q, j-2}(m)
$$
with initial conditions:
$$
\widetilde{K}_{d, q, 0}(m) = 1,
\quad
\widetilde{K}_{d, q, 1}(m) = 1 - \frac{q}{(q-1)d} m,
$$
which allows for their efficient computation without the need to compute large sums of the individual Vilenkin functions.

With that, the kernels on the Hamming graph can be computed efficiently using the formula:
$$
k_{\nu, \kappa}(x, x')
=
\frac{1}{C_{\nu, \kappa}}
\sum_{j=0}^{L-1}
\Phi_{\nu, \kappa}(\lambda_j)
\binom{d}{j} (q-1)^j \widetilde{K}_{d, q, j}(m)
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
where $m$ is the Hamming distance between $x$ and $x'$, and $L \leq d + 1$ is the user-controlled number of levels parameter.

**Notes:**

#. We define the dimension of the :class:`~.spaces.HammingGraph` space $H(d,q)$ to be $d$, in contrast to the graphs represented by the :class:`~.spaces.Graph` space, whose dimension is defined to be $0$.

   Because of this, much like in the Euclidean or the manifold case, the $1/2, 3/2, 5/2$ *are* in fact reasonable values for the smoothness parameter $\nu$.

#. When $q=2$, the Hamming graph $H(d,2)$ is isomorphic to the hypercube graph $C^d$, and the generalized Kravchuk polynomials reduce to the standard Kravchuk polynomials ($q=2$).

.. rubric:: Footnotes

.. [#] Since the Hamming graph $H(d,q)$ is $(q-1)d$-regular, the unnormalized Laplacian and the symmetric normalized Laplacian coincide up to a multiplication by $(q-1)d$. Thus their eigenfunctions are the same and eigenvalues coincide up to this multiplication. For better numerical stability, we use symmetric normalized Laplacian in the implementation and assume its use throughout this page.
