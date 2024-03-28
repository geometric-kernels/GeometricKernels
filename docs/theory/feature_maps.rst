##########################################
  Feature Maps and Sampling
##########################################

**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`default_feature_map <geometric_kernels.kernels.default_feature_map>`, following the example notebook on the specific space of interest.
This is optional material meant to explain the basic theory.

=======
Theory
=======

Any kernel $k: X \times X \to \mathbb{R}$ satisfies $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$ for some Hilbert space $\mathcal{H}$ and some *feature map* $\phi: X \to \mathcal{H}$.

Usually, the Hilbert space $\mathcal{H}$ is infinite-dimensional, rendering $\phi$ intractable.
However, we can often find an *approximate* feature map $\widetilde{\phi}: X \to \mathbb{R}^M$, such that
$$
k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}} \approx \langle \widetilde{\phi}(x), \widetilde{\phi}(x') \rangle_{\mathbb{R}^M}.
$$

Such approximate finite-dimensional feature maps can be used to speed up computations, as in, for example, `Rahimi and Recht (2007) <https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf>`_.
Importantly, it can be used to efficiently sample (without incurring cubic costs) the Gaussian process $f \sim \mathrm{GP}(0, k)$.
The key idea is that
$$
f(x) \approx \sum_{m=1}^M w_j \widetilde{\phi}_j(x)
,
\qquad
w_j \sim \mathrm{N}(0, 1)
,
\qquad
\widetilde{\phi}(x) = (\widetilde{\phi}_1(x), \ldots, \widetilde{\phi}_M(x))
.
$$

Matérn kernels on various spaces usually possess natural approximate finite-dimensional feature maps.
In some cases, these are deterministic, in others—random.
For the specific constructions, we refer the reader to the theory on specific spaces and the respective papers.
