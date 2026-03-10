##########################################
  Feature Maps and Sampling
##########################################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :func:`~.kernels.default_feature_map`, following the example notebook on the specific space of interest.

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

.. note::
    If the feature map is complex-valued $\widetilde{\phi}: X \to \mathbb{C}^M$, then 

    .. math:: k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}} \approx \mathrm{Re} \langle \widetilde{\phi}(x), \widetilde{\phi}(x') \rangle_{\mathbb{C}^M}.

Such approximate finite-dimensional feature maps can be used to speed up computations, as in, for example, :cite:t:`rahimi2007`.
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
