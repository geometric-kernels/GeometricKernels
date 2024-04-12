#################
Addition Theorem
#################


**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the example notebook :doc:`example notebook on hypersheres </examples/Hypersphere>`.
This is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.

======================
Theory
======================

This builds on the general :doc:`theory on compact manifolds <compact>` and uses the same notation.

Consider, for example, a hypersphere: $M = \mathbb{S}_d$.
Then closed form expressions for $\lambda_j$ and $f_j$ are known (see e.g. Appendix B of `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_).
The eigenfunctions $f_j$ in this case are the *(hyper)spherical harmonics*, restrictions on the unit sphere of certain known polynomials in $\mathbb{R}^{d+1}$.

However, although $\lambda_j, f_j$ are known for $M = \mathbb{S}_d$, using the general formula for $k(x, x')$ is suboptimal in this case because of the following theorem.

**Addition Theorem**.
The spherical harmonics $f_j$ can be re-indexed as $f_{l s}$ with $l = 0, \ldots, \infty$ and $s = 1, \ldots, d_l$ with $d_l = (2l+d-1) \frac{\Gamma(l+d-1)}{\Gamma(d) \Gamma(l+1)}$ such that

* all eigenfunctions in the set $\{f_{l s}\}_{s=1}^{d_l}$ correspond to the same eigenvalue $\lambda_l = l(l+d-1)$, where $\Gamma$ is the gamma function, $\Gamma(j) = (j-1)!$ for integer $j > 0$,

* the following equation holds 
  $$
  \sum_{s=1}^{d_l} f_{l s}(x) f_{l s}(x')
  =
  c_{l, d} \mathcal{C}^{(d-1)/2}_l(\cos(\mathrm{d}_{\mathbb{S}_d}(x, x')))
  \qquad
  c_{l, d}
  =
  d_l \frac{\Gamma((d+1)/2)}{2 \pi^{(d+1)/2} \mathcal{C}_l^{(d-1)/2}(1)}
  ,
  $$
  where $\mathcal{C}^{(d-1)/2}_l$ are certain known polynomials called *Gegenbauer polynomials* and $\mathrm{d}_{\mathbb{S}_d}$ is the geodesic distance on the (hyper)sphere.

Thanks to this, we have

$$
k_{\nu, \kappa}(x,x')
=
\frac{1}{C_{\nu, \kappa}} \sum_{l=0}^{L-1} \Phi_{\nu, \kappa}(\lambda_l) c_{l, d} \mathcal{C}^{(d-1)/2}_l(\cos(\mathrm{d}_{\mathbb{S}_d}(x, x')))
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
which is more efficient to use than the general formula above. The reason is simple: it is not harder to evaluate a Gegenbauer polynomial $\mathcal{C}^{(d-1)/2}_l$ than each single one of the respective (hyper)spherical harmonics.
And you need much fewer Gegenbauer polynomials to achieve the same quality of approximation.
For example, for $M = \mathbb{S}_2$ and $L = 20$ the corresponding $J$ is $400$.

.. note::
    The $l$ in the example above indexes what we call *levels* in the library.
    These are certain sets of eigenfunctions that correspond to the same eigenvalue (not necessarily a maximal set of those), for which one can efficiently compute the outer product $\sum_{s} f_{l s}(x) f_{l s}(x')$ without having to compute the individual eigenfunctions.

In the simplest special case of $\mathbb{S}_d$, the circle $\mathbb{S}_1$, the eigenfunctions are given by $\sin(l \theta), \cos(l \theta)$, where $l$ indexes levels.
The outer product $\cos(l \theta) \cos(l \theta') + \sin(l \theta) \sin(l \theta')$ in this case can be simplified to $\cos(l (\theta-\theta')) = \cos(l d_{\mathbb{S}_1}(\theta, \theta'))$ thanks to an elementary trigonometric identity.

Such addition theorems appear beyond hyperspheres, for example for Lie groups and other compact homogeneous spaces.
In the library, such spaces use the class :class:`~.spaces.eigenfunctions.EigenfunctionsWithAdditionTheorem` to represent the spectrum of $\Delta_{\mathcal{M}}$ instead of the simpler :class:`~.spaces.eigenfunctions.Eigenfunctions`.
For them, the *number of levels* parameter of the :class:`~.kernels.MaternKarhunenLoeveKernel` maps to $L$ in the above formula.
