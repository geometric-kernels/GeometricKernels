#################
Addition Theorem
#################


**Warning:** you can get by fine without reading this page for almost all use cases, just use the standard :class:`MaternGeometricKernel <geometric_kernels.kernels.MaternGeometricKernel>`, following the example notebook `on hypersheres <https://github.com/GPflow/GeometricKernels/blob/main/notebooks/Hypersphere.ipynb>`_. This is optional material meant to explain the basic theory and based mainly on `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_.

======================
Theory
======================

This builds on the general :doc:`theory on compact manifolds <compact>` and uses same notation.

Consider, for example, a hypersphere: $M = \mathbb{S}_d$.
Then closed form expressions for $\lambda_n$ and $f_n$ are known (see e.g. Appendix B of `Borovitskiy et al. (2020) <https://arxiv.org/abs/2006.10160>`_).
The eigenfunctions $f_n$ in this case are the *(hyper)spherical harmonics*, restrictions on the unit sphere of certain known polynomials in $\mathbb{R}^{d+1}$.

However, although $\lambda_n, f_n$ are known for $M = \mathbb{S}_d$, using the general formula for $k(x, x')$ is suboptimal in this case because of the following theorem.

**Addition Theorem**.
The spherical harmonics $f_n$ can be re-indexed as $f_{j, s}$ with $j = 0, \ldots, \infty$ and $s = 1, \ldots, d_j$ with $d_j = (2j+d-1) \frac{\Gamma(j+d-1)}{\Gamma(d) \Gamma(j+1)}$ such that

* all eigenfunctions in the set $\{f_{j, s}\}_{s=1}^{d_j}$ correspond to the same eigenvalue $\lambda_j = j(j+d-1)$, where $\Gamma$ is the gamma function, $\Gamma(l) = (l-1)!$ for integer $l > 0$,

* the following equation holds 
  $$
  \sum_{s=1}^{d_j} f_{j, s}(x) f_{j, s}(x')
  =
  c_{j, d} \mathcal{C}^{(d-1)/2}_j(\cos(\mathrm{d}_{\mathbb{S}_d}(x, x')))
  \qquad
  c_{j, d}
  =
  d_j \frac{\Gamma((d+1)/2)}{2 \pi^{(d+1)/2} \mathcal{C}_j^{(d-1)/2}(1)}
  ,
  $$
  where $\mathcal{C}^{(d-1)/2}_j$ are certain known polynomials called *Gegenbauer polynomials* and $\mathrm{d}_{\mathbb{S}_d}$ is the geodesic distance on the (hyper)sphere.

Thanks to this, we have

$$
k_{\nu, \kappa}(x,x')
=
\frac{1}{C_{\nu, \kappa}} \sum_{j=0}^J \Phi_{\nu, \kappa}(\lambda_j) c_{j, d} \mathcal{C}^{(d-1)/2}_j(\cos(\mathrm{d}_{\mathbb{S}_d}(x, x')))
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
which is more efficient to use than the general formula above. The reason is simple: it is not harder to evaluate a Gegenbauer polynomial $\mathcal{C}^{(d-1)/2}_j$ than each single one of the respective (hyper)spherical harmonics.
And you need much less Gegenbauer polynomials to achieve the same quality of approximation.
For example, for $M = \mathbb{S}_2$ and $J = 20$ the corresponding $N$ is $441$.

Such addition theorems appear beyond hyperspheres, for example for Lie groups and other compact homogeneous spaces.
In the library, such spaces use the class :class:`EigenfunctionWithAdditionTheorem <geometric_kernels.spaces.eigenfunctions.EigenfunctionWithAdditionTheorem>` to represent the spectrum of $\Delta_{\mathcal{M}}$ instead of the simpler :class:`Eigenfunctions <geometric_kernels.spaces.eigenfunctions.Eigenfunctions>`.
For them, the *number of levels* parameter of the :class:`MaternKarhunenLoeveKernel <geometric_kernels.kernels.MaternKarhunenLoeveKernel>` maps to $J$ in the above formula.
