##########################################
  Kernels on Non-compact Symmetric Spaces
##########################################

.. warning::
    You can get by fine without reading this page for almost all use cases, just use the standard :class:`~.kernels.MaternGeometricKernel`, following the example notebooks :doc:`on hyperbolic spaces </examples/Hyperbolic>` and :doc:`on the space of symmetric positive definite matrices (SPD) </examples/SPD>`.

    This is optional material meant to explain the basic theory and based mainly on :cite:t:`azangulov2024b`.

=======
Theory
=======

The theory for *non-compact symmetric spaces*—like hyperbolic spaces or manifolds of symmetric positive definite matrices (endowed with the affine-invariant metric)—is quite different from the theory for *discrete spectrum spaces* such as compact manifolds, graphs or meshes.
For the latter, kernels are given by a finite sum or an infinite series and are approximated using *truncation*.
For the former, kernels are given by integrals and are approximated using *Monte Carlo*.

More specifically, for non-compact symmetric spaces, there exists an analog of the *random Fourier features* technique of :cite:t:`rahimi2007`.
In the Euclidean case, closed form expressions for kernels are available and random Fourier features are only used to speed up computations.
No closed form expressions for kernels are usually available on other non-compact symmetric spaces.
Because of that, random Fourier features are the basic means of computing the kernels in this case.

A complete mathematical treatise can be found in :cite:t:`azangulov2024b`.
Here we briefly present the main ideas.
Recall that the usual Euclidean random Fourier features boil down to

$$
k(x, x') = \int_{\mathbb{R}^d} S(\lambda) e^{2 \pi i \langle x - x', \lambda \rangle} \mathrm{d} \lambda \approx \frac{1}{L} \sum_{l=1}^L e^{2 \pi i \langle x - x', \lambda_l\rangle}
\qquad
\lambda_l \sim S(\lambda)
$$
where $S(\cdot)$ is the spectral density of the kernel $k$.
For Matérn kernels, $S(\cdot)$ coincides with the Gaussian density if $\nu = \infty$ and with the Student's $t$ density with $\nu$ degrees of freedom if $\nu < \infty$.

On a non-compact symmetric space, the following holds instead:
$$
k(x, x') = \int_{\mathbb{R}^r} S(\lambda) \pi^{(\lambda)}(x, x') c(\lambda)^{-2} \mathrm{d} \lambda \approx \frac{1}{L} \sum_{l=1}^L \pi^{(\lambda_l)}(x, x')
\qquad
\lambda_l \sim c(\lambda)^{-2} S(\lambda)
$$
Here,

* $r$ is called the *rank* of the symmetric space,

* $\pi^{(\lambda)}$ are called *zonal spherical functions*,

* $c(\lambda)$ is called the *Harish-Chandra's $c$ function*.

Both $r$ and $c$ can be computed exactly using algebraic-only considerations.
On the other hand, $\pi^{(\lambda_l)}(x, x')$ are integrals that require numerical approximation.
There are multiple ways to do this.
The most important one is as follows:
$$
\pi^{(\lambda_l)}(x, x') = \mathbb{E}_{h \sim \mu_H}
e^{\langle i \lambda + \rho, \,a(h, x)\rangle}
\overline{
e^{\langle i \lambda + \rho, \,a(h, x')\rangle}}
\approx
\frac{1}{P} \sum_{p=1}^P
e^{\langle i \lambda + \rho, \,a(h_p, x)\rangle}
\overline{
e^{\langle i \lambda + \rho, \,a(h_p, x')\rangle}}
\qquad
h_p \sim \mu_H
$$
where, this time,

* $\mu_H$ is some measure which is usually easy to sample from,

* $i$ is the imaginary unit,

* $a(\cdot, \cdot)$ is a function that can be computed exactly using algebraic-only considerations.

The right-hand side here is an inner product.
Same is true for the result of substituting this approximation of $\pi^{(\lambda_l)}(x, x')$ into the approximation of $k(x, x')$ above.
More specifically, defining
$$
\phi(x) =
\frac{1}{\sqrt{P L}}
(
e^{\langle i \lambda_1 + \rho, \,a(h_1, x)\rangle},
\ldots,
e^{\langle i \lambda_1 + \rho, \,a(h_P, x)\rangle},
\ldots,
e^{\langle i \lambda_L + \rho, \,a(h_1, x)\rangle},
\ldots,
e^{\langle i \lambda_L + \rho, \,a(h_P, x)\rangle})
$$
we have
$$
k(x, x') \approx \langle \phi(x), \phi(x') \rangle_{\mathbb{C}^{P L}}.
$$

.. note::
   Typically, in practice we set $P = 1$. This is akin to how it is commonly done for the random phase Fourier feature approximation in the Euclidean case, as in Equation (2) of :cite:t:`sutherland2015`.

For non-compact symmetric spaces, :class:`~.kernels.MaternGeometricKernel` is an alias to :class:`~.kernels.MaternFeatureMapKernel`.
The latter is a kernel defined in terms of feature map just like in the equation above.
The feature map is exactly the $\phi(\cdot)$ above, implemented as :class:`~.feature_maps.RejectionSamplingFeatureMapHyperbolic` for hyperbolic spaces and as :class:`~.feature_maps.RejectionSamplingFeatureMapSPD` for manifolds of symmetric positive definite matrices.
