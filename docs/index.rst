.. Copyright 2020 The Geometric Kernels Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. Geometric Kernels documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Overview
========

Geometric Kernels is a library that implements natural kernels (Heat, Matérn) on such non-Euclidean spaces as **Riemannian manifolds** or **graphs**.

The main projected application is within the Gaussian process framework.
Some of the features are specifically inspired by it, most notably by the problem of efficient sampling.

Installation and Requirements
=============================

This is a **Python 3** library.
To install Geometric Kernels, run

.. code::

   $ pip install git+https://github.com/GPflow/GeometricKernels.git

The kernels are compatible with several backends:

- `TensorFlow <https://www.tensorflow.org/>`_ (can be used together with the GP library `GPflow <https://github.com/GPflow/GPflow>`_),
- `PyTorch <https://pytorch.org/>`_ (can be used together with the GP library `GPyTorch <https://gpytorch.ai/>`_),
- `NumPy <https://numpy.org/>`_,
- `JAX <https://github.com/google/jax/>`_.

A Basic Example
===============

In the following example we show how to initialize the Matern52 kernel on the two-dimensional sphere and how to compute a kernel matrix for a vector of random points on the sphere.

.. doctest:: python

   >>> # Import a backend.
   >>> import numpy as np
   >>> # Import the geometric_kernels backend.
   >>> import geometric_kernels
   >>> # Import a space and an appropriate kernel.
   >>> from geometric_kernels.spaces.hypersphere import Hypersphere
   >>> from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Generate 3 random points on the sphere.
   >>> xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel, use 100 terms to approximate the infinite series.
   >>> kernel = MaternKarhunenLoeveKernel(hypersphere, 100)
   >>> params, state = kernel.init_params_and_state()
   >>> params["nu"] = np.array([5/2])
   >>> params["lengthscale"] = np.array([1.])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(kernel.K(params, state, xs))
   [[0.00855354 0.00305004 0.00305004]
    [0.00305004 0.00855354 0.00305004]
    [0.00305004 0.00305004 0.00855354]]


We used NumPy above. To use other backends (PyTorch, TensorFlow, JAX), the line

.. code::

   import geometric_kernels

should be changed into the line

.. code::

   import geometric_kernels.<backend>

where :code:`<backend>` may be one of :code:`torch`, :code:`tensorflow` or :code:`jax`.
Of course, the relevant type of tensors should be used instead of :code:`np.array` as well.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:

   Geometric Kernels <self>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   API reference <autoapi/geometric_kernels/index>
