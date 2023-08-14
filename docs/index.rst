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

Geometric Kernels is a library that implements natural kernels (Heat, Matérn) on such non-Euclidean spaces as **Riemannian manifolds**, **graphs** and **meshes**.

The main projected application is within the Gaussian process framework.

Installation and Requirements
=============================

This is a **Python 3** library. To install it (together with the dependencies) follow the steps below.

Before doing anything, you might want to create and activate a new virtualenv environment:

.. code::

   pip install virtualenv
   virtualenv [env_name]
   source [env_name]/bin/activate


First of all, you will need to install `LAB <https://github.com/wesselb/lab>`_, the library that makes it possible for GeometricKernels to be backend-independent. To do this, follow `the instructions <https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc>`_.

After getting ``LAB``, to install GeometricKernels, run

.. code::

   pip install git+https://github.com/GPflow/GeometricKernels.git

The kernels are compatible with several backends, namely

- `NumPy <https://numpy.org/>`_,
- `TensorFlow <https://www.tensorflow.org/>`_ (can be used together with the GP library `GPflow <https://github.com/GPflow/GPflow>`_),
- `PyTorch <https://pytorch.org/>`_ (can be used together with the GP library `GPyTorch <https://gpytorch.ai/>`_),
- `JAX <https://github.com/google/jax/>`_. (can be used together with the GP library `GPJax <https://github.com/JaxGaussianProcesses/GPJax>`_)

Any backend, except for ``NumPy``, should be installed.

.. raw:: html

   <div class="bootstrap">
   <div class="accordion" id="backends">
     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="backendsHeadingOne">
         <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseOne" aria-expanded="true" aria-controls="backendsCollapseOne">
           TensorFlow Installation
         </button>
       </h2>
       <div id="backendsCollapseOne" class="accordion-collapse collapse show" aria-labelledby="backendsHeadingOne" data-bs-parent="#backends">
         <div class="accordion-body pb-0">

You need both ``tensorflow`` and ``tensorflow-probability``. You can get them by running

.. code::

   pip install tensorflow tensorflow-probability

[Optional] We support the TensorFlow-based Gaussian process library ``gpflow`` which you can install by running

.. code::

   pip install gpflow


.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="backendsHeadingTwo">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseTwo" aria-expanded="false" aria-controls="backendsCollapseTwo">
           PyTorch Installation
         </button>
       </h2>
       <div id="backendsCollapseTwo" class="accordion-collapse collapse" aria-labelledby="backendsHeadingTwo" data-bs-parent="#backends">
         <div class="accordion-body pb-0">

You can get PyTorch by running

.. code::

   pip install torch

[Optional] We support the PyTorch-based Gaussian process library ``gpytorch`` which you can install by running

.. code::

   pip install gpytorch

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="backendsHeadingThree">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseThree" aria-expanded="false" aria-controls="backendsCollapseThree">
           JAX Installation
         </button>
       </h2>
       <div id="backendsCollapseThree" class="accordion-collapse collapse" aria-labelledby="backendsHeadingThree" data-bs-parent="#backends">
         <div class="accordion-body pb-0">


To install JAX, follow `these instructions <https://github.com/google/jax#installation>`_.

[Optional] We support the JAX-based Gaussian process library ``GPJax`` which you can install by running

.. code::

   pip install gpjax

.. raw:: html

         </div>
       </div>
     </div>
   </div>
   </div>
   <br>

A Basic Example
===============

In the following example we show how to initialize the Matern52 kernel on the two-dimensional sphere and how to compute a kernel matrix for a vector of random points on the sphere.

.. raw:: html

   <div class="bootstrap">
   <div class="accordion" id="example">
     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="exampleHeadingOne">
         <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseOne" aria-expanded="true" aria-controls="exampleCollapseOne">
           Numpy
         </button>
       </h2>
       <div id="exampleCollapseOne" class="accordion-collapse collapse show" aria-labelledby="exampleHeadingOne" data-bs-parent="#example">
         <div class="accordion-body pb-0">

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

   >>> # Initialize kernel, use 10 levels to approximate the infinite series.
   >>> kernel = MaternKarhunenLoeveKernel(hypersphere, 10)
   >>> params, state = kernel.init_params_and_state()
   >>> params["nu"] = np.array([5/2])
   >>> params["lengthscale"] = np.array([1.])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(kernel.K(params, state, xs))
   [[0.00855354 0.00305004 0.00305004]
    [0.00305004 0.00855354 0.00305004]
    [0.00305004 0.00305004 0.00855354]]

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="exampleHeadingTwo">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseTwo" aria-expanded="false" aria-controls="exampleCollapseTwo">
           TensorFlow
         </button>
       </h2>
       <div id="exampleCollapseTwo" class="accordion-collapse collapse" aria-labelledby="exampleHeadingTwo" data-bs-parent="#example">
         <div class="accordion-body pb-0">

.. doctest:: python

   >>> import numpy as np
   >>> # Import a backend.
   >>> import tensorflow as tf
   >>> # Import the geometric_kernels backend.
   >>> import geometric_kernels.tensorflow
   >>> # Import a space and an appropriate kernel.
   >>> from geometric_kernels.spaces.hypersphere import Hypersphere
   >>> from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Generate 3 random points on the sphere.
   >>> xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel, use 10 levels to approximate the infinite series.
   >>> kernel = MaternKarhunenLoeveKernel(hypersphere, 10)
   >>> params, state = kernel.init_params_and_state()
   >>> params["nu"] = tf.convert_to_tensor(5/2)
   >>> params["lengthscale"] = tf.convert_to_tensor(1.)

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(kernel.K(params, state, tf.convert_to_tensor(xs)).numpy())
   [[0.00855354 0.00305004 0.00305004]
    [0.00305004 0.00855354 0.00305004]
    [0.00305004 0.00305004 0.00855354]]

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="exampleHeadingThree">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseThree" aria-expanded="false" aria-controls="exampleCollapseThree">
           PyTorch
         </button>
       </h2>
       <div id="exampleCollapseThree" class="accordion-collapse collapse" aria-labelledby="exampleHeadingThree" data-bs-parent="#example">
         <div class="accordion-body pb-0">

.. doctest:: python

   >>> import numpy as np
   >>> # Import a backend.
   >>> import torch
   >>> # Import the geometric_kernels backend.
   >>> import geometric_kernels.torch
   >>> # Import a space and an appropriate kernel.
   >>> from geometric_kernels.spaces.hypersphere import Hypersphere
   >>> from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Generate 3 random points on the sphere.
   >>> xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel, use 10 terms to approximate the infinite series.
   >>> kernel = MaternKarhunenLoeveKernel(hypersphere, 10)
   >>> params, state = kernel.init_params_and_state()
   >>> params["nu"] = torch.tensor(5/2)
   >>> params["lengthscale"] = torch.tensor(1.)

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(kernel.K(params, state, torch.from_numpy(xs)).detach().cpu().numpy())
   [[0.00855354 0.00305004 0.00305004]
    [0.00305004 0.00855354 0.00305004]
    [0.00305004 0.00305004 0.00855354]]

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item">
       <h2 class="accordion-header mb-0" id="exampleHeadingFour">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseFour" aria-expanded="false" aria-controls="exampleCollapseFour">
           JAX
         </button>
       </h2>
       <div id="exampleCollapseFour" class="accordion-collapse collapse" aria-labelledby="exampleHeadingFour" data-bs-parent="#example">
         <div class="accordion-body pb-0">

.. doctest:: python

   >>> import numpy as np
   >>> # Import a backend.
   >>> import jax.numpy as jnp
   >>> # Import the geometric_kernels backend.
   >>> import geometric_kernels.jax
   >>> # Import a space and an appropriate kernel.
   >>> from geometric_kernels.spaces.hypersphere import Hypersphere
   >>> from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Generate 3 random points on the sphere.
   >>> xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel, use 10 levels to approximate the infinite series.
   >>> kernel = MaternKarhunenLoeveKernel(hypersphere, 10)
   >>> params, state = kernel.init_params_and_state()
   >>> params["nu"] = jnp.r_[5/2]
   >>> params["lengthscale"] = jnp.r_[1.]

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(kernel.K(params, state, jnp.array(xs)))
   [[0.00855354 0.00305004 0.00305004]
    [0.00305004 0.00855354 0.00305004]
    [0.00305004 0.00305004 0.00855354]]


.. raw:: html

         </div>
       </div>
     </div>
   </div>
   </div>
   <br>

You can find more examples in our `example notebooks <https://github.com/GPflow/GeometricKernels/tree/main/notebooks>`_.

.. toctree::
   :hidden:

   Geometric Kernels <self>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   Examples <examples/index>
   API reference <autoapi/geometric_kernels/index>
   GitHub <https://github.com/GPflow/GeometricKernels>

