.. GeometricKernels documentation master file, should contain the root
   `toctree` directive.


Overview
========

GeometricKernels is a library that implements natural kernels (heat [#]_, Matérn) on such non-Euclidean spaces as **Riemannian manifolds**, **graphs** and **meshes**.

The main projected application is Gaussian processes.

Installation and Requirements
=============================

This is a **Python 3** library.

Before doing anything, you might want to create and activate a new virtual environment:

.. raw:: html

   <div class="bootstrap">
   <div class="accordion" id="virtualenvs">
     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="virtualenvsHeadingOne">
         <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#virtualenvsCollapseOne" aria-expanded="true" aria-controls="virtualenvsCollapseOne" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
           Conda
         </button>
       </h2>
       <div id="virtualenvsCollapseOne" class="accordion-collapse collapse show" aria-labelledby="virtualenvsHeadingOne" data-bs-parent="#virtualenvs">
         <div class="accordion-body pb-0">

..  code-block:: bash

   conda create -n [env_name] python=[version]
   conda activate [env_name]

where [env_name] is the name of the environment and [version] is the version of Python you want to use, we currently support 3.8, 3.9, 3.10, 3.11.

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="virtualenvsHeadingTwo">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#virtualenvsCollapseTwo" aria-expanded="false" aria-controls="virtualenvsCollapseTwo" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
           Virtualenv
         </button>
       </h2>
       <div id="virtualenvsCollapseTwo" class="accordion-collapse collapse" aria-labelledby="virtualenvsHeadingTwo" data-bs-parent="#virtualenvs">
         <div class="accordion-body pb-0">

..  code-block:: bash

   virtualenv [env_name]
   source [env_name]/bin/activate

.. raw:: html

         </div>
       </div>
     </div>
   </div>
   </div>
   <br>


To install GeometricKernels, run

..  code-block:: bash

    pip install geometric_kernels

.. note::
  If you want to install specific GitHub branch called `[branch]`, run

  ..  code-block:: bash

      pip install "git+https://github.com/geometric-kernels/GeometricKernels@[branch]"

The kernels are compatible with several backends, namely

- `NumPy <https://numpy.org/>`_,
- `TensorFlow <https://www.tensorflow.org/>`_ (can be used together with the GP library `GPflow <https://www.gpflow.org/>`_),
- `PyTorch <https://pytorch.org/>`_ (can be used together with the GP library `GPyTorch <https://gpytorch.ai/>`_),
- `JAX <https://jax.readthedocs.io/>`_ (can be used together with the GP library `GPJax <https://jaxgaussianprocesses.com/>`_).

Any backend, except for ``NumPy``, should be manually installed.

.. raw:: html

   <div class="bootstrap">
   <div class="accordion" id="backends">
     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="backendsHeadingOne">
         <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseOne" aria-expanded="true" aria-controls="backendsCollapseOne" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
           TensorFlow Installation
         </button>
       </h2>
       <div id="backendsCollapseOne" class="accordion-collapse collapse show" aria-labelledby="backendsHeadingOne" data-bs-parent="#backends">
         <div class="accordion-body pb-0">

You need both ``tensorflow`` and ``tensorflow-probability``. You can get them by running

..  code-block:: bash

   pip install tensorflow tensorflow-probability


[Optional] We support the TensorFlow-based Gaussian process library ``gpflow`` which you can install by running

..  code-block:: bash

    pip install gpflow


.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="backendsHeadingTwo">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseTwo" aria-expanded="false" aria-controls="backendsCollapseTwo" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
           PyTorch Installation
         </button>
       </h2>
       <div id="backendsCollapseTwo" class="accordion-collapse collapse" aria-labelledby="backendsHeadingTwo" data-bs-parent="#backends">
         <div class="accordion-body pb-0">

You can get PyTorch by running

..  code-block:: bash

   pip install torch

[Optional] We support the PyTorch-based Gaussian process library ``gpytorch`` which you can install by running

..  code-block:: bash

   pip install gpytorch

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="backendsHeadingThree">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#backendsCollapseThree" aria-expanded="false" aria-controls="backendsCollapseThree" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
           JAX Installation
         </button>
       </h2>
       <div id="backendsCollapseThree" class="accordion-collapse collapse" aria-labelledby="backendsHeadingThree" data-bs-parent="#backends">
         <div class="accordion-body pb-0">


To install JAX, follow `these instructions <https://github.com/google/jax#installation>`_.

[Optional] We support the JAX-based Gaussian process library ``GPJax`` which you can install by running

..  code-block:: bash

   pip install gpjax

.. warning::

    .. raw:: html

        <div style="color: var(--color-content-foreground);">

    Currently, only some versions of `gpjax` are supported (we tested `gpjax==0.6.9`).

    Furthermore, installation might be far from trivial and result in a broken environment. This is due to our conflicting dependencies, see https://github.com/JaxGaussianProcesses/GPJax/issues/441.

    .. raw:: html

        </div>

.. raw:: html

         </div>
       </div>
     </div>
   </div>
   </div>
   <br>

A Basic Example
===============

In the following example we show how to initialize the Matern52 kernel on the two-dimensional sphere and how to compute a kernel matrix for a few points on the sphere.

.. raw:: html

   <div class="bootstrap">
   <div class="accordion" id="example">
     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="exampleHeadingOne">
         <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseOne" aria-expanded="true" aria-controls="exampleCollapseOne" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
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
   >>> from geometric_kernels.spaces import Hypersphere
   >>> from geometric_kernels.kernels import MaternGeometricKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Define 3 points on the sphere.
   >>> xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel.
   >>> kernel = MaternGeometricKernel(hypersphere)
   >>> params = kernel.init_params()
   >>> params["nu"] = np.array([5/2])
   >>> params["lengthscale"] = np.array([1.])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(np.around(kernel.K(params, xs), 2))
   [[1.   0.36 0.36]
    [0.36 1.   0.36]
    [0.36 0.36 1.  ]]

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="exampleHeadingTwo">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseTwo" aria-expanded="false" aria-controls="exampleCollapseTwo" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
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
   >>> from geometric_kernels.spaces import Hypersphere
   >>> from geometric_kernels.kernels import MaternGeometricKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Define 3 points on the sphere.
   >>> xs = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel.
   >>> kernel = MaternGeometricKernel(hypersphere)
   >>> params = kernel.init_params()
   >>> params["nu"] = tf.convert_to_tensor([5/2])
   >>> params["lengthscale"] = tf.convert_to_tensor([1.])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(np.around(kernel.K(params, xs).numpy(), 2))
   [[1.   0.36 0.36]
    [0.36 1.   0.36]
    [0.36 0.36 1.  ]]


.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="exampleHeadingThree">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseThree" aria-expanded="false" aria-controls="exampleCollapseThree" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
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
   >>> from geometric_kernels.spaces import Hypersphere
   >>> from geometric_kernels.kernels import MaternGeometricKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Define 3 points on the sphere.
   >>> xs = torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel.
   >>> kernel = MaternGeometricKernel(hypersphere)
   >>> params = kernel.init_params()
   >>> params["nu"] = torch.tensor([5/2])
   >>> params["lengthscale"] = torch.tensor([1.])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(np.around(kernel.K(params, xs).detach().cpu().numpy(), 2))
   [[1.   0.36 0.36]
    [0.36 1.   0.36]
    [0.36 0.36 1.  ]]

.. raw:: html

         </div>
       </div>
     </div>

.. raw:: html

     <div class="accordion-item" style="background-color: var(--color-background-primary);">
       <h2 class="accordion-header mb-0" id="exampleHeadingFour">
         <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#exampleCollapseFour" aria-expanded="false" aria-controls="exampleCollapseFour" style="background-color: var(--color-background-secondary); color: var(--color-foreground-primary);">
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
   >>> from geometric_kernels.spaces import Hypersphere
   >>> from geometric_kernels.kernels import MaternGeometricKernel

   >>> # Create a manifold (2-dim sphere).
   >>> hypersphere = Hypersphere(dim=2)

   >>> # Define 3 points on the sphere.
   >>> xs = jnp.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

   >>> # Initialize kernel.
   >>> kernel = MaternGeometricKernel(hypersphere)
   >>> params = kernel.init_params()
   >>> params["nu"] = jnp.array([5/2])
   >>> params["lengthscale"] = jnp.array([1.0])

   >>> # Compute and print out the 3x3 kernel matrix.
   >>> print(np.around(np.asarray(kernel.K(params, xs)), 2))
   [[1.   0.36 0.36]
    [0.36 1.   0.36]
    [0.36 0.36 1.  ]]


.. raw:: html

         </div>
       </div>
     </div>
   </div>
   </div>
   <br>

You can find more examples :doc:`here <examples/index>`.

Citation
========

If you are using GeometricKernels, please consider citing the theoretical papers it is based on.

You can find the relevant references for any space in

- the docstring of the respective space class,
- at the end of the respective tutorial notebook.

.. rubric:: Footnotes

.. [#] The heat kernel (or diffusion kernel) is a far-reaching generalization of the RBF kernel (a.k.a. Gaussian kernel, or squared exponential kernel). It can be considered to be a Matérn kernel with smoothness parameter :math:`\nu = \infty`, as we do in this library.

.. toctree::
   :hidden:

   GeometricKernels <self>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   Examples <examples/index>
   Theory <theory/index>
   API reference <autoapi/geometric_kernels/index>
   Bibliography <bibliography>
   GitHub <https://github.com/geometric-kernels/GeometricKernels>

