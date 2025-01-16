# GeometricKernels

[![Quality checks and Tests](https://github.com/geometric-kernels/GeometricKernels/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/geometric-kernels/GeometricKernels/actions/workflows/quality-checks.yaml)
[![Documentation](https://github.com/geometric-kernels/GeometricKernels/actions/workflows/docs.yaml/badge.svg)](https://geometric-kernels.github.io/GeometricKernels/index.html)
[![Landing Page](https://img.shields.io/badge/Landing_Page-informational)](https://geometric-kernels.github.io/)

[![GeometricKernels](https://geometric-kernels.github.io/assets/title-sm.png)](https://geometric-kernels.github.io/)

GeometricKernels is a library that implements kernels &mdash; most importantly, heat and Matérn kernels &mdash; on non-Euclidean spaces such as **Riemannian manifolds**, **graphs** and **meshes**.
This enables kernel methods &mdash; in particular Gaussian process models &mdash; to be deployed on such spaces.

## Installation

0. [Optionally] create and activate a new virtual environment.

    You can use Conda

    ```bash
    conda create -n [env_name] python=3.[version]
    conda activate [env_name]
    ```

    or virtualenv

    ```bash
    virtualenv [env_name]
    source [env_name]/bin/activate
    ```

1. Install the library in the active environment by running

    ```bash
    pip install geometric_kernels
    ```

    If you want to install specific GitHub branch called `[branch]`, run

    ```bash
    pip install "git+https://github.com/geometric-kernels/GeometricKernels@[branch]"
    ```

2. Install a backend of your choice

    We use [LAB](https://github.com/wesselb/lab) to support multiple backends (e.g., TensorFlow, Jax, PyTorch). However, you are not required to install all of them on your system to use the GeometricKernels package. Simply install the backend (and (optionally) a GP package) of your choice. For example,

    - [Tensorflow](https://www.tensorflow.org/)

        ```
        pip install tensorflow tensorflow-probability
        ```

        Optionally, you can install the Tensorflow-based Gaussian processes library [GPflow](https://github.com/GPflow/GPflow), for which we provide a frontend.

        ```
        pip install gpflow
        ```

    - [PyTorch](https://pytorch.org/)

        ```
        pip install torch
        ```

        Optionally, you can install the PyTorch-based Gaussian processes library [GPyTorch](https://gpytorch.ai/), for which we provide a frontend.

        ```
        pip install gpytorch
        ```

    - [JAX](https://jax.readthedocs.io/) (the cpu version)—the gpu and tpu versions can be installed [similarly](https://jax.readthedocs.io/en/latest/installation.html).

        ```
        pip install "jax[cpu]"
        ```

        Optionally, you can install the JAX-based Gaussian processes library [GPJax](https://github.com/JaxGaussianProcesses/GPJax), for which we provide a frontend.

        ```
        pip install gpjax
        ```

        **Note**. Currently, only some versions of `gpjax` are supported (we tested `gpjax==0.6.9`).

        Furthermore, installation might be far from trivial and result in a broken environment. This is due to our conflicting dependencies, see https://github.com/JaxGaussianProcesses/GPJax/issues/441.

## A basic example

This example shows how to compute a 3x3 kernel matrix for the Matern52 kernel on the standard two-dimensional sphere. It relies on the numpy-based backend. Look up the information on how to use other backends in [the documentation](https://geometric-kernels.github.io/GeometricKernels/index.html).

```python
# Import a backend.
import numpy as np
# Import the geometric_kernels backend.
import geometric_kernels
# Import a space and an appropriate kernel.
from geometric_kernels.spaces import Hypersphere
from geometric_kernels.kernels import MaternGeometricKernel

# Create a manifold (2-dim sphere).
hypersphere = Hypersphere(dim=2)

# Define 3 points on the sphere.
xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

# Initialize kernel.
kernel = MaternGeometricKernel(hypersphere)
params = kernel.init_params()
params["nu"] = np.array([5/2])
params["lengthscale"] = np.array([1.])

# Compute and print out the 3x3 kernel matrix.
print(np.around(kernel.K(params, xs), 2))
```

This should output

```
[[1.   0.36 0.36]
 [0.36 1.   0.36]
 [0.36 0.36 1.  ]]
```

## Documentation

The documentation for GeometricKernels is available on a [separate website](https://geometric-kernels.github.io/GeometricKernels/index.html).

## For development and running the tests

Run these commands from the root directory of the repository.

Install all backends and the dev requirements (Pytest, black, etc.)

```bash
make install
```

Run style checks
```bash
make lint
```

Run the tests

```bash
make test
```

## If you have a question

Post it in issues using the `"How do I do ..." and other issues` template and the "question" label.

This [link](https://github.com/geometric-kernels/GeometricKernels/issues/new?assignees=&labels=question&projects=&template=other-issue.md) chooses the right template and label for you.

## Citation

If you are using GeometricKernels, please cite the following [paper](https://arxiv.org/pdf/2407.08086):
```
@article{mostowsky2024,
      title = {The GeometricKernels Package: Heat and Matérn Kernels for Geometric Learning on Manifolds, Meshes, and Graphs},
      author = {Peter Mostowsky and Vincent Dutordoir and Iskander Azangulov and Noémie Jaquier and Michael John Hutchinson and Aditya Ravuri and Leonel Rozo and Alexander Terenin and Viacheslav Borovitskiy},
      year = {2024},
      journal = {arXiv:2407.08086},
}
```

Furthermore, please consider citing the theoretical papers the package is based on.
You can find the relevant references for any space in
- the docstring of the respective space class,
- at the end of the respective tutorial notebook.
