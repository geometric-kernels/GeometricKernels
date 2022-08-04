**WARNING: this is a development (unstable) version of the package**

# Geometric Kernels
[![Quality checks and Tests](https://github.com/GPflow/GeometricKernels/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/GPflow/GeometricKernels/actions/workflows/quality-checks.yaml)
[![Documentation](https://github.com/GPflow/GeometricKernels/actions/workflows/docs.yaml/badge.svg)](https://github.com/GPflow/GeometricKernels/actions/workflows/docs.yaml)

Geometric Kernels is a library that implements natural kernels (Heat, Matérn) on such non-Euclidean spaces as **Riemannian manifolds** or **graphs**.

The main projected application is within the Gaussian process framework.
Some of the features are specifically inspired by it, most notably by the problem of efficient sampling.


##  Installation

0. [Optionally] install and run virtualenv
```bash
[sudo] pip install virtualenv
virtualenv [env_name]
source [env_name]/bin/activate
```

1. [Prerequisite] install [LAB](https://github.com/wesselb/lab) following [these instructions](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

2. Install the library in the active environment by running
```bash
pip install git+https://github.com/gpflow/geometrickernels.git
```

4. Install a backend of your choice

We use [LAB](https://github.com/wesselb/lab) to support multiple backends (e.g., TensorFlow, Jax, PyTorch). However, you are not required to install all of them on your system to use the Geometric Kernel package. Simply install the backend (and (optionally) a GP package) of your choice. For example,

- TensorFlow and GPflow
```
pip install tensorflow tensorflow-probability gpflow
```

- PyTorch and GPyTorch
```
pip install torch gpytorch
```

- JAX (the cpu version)
```
pip install "jax[cpu]"
```

### Supported backends with associated GP packaes

Ready|Backend                                      | GP package
-----|---------------------------------------------|------------------------------------------
✅   |[Tensorflow](https://www.tensorflow.org/)    |[GPflow](https://github.com/GPflow/GPflow)
✅   |[PyTorch](https://github.com/pytorch/pytorch)|[GPyTorch](https://gpytorch.ai/)
✅   |[Numpy](https://numpy.org/)                  | -
✅   |[JAX](https://github.com/google/jax)         | -
 
## A basic example

This example shows how to compute a 3x3 kernel matrix for the Matern52 kernel on the standard two-dimensional sphere. It relies on the numpy-based backend. Look up the information on how to use other backends in [the documentation](TODO).

```python
# Import a backend.
import numpy as np
# Import the geometric_kernels backend.
import geometric_kernels
# Import a space and an appropriate kernel.
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

# Create a manifold (2-dim sphere).
hypersphere = Hypersphere(dim=2)

# Generate 3 random points on the sphere.
xs = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])

# Initialize kernel, use 100 terms to approximate the infinite series.
kernel = MaternKarhunenLoeveKernel(hypersphere, 100)
params, state = kernel.init_params_and_state()
params["nu"] = np.array([5/2])
params["lengthscale"] = np.array([1.])

# Compute and print out the 3x3 kernel matrix.
print(kernel.K(params, state, xs))
```

This should output
```
[[0.00855354 0.00305004 0.00305004]
 [0.00305004 0.00855354 0.00305004]
 [0.00305004 0.00305004 0.00855354]]
```

## Documentation

The documentation for GeometricKernels is available on a [separate website](TODO).

## For development and running the tests

Run these commands from the root directory of the repository. 

Install all backends and the dev requirements (Pytest, black, etc.)
```bash
pip install -r dev_requirements.txt -r requirements.txt
```

Run the tests
```bash
make test
```

## The structure of the library
<img alt="class diagram" src="docs/class_diagram.svg">
