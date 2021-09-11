# Geometric Kernels

This package implements a suite of Geometric kernels.

<img alt="class diagram" src="docs/class_diagram.svg">


## Requirements and Installation

We recommend creating a virtual environment before installing the package.

First, follow [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply run
```
pip install -e .
```
We use [LAB](https://github.com/wesselb/lab) to support multiple backends (e.g., TensorFlow, Jax, PyTorch). However, you are not required to install all of them on your system to use the Geometric Kernel package. Simply install the backend (and GP package) of your choice. For instance,

- TensorFlow and GPflow
```
pip install tensorflow tensorflow-probability gpflow
```
then import as follows
```
import geometric_kernels.tensorflow  # noqa
import tensorflow as tf
```

- PyTorch and GPyTorch
```
pip install torch gpytorch
```
then import as follows
```
import geometric_kernels.pytorch  # noqa
import torch
```


### Supported backends with associated GP packaes

Ready|Backend                                      | GP package
-----|---------------------------------------------|------------------------------------------
✅   |[Tensorflow](https://www.tensorflow.org/)    |[GPflow](https://github.com/GPflow/GPflow)
✅   |[PyTorch](https://github.com/pytorch/pytorch)|[GPyTorch](https://gpytorch.ai/)
✅   |[Numpy](https://numpy.org/)                  |??
🚧   |[Jax](https://github.com/google/jax)         |??
 

## Running the tests

Run these commands from the root directory of this repository. 
To run the full test suite, including pylint and Mypy, run: 

```bash
poetry run task test
```

Alternatively, you can run just the unit tests, starting with the failing tests and exiting after
the first test failure:

```bash
poetry run task quicktest
```

**NOTE:** Running the tests requires
that the project virtual environment has been updated. See [Installation](#Installation).

## Adding new Python dependencies

- To specify dependencies required by `GeometricKernels`, run `poetry add`.
- To specify dependencies required to build or test the project, run `poetry add --dev`.
