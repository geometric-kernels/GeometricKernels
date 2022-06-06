"""
The library root. The kernel classes are contained within the
:py:mod:`kernels <geometric_kernels.kernels>` package. They need to be paired
with one of the space classes from the
:py:mod:`spaces <geometric_kernels.spaces>` package.

The :py:mod:`frontends <geometric_kernels.frontends>` package contains kernel
wrapper classes compatible with Gaussian process libraries like
`GPFlow <https://www.gpflow.org/>`_ and `GPyTorch <https://gpytorch.ai/>`_.

The :py:mod:`lab_extras <geometric_kernels.lab_extras>` package contains our
custom additions to `LAB <https://github.com/wesselb/lab>`_, the framework that
allows our library to be backend-independent. This is an internal part of our
library.
"""

import geometric_kernels._logging  # noqa: F401
from geometric_kernels.lab_extras import *
