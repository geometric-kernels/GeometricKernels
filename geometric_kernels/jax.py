"""
Loads JAX backend in lab, spherical_harmonics and geometric_kernels.

..note::
    A tutorial on the JAX backend is available in the
    :doc:`backends/JAX_Graph.ipynb </examples/backends/JAX_Graph>` notebook.
"""

import logging

import lab.jax  # noqa
import spherical_harmonics.jax  # noqa

import geometric_kernels.lab_extras.jax  # noqa

logging.getLogger(__name__).info("JAX backend enabled.")
