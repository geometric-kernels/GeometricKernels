"""
Loads PyTorch backend in lab, spherical_harmonics and geometric_kernels.

..note::
    A tutorial on the JAX backend is available in the
    :doc:`backends/PyTorch_Graph.ipynb </examples/backends/PyTorch_Graph>`
    notebook.
"""

import logging

import lab.torch  # noqa
import spherical_harmonics.torch  # noqa

import geometric_kernels.lab_extras.torch  # noqa

logging.getLogger(__name__).info("Torch backend enabled.")
