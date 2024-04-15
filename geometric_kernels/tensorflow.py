"""
Loads TensorFlow backend in lab, spherical_harmonics and geometric_kernels.

..note::
    A tutorial on the JAX backend is available in the
    :doc:`backends/TensorFlow_Graph.ipynb </examples/backends/TensorFlow_Graph>`
    notebook.
"""

import logging

import lab.tensorflow  # noqa
import spherical_harmonics.tensorflow  # noqa

import geometric_kernels.lab_extras.tensorflow  # noqa

logging.getLogger(__name__).info("Tensorflow backend enabled.")
