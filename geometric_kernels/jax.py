"""
Load Jax backend in lab, spherical_harmonics and geometric_kernels using:

.. code-block::

    import geometric_kernels.jax
"""

import logging

import lab.jax  # noqa
import spherical_harmonics.jax  # noqa

import geometric_kernels.lab_extras.jax  # noqa

logging.getLogger(__name__).info("JAX backend enabled.")
