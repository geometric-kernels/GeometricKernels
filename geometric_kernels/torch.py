"""
Loads PyTorch backend in lab, spherical_harmonics and geometric_kernels.
"""

import logging

import lab.torch  # noqa
import spherical_harmonics.torch  # noqa

import geometric_kernels.lab_extras.torch  # noqa

logging.getLogger(__name__).info("Torch backend enabled.")
