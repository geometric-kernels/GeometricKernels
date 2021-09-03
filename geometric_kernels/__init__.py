"""
Root __init__. Sets 'BACKEND'.
"""
import logging
import os

import geometric_kernels._logging  # noqa: F401

_SUPPORTED_BACKENDS = [
    "tensorflow",
    "pytorch",
    "numpy",
    # "jax",  TODO
]


def _read_backend() -> str:
    """
    Reads the environment variable 'GEOMETRIC_KERNELS_BACKEND' to retrieve the
    backend. If it is not set it defaults to numpy.
    """
    value = os.environ.get("GEOMETRIC_KERNELS_BACKEND", "numpy").lower()
    if value not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "Unknown value exported to 'GEOMETRIC_KERNELS_BACKEND' environment variable. "
            "Supported backends are 'tensorflow', 'pytorch', 'numpy' (default)."
        )
    return value


BACKEND = _read_backend()
"""
Backend used for computations. Supported values are 'tensorflow', 'pytorch', and 'numpy' (default).
Value is set using the 'GEOMETRIC_KERNELS_BACKEND' environment variable.

.. code :
    export GEOMETRIC_KERNELS_BACKEND='tensorflow'

If not set the default is used, which is 'numpy'.
"""

logging.info(f"Using backend: {BACKEND}.")
