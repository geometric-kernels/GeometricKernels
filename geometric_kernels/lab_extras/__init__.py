"""
Custom extensions for `LAB <https://github.com/wesselb/lab>`_.
"""

from lab import dispatch

from geometric_kernels.lab_extras.extras import *

# Always load the numpy backend because we assume numpy is always installed.
from geometric_kernels.lab_extras.numpy import *
