"""
Wrappers to use the geometric kernels in GPflow and other downstream packages.
"""
# noqa: F401
import logging

from geometric_kernels import BACKEND

if BACKEND == "tensorflow":
    from geometric_kernels.frontends.gpflow import GPflowGeometricKernel

    logging.info(f"Importing GPflow model")
elif BACKEND == "pytorch":
    from geometric_kernels.frontends.gpytorch import GPytorchGeometricKernel

    logging.info(f"Importing GPyTorch model")
