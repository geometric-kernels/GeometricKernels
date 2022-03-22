import lab as B
import lab.autograd    # Load the AutoGrad extension.
import lab.torch       # Load the PyTorch extension.
import lab.tensorflow  # Load the TensorFlow extension.
import lab.jax         # Load the JAX extension.

import numpy as np
import jax.numpy as jnp
import torch
import tensorflow as tf

from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

_TRUNCATION_LEVEL = 10
_NU = 2.5
_LENGTHSCALE = 1.0

if __name__ == '__main__':
    # Parameters
    dimension = 3
    nb_samples = 10

    # Create manifold
    hypersphere = Hypersphere(dim=dimension)

    # Generate samples
    xs = hypersphere.random_point(nb_samples)

    # Kernel
    kernel = MaternKarhunenLoeveKernel(hypersphere, _TRUNCATION_LEVEL)
    params, state = kernel.init_params_and_state()

    # Numpy backend
    params["nu"] = np.r_[_NU]
    params["lengthscale"] = np.r_[_LENGTHSCALE]
    kernel.K(params, state, xs)

    # Jax backend
    params["nu"] = jnp.r_[_NU]
    params["lengthscale"] = jnp.r_[_LENGTHSCALE]
    kernel.K(params, state, jnp.array(xs))

    # Torch backend
    params["nu"] = torch.tensor(_NU)
    params["lengthscale"] = torch.tensor(_LENGTHSCALE)
    kernel.K(params, state, torch.from_numpy(xs))

    # Tensorflow backend
    params["nu"] = tf.convert_to_tensor(_NU)
    params["lengthscale"] = tf.convert_to_tensor(_LENGTHSCALE)
    kernel.K(params, state, tf.convert_to_tensor(xs))

