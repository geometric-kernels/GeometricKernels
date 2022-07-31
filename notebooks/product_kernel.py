# %%
%load_ext autoreload
%autoreload 2
# %%
import lab.numpy as B
import matplotlib.pyplot as plt
from geometric_kernels.lab_extras.extras import from_numpy
import numpy as np
# %%

from geometric_kernels.spaces import Circle, ProductDiscreteSpectrumSpace

circle = Circle()

# %%
product = ProductDiscreteSpectrumSpace(circle, circle, num_eigen=11**2)
eigenfunctons = product.get_eigenfunctions(11**2)
# %%
grid = B.linspace(0, 2*B.pi, 20)
ones = B.ones(20)
grid = B.stack(
    grid[:, None] * ones[None, :],
    grid[None, :] * ones[:, None],
    axis=-1
)

grid_ = B.reshape(grid, 20**2, 2)

# %%
ef_vls = eigenfunctons(grid_)
ef_vls = ef_vls.reshape(20,20,-1)

# %%
from geometric_kernels.kernels import MaternKarhunenLoeveKernel

kernel = MaternKarhunenLoeveKernel(product, 11**2)

params, state = kernel.init_params_and_state()
params['nu'] = from_numpy(grid_, np.inf)

# %%
k_xx = kernel.K(params, state, grid_, grid_)
k_xx = k_xx.reshape(20,20,20,20)

plt.imshow(k_xx[10,10])
# %%
kernel_single = MaternKarhunenLoeveKernel(circle, 11)


params_single, state_single = kernel_single.init_params_and_state()
params_single['nu'] = from_numpy(grid_, np.inf)

k_xx_single_1 = kernel_single.K(params_single, state_single, grid_[..., :1], grid_[..., :1]).reshape(20,20,20,20)
plt.imshow(k_xx_single_1[10,10])

# %%
k_xx_single_2 = kernel_single.K(params_single, state_single, grid_[..., 1:], grid_[..., 1:]).reshape(20,20,20,20)
plt.imshow(k_xx_single_2[10,10])
# %%
k_xx_product = k_xx_single_1 * k_xx_single_2
plt.imshow(k_xx_product[10,10])

# %%
k_xx - k_xx_product
# %%
