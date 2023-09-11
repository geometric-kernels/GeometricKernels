# %%
%load_ext autoreload
%autoreload 2
# %%
import lab as B
import numpy as np
from geometric_kernels.spaces import ProductDiscreteSpectrumSpace
from geometric_kernels.spaces import Circle
# %%

circ = Circle()
prod_space = ProductDiscreteSpectrumSpace(Circle(), Circle(), num_eigen=101)
# %%
x = B.linspace(0, 2*3.14159, 101)
xx = np.meshgrid(x,x)
X = B.stack(*xx, axis=-1).reshape((-1, 2))
eigs = prod_space.get_eigenfunctions(101)
eigfuncs = eigs(X).reshape((101,101,-1))
# %%
import matplotlib.pyplot as plt
# %%
fig, axes = plt.subplots(10,10, sharex=True, sharey=True, figsize=(10,10))

for i, ax in enumerate([item for l in axes for item in l]):
    ax.imshow(eigfuncs[..., i])
    plt.axis('off')

plt.tight_layout()
# %%
from geometric_kernels.kernels import MaternKarhunenLoeveKernel


kernel = MaternKarhunenLoeveKernel(prod_space, 101)
params = kernel.init_params()

k_xx = kernel.K(params, state, X, X).reshape((101,101,101,101))


fig, axes = plt.subplots(2,5, sharex=True, sharey=True, figsize=(10,4))

for i, ax in zip([0,10,20,30,40,50,60,70,80,90], [item for l in axes for item in l]):
    ax.imshow(k_xx[i,i])
    plt.axis('off')

plt.tight_layout()
# %%
