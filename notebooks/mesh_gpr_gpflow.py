# %% [markdown]
# # Gaussian process on mesh using GPflow
#
# This notebooks shows how to fit a GPflow Gaussian process (GP) on a mesh object. The mesh is represented through a finite set of vertices and edges.

# %%
import gpflow
import numpy as np
import meshzoo
import plotly.graph_objects as go

# %%
from geometric_kernels.frontends.tensorflow.gpflow import (
    DefaultFloatZeroMeanFunction,
    GPflowGeometricKernel,
)
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.kernels import MaternKarhunenLoeveKernel

# %% [markdown]
# ## Load and plot mesh
#
# In this example we will use a simple Teddy shaped mesh.

# %%
from pathlib import Path


def update_figure(fig):
    """Utility to clean up figure"""
    fig.update_layout(scene_aspectmode="cube")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.update_traces(showscale=False, hoverinfo="none")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )
    return fig


def plot_mesh(mesh: Mesh, vertices_colors=None):
    plot = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        colorscale="Viridis",
        intensity=vertices_colors,
    )
    return plot


resolution = 30
vertices, faces = meshzoo.icosa_sphere(resolution)
mesh = Mesh(vertices, faces)

# mesh = Mesh.load_mesh(str(Path.cwd() / "data" / "teddy.obj"))
print("Number of vertices in the mesh:", mesh.num_vertices)
plot = plot_mesh(mesh)
fig = go.Figure(plot)
update_figure(fig)

# %% [markdown]
# ## Create dummy dataset on mesh
#
# We sample from the prior of the GP to create a simple dataset we can afterwards fit using a exact Gaussian process regression (GPR) model. The input vector $X \in \mathbb{N}^{n \times 1}$ consists of **indices** indexing the different vertices of the mesh. Consequently, the elements of $X$ are in $[0, N_v-1]$, where $N_v$ are the number of vertices in the mesh. In this example $N_v = 1598$.
#
# To sample from the prior we create a `MaternKarhunenLoeveKernel` object and pass this to a GPflow wrapper `GPflowGeometricKernel`. `MaternKarhunenLoeveKernel` contains all of the logic to decompose the space into its Laplace eigensystem in order to create a valid kernel.

# %%
nu = 1 / 2.0
truncation_level = 20
base_kernel = MaternKarhunenLoeveKernel(mesh, nu, truncation_level)
kernel = GPflowGeometricKernel(base_kernel)
num_data = 10  # n


def draw_random_data_from_prior():
    # np.random.seed(1)
    _X = np.random.randint(mesh.num_vertices, size=(num_data, 1))
    _K = kernel.K(_X).numpy()
    _y = np.linalg.cholesky(_K + np.eye(num_data) * 1e-6) @ np.random.randn(num_data, 1)
    return _X, _y


X, y = draw_random_data_from_prior()
print("Inputs", X[:3])
print("Outputs", y[:3])

# %% [markdown]
# ## Build GPflow model

# %%
model = gpflow.models.GPR(
    (X, y), kernel, mean_function=DefaultFloatZeroMeanFunction(), noise_variance=1.1e-6
)
print("LML", model.log_marginal_likelihood().numpy())

# %% [markdown]
# ## Evaluate
#
# Given the dataset, which dictates how the function behaves at $n$ locations on the mesh we want to predict the values at other locations. Therefore we build a test vector $X_{test}$ containing all the indices, i.e. $[0, N_v-1]$.

# %%
X_test = np.arange(mesh.num_vertices).reshape(-1, 1)
print(X_test.shape)
print(X_test[:3])

# predict mean and variance
mean_prediction, variance_prediction = model.predict_f(X_test)
mean_prediction = mean_prediction.numpy()

# predict sample
sample = model.predict_f_samples(X_test).numpy()

# %%
prediction_plot = plot_mesh(mesh, vertices_colors=mean_prediction)
data_plot = go.Scatter3d(
    x=mesh.vertices[X.ravel()][:, 0],
    y=mesh.vertices[X.ravel()][:, 1],
    z=mesh.vertices[X.ravel()][:, 2],
    marker=dict(
        size=12,
        color=y.ravel(),  # set color to an array/list of desired values
        colorscale="Viridis",  # choose a colorscale
        opacity=0.8,
        cmin=mean_prediction.min(),
        cmax=mean_prediction.max(),
    ),
)
fig = go.Figure(data=[prediction_plot, data_plot])
fig = update_figure(fig)
fig
