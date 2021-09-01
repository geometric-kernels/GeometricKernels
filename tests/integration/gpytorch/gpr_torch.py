import gpytorch
import meshzoo
import numpy as np
import polyscope as ps
import torch

from geometric_kernels.frontends import GPytorchGeometricKernel
from geometric_kernels.kernels import MeshKernel
from geometric_kernels.spaces import Mesh


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':
    resolution = 10
    vertices, faces = meshzoo.icosa_sphere(resolution)
    mesh = Mesh(vertices, faces)

    nu = 1 / 2.0
    truncation_level = 20
    base_kernel = MeshKernel(mesh, nu, truncation_level)
    kernel = GPytorchGeometricKernel(base_kernel)
    kernel.double()
    num_data = 25


    def get_data():
        _X = torch.tensor(np.random.randint(mesh.num_vertices, size=(num_data,)))
        _K = kernel(_X).numpy()
        _y = torch.tensor(
            np.linalg.cholesky(_K + np.eye(num_data) * 1e-6) @ np.random.randn(num_data)
        ).float()
        return _X, _y


    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-7)
    )
    likelihood.noise = torch.tensor(1e-6)

    X, y = get_data()
    model = ExactGPModel(X, y, likelihood, kernel)
    model.double()
    likelihood.double()
    model.eval()

    X_test = torch.tensor(np.arange(mesh.num_vertices))
    f_preds = model(X_test)
    m, v = f_preds.mean, f_preds.variance
    m, v = m.detach().numpy(), v.detach().numpy()
    sample = f_preds.sample(sample_shape=torch.Size([1])).detach().numpy()

    X_numpy = X.numpy().astype(np.int32)

    ps.init()
    ps_cloud = ps.register_point_cloud("my points", vertices[X_numpy.flatten()])
    ps_cloud.add_scalar_quantity("data", y.numpy().flatten())


    my_mesh = ps.register_surface_mesh("my mesh", vertices, faces, smooth_shade=True)
    my_mesh.add_scalar_quantity(f"sample", sample.squeeze(), enabled=True)
    my_mesh.add_scalar_quantity(f"mean", m.squeeze(), enabled=True)
    my_mesh.add_scalar_quantity(f"variance", v.squeeze(), enabled=True)
    ps.show()
