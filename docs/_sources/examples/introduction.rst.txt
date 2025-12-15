Here you can find numerous example notebooks providing tutorials into GeometricKernels.

Each of the *Spaces* notebooks explains the basic functionality of the library in relation to a specific space.

If you are interested in using GeometricKernels with a backend other than NumPy, checkout the *Backends* notebooks.

If you want to use GeometricKernels with one of the popular Gaussian process libraries, checkout the *Frontends* notebooks.

If you want to learn how to implement your own space or kernel component, checkout `CustomSpacesAndKernels.ipynb <CustomSpacesAndKernels.html>`_ notebook.

Finally, if you are interested in application examples:

* `PeMS Regression <https://github.com/vabor112/pems-regression>`__  
    A benchmark suite for graph node regression with uncertainty. This project employs GeometricKernels among other tools, and offers processed data, baseline models, and an `example notebook <https://github.com/vabor112/pems-regression/tree/main/notebooks/GeometricProbabilisticModels.ipynb>`__ for experiments on graph-structured data.  
    Notably, in this benchmark, geometric Gaussian processes built with GeometricKernels have been shown to outperform various alternative methods, including ensembles of graph neural networks and Bayesian graph neural networks.

* `Bayesian optimization demonstration <https://github.com/geometric-kernels/GeometricKernels/blob/main/notebooks/other/Bayesian%20Optimization.ipynb>`__  
    A minimal notebook illustrating the use of GeometricKernels with the `botorch <https://botorch.org/>`__ library for Bayesian optimization.  
    This is a simple, self-contained example designed to demonstrate core concepts rather than to reflect a real-world scenario.