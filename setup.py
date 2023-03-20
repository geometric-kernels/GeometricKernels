#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "plum-dispatch==1.7.4",
    "backends==1.4.32",
    "potpourri3d",
    "robust_laplacian",
    "meshzoo",
    "opt-einsum",
    "geomstats",
    "einops",
    "spherical-harmonics @ git+https://github.com/vdutor/SphericalHarmonics.git",
]

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("VERSION", "r") as file:
    version = file.read().strip()

setup(
    name="GeometricKernels",
    version=version,
    author="The GeometricKernels contributors",
    author_email="geometric-kernels@googlegroups.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="A Python Package for Geometric Kernels in TensorFlow, PyTorch and Jax",
    license="Apache License 2.0",
    keywords="Geometric-kernels",
    install_requires=requirements,
    packages=find_namespace_packages(include=["geometric_kernels*"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
