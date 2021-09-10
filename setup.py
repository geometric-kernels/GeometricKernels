#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

requirements = [
    "numpy==1.20.3",
    "scipy",
]

with open("README.md", "r") as file:
    long_description = file.read()

with open("VERSION", "r") as file:
    version = file.read().strip()

setup(
    name="GeometricKernels",
    version=version,
    author="Vincent Dutordoir",
    author_email="vd309@cam.ac.uk",
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