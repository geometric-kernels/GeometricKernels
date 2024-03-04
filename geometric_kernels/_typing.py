"""
Types used across the package.

This private module provides a unified interface to the otherwise different
treatment of typing hints by Python<3.9 and Python>=3.9.
Read more at https://beartype.readthedocs.io/en/latest/api_roar/#pep-585-deprecations
"""
from sys import version_info

if version_info >= (3, 9):
    Type = type
    List = list
    Tuple = tuple
    Dict = dict
    from collections.abc import Callable, Mapping  # noqa: F401
else:
    from typing import Callable, Dict, List, Mapping, Tuple, Type  # noqa: F401

# These are the types you need to import from `typing` even after PEP 585.
from typing import Any, Generic, Optional, TypeVar, Union  # noqa: F401

FeatureMap = Callable[[Any], Any]  # alas, B.Numeric is not a type
