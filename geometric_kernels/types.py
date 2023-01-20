"""
Types used across the package.
"""

from typing import Any, Callable

FeatureMap = Callable[[Any], Any]  # alas, B.Numeric is not a type
