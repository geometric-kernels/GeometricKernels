"""
Types used across the package.
"""

from typing import Callable

import lab as B

FeatureMap = Callable[[B.Numeric], B.Numeric]
