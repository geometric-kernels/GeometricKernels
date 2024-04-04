"""
This module provides the abstract base class :class:`FeatureMap` that all
feature maps inherit from. It can be used for type hinting.
"""

import abc


class FeatureMap(abc.ABC):
    """
    Abstract base class for all feature maps.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        r"""
        `FeatureMap`\ s are callable.
        """
        raise NotImplementedError
