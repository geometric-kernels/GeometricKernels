"""
Types used across the package.

This private module provides a unified interface to the otherwise different
treatment of typing hints by Python<3.9 and Python>=3.9.
Read more at https://beartype.readthedocs.io/en/latest/api_roar/#pep-585-deprecations

We follow https://peps.python.org/pep-0585/.
"""
import collections
import contextlib
import re
from sys import version_info

if version_info >= (3, 9):
    Tuple = tuple
    List = list
    Dict = dict
    Set = set
    FrozenSet = frozenset
    Type = type

    Deque = collections.deque
    DefaultDict = collections.defaultdict
    OrderedDict = collections.OrderedDict
    Counter = collections.Counter
    ChainMap = collections.ChainMap

    AbstractSet = collections.abc.Set

    ContextManager = contextlib.AbstractContextManager
    AsyncContextManager = contextlib.AbstractAsyncContextManager

    Pattern = re.Pattern
    Match = re.Match

    from collections.abc import (  # noqa: F401 isort:skip
        Awaitable,
        Coroutine,
        AsyncIterable,
        AsyncIterator,
        AsyncGenerator,
        Iterable,
        Iterator,
        Generator,
        Reversible,
        Container,
        Collection,
        Callable,
        # Set,
        MutableSet,
        Mapping,
        MutableMapping,
        Sequence,
        MutableSequence,
        ByteString,
        MappingView,
        KeysView,
        ItemsView,
        ValuesView,
    )
else:
    from typing import Tuple, List, Dict, Set, FrozenSet, Type  # noqa: F401 isort:skip
    from typing import (  # noqa: F401 isort:skip
        Deque,
        DefaultDict,
        OrderedDict,
        Counter,
        ChainMap,
    )
    from typing import AbstractSet  # noqa: F401 isort:skip
    from typing import ContextManager, AsyncContextManager  # noqa: F401 isort:skip
    from typing import Pattern, Match  # noqa: F401 isort:skip
    from typing import (  # noqa: F401 isort:skip
        Awaitable,
        Coroutine,
        AsyncIterable,
        AsyncIterator,
        AsyncGenerator,
        Iterable,
        Iterator,
        Generator,
        Reversible,
        Container,
        Collection,
        Callable,
        # Set,
        MutableSet,
        Mapping,
        MutableMapping,
        Sequence,
        MutableSequence,
        ByteString,
        MappingView,
        KeysView,
        ItemsView,
        ValuesView,
    )

# These are the types you need to import from `typing` even after PEP 585.
from typing import Any, Generic, Optional, TypeVar, Union  # noqa: F401

FeatureMap = Callable[[Any], Any]  # alas, B.Numeric is not a type
