"""Main module."""

from .cli import entrypoint
from .core import hello
from .info import (
    __author__,
    __email__,
    __license__,
    __maintainer__,
    __summary__,
    __version__,
)

__all__ = [
    "entrypoint",
    "hello",
    "__author__",
    "__email__",
    "__license__",
    "__maintainer__",
    "__summary__",
    "__version__",
]
