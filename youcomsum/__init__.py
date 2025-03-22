"""Main module."""

from .cli import entrypoint
from .core import YouComSum
from .info import (
    __author__,
    __email__,
    __license__,
    __maintainer__,
    __summary__,
    __version__,
)

__all__ = [
    "YouComSum",
    "__author__",
    "__email__",
    "__license__",
    "__maintainer__",
    "__summary__",
    "__version__",
    "entrypoint",
]
