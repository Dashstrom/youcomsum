"""Main module."""

from .cli import entrypoint
from .core import summarize_youtube_comment
from .info import (
    __author__,
    __email__,
    __license__,
    __maintainer__,
    __summary__,
    __version__,
)

__all__ = [
    "__author__",
    "__email__",
    "__license__",
    "__maintainer__",
    "__summary__",
    "__version__",
    "entrypoint",
    "summarize_youtube_comment",
]
