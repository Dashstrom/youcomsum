"""Core module."""

from typing import Optional


def hello(text: Optional[str]) -> str:
    """Add hello text before the provided text.

    Args:
        text: Text to add after hello.

    Returns:
        A string with Hello + text.

    Examples:
        >>> hello("world")
        'Hello world'
        >>> hello(author).startswith('Hello')
        True
        >>> hello("")
        'Hello'
        >>> hello()
        Traceback (most recent call last):
        TypeError: hello() missing 1 required positional argument: 'text'
    """
    return f"Hello {text or ''}".strip()
