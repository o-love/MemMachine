"""Common utility functions."""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

async def async_with(
    async_context_manager: AbstractAsyncContextManager[object],
    awaitable: Awaitable[T],
) -> T:
    """
    Helper function to use an async context manager with an awaitable.

    Args:
        async_context_manager (AbstractAsyncContextManager):
            The async context manager to use.
        awaitable (Awaitable):
            The awaitable to execute within the context.

    Returns:
        Any:
            The result of the awaitable.

    """
    async with async_context_manager:
        return await awaitable


def async_locked(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to ensure that a coroutine function is executed with a lock.
    The lock is shared across all invocations of the decorated coroutine function.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with lock:
            return await func(*args, **kwargs)

    return wrapper
