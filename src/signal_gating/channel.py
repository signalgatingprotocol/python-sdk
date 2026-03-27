"""Channels — async typed conduits for signal transport."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from signal_gating.errors import ChannelClosed, ChannelFull
from signal_gating.signal import Signal

T = TypeVar("T", bound=Signal)


class Channel(Generic[T]):
    """An async channel for transporting signals between agents.

    Supports bounded buffering and async iteration:

        channel = Channel(TaskSignal, buffer_size=100)

        # Producer
        await channel.send(TaskSignal(task="work"))

        # Consumer
        async for signal in channel:
            process(signal)
    """

    def __init__(self, signal_type: type[T] | None = None, buffer_size: int = 0):
        self.signal_type = signal_type
        self._queue: asyncio.Queue[T | None] = asyncio.Queue(
            maxsize=buffer_size if buffer_size > 0 else 0
        )
        self._closed = False
        self._receivers: int = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def send(self, signal: T) -> None:
        """Send a signal into the channel."""
        if self._closed:
            raise ChannelClosed()
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            raise ChannelFull()

    async def receive(self) -> T:
        """Receive the next signal. Blocks until one is available."""
        if self._closed and self._queue.empty():
            raise ChannelClosed()
        item = await self._queue.get()
        if item is None:
            raise ChannelClosed()
        return item

    def try_receive(self) -> T | None:
        """Non-blocking receive. Returns None if no signal is available."""
        try:
            item = self._queue.get_nowait()
            if item is None:
                return None
            return item
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        """Close the channel. Pending signals can still be drained."""
        self._closed = True
        # Wake up any waiting receivers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def __aiter__(self) -> Channel[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive()
        except ChannelClosed:
            raise StopAsyncIteration

    async def drain(self) -> list[T]:
        """Drain all pending signals from the channel."""
        signals: list[T] = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not None:
                signals.append(item)
        return signals
