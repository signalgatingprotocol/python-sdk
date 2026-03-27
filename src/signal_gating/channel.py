"""Channels — async typed conduits for signal transport."""

from __future__ import annotations

import asyncio
import heapq
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

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def send(self, signal: T) -> None:
        """Send a signal into the channel. Raises ChannelFull if buffer is full."""
        if self._closed:
            raise ChannelClosed()
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            raise ChannelFull()

    async def send_wait(self, signal: T, timeout: float | None = None) -> None:
        """Send a signal, waiting for space if the channel is full.

        Unlike `send()`, this method applies backpressure by blocking until
        buffer space is available instead of raising ChannelFull.
        """
        if self._closed:
            raise ChannelClosed()
        if timeout is not None:
            await asyncio.wait_for(self._queue.put(signal), timeout=timeout)
        else:
            await self._queue.put(signal)

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


class PriorityChannel(Generic[T]):
    """A channel that dequeues signals by priority (highest first).

    Drop-in replacement for Channel when signal priority ordering matters.
    In agent systems, high-priority signals should not wait behind low-priority ones.

        channel = PriorityChannel(Signal, buffer_size=1000)

        await channel.send(Signal(priority=1))
        await channel.send(Signal(priority=10))

        first = await channel.receive()
        assert first.priority == 10  # highest priority dequeued first
    """

    def __init__(self, signal_type: type[T] | None = None, buffer_size: int = 0):
        self.signal_type = signal_type
        self._heap: list[tuple[int, int, T]] = []
        self._counter = 0
        self._max_size = buffer_size
        self._closed = False
        self._has_items = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return len(self._heap)

    async def send(self, signal: T) -> None:
        """Send a signal into the priority channel."""
        if self._closed:
            raise ChannelClosed()
        async with self._lock:
            if self._max_size and len(self._heap) >= self._max_size:
                raise ChannelFull()
            # Negate priority for max-heap behavior (heapq is a min-heap)
            heapq.heappush(self._heap, (-signal.priority, self._counter, signal))
            self._counter += 1
            self._has_items.set()

    async def receive(self) -> T:
        """Receive the highest-priority signal. Blocks until one is available."""
        while True:
            async with self._lock:
                if self._heap:
                    _, _, signal = heapq.heappop(self._heap)
                    if not self._heap:
                        self._has_items.clear()
                    return signal
                if self._closed:
                    raise ChannelClosed()
                self._has_items.clear()
            await self._has_items.wait()

    def try_receive(self) -> T | None:
        """Non-blocking receive of the highest-priority signal."""
        if self._heap:
            _, _, signal = heapq.heappop(self._heap)
            if not self._heap:
                self._has_items.clear()
            return signal
        return None

    def close(self) -> None:
        """Close the channel. Pending signals can still be drained."""
        self._closed = True
        self._has_items.set()

    def __aiter__(self) -> PriorityChannel[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive()
        except ChannelClosed:
            raise StopAsyncIteration

    async def drain(self) -> list[T]:
        """Drain all pending signals in priority order (highest first)."""
        async with self._lock:
            signals: list[T] = []
            while self._heap:
                _, _, signal = heapq.heappop(self._heap)
                signals.append(signal)
            self._has_items.clear()
            return signals
