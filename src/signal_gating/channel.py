"""Channels: async typed conduits for signal transport."""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from signal_gating.errors import ChannelClosed, ChannelFull, SignalValidationError
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

    Close semantics: pending items are drained, then receivers raise
    ChannelClosed. Receivers blocked on a full closed channel wake up
    immediately; close never silently strands a waiter.
    """

    def __init__(self, signal_type: type[T] | None = None, buffer_size: int = 0):
        self.signal_type = signal_type
        self._queue: asyncio.Queue[T] = asyncio.Queue(
            maxsize=buffer_size if buffer_size > 0 else 0
        )
        self._closed = False
        self._close_event = asyncio.Event()

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
        self._validate_type(signal)
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            raise ChannelFull() from None

    async def send_wait(self, signal: T, timeout: float | None = None) -> None:
        """Send a signal, waiting for space if the channel is full.

        Unlike `send()`, this method applies backpressure by blocking until
        buffer space is available instead of raising ChannelFull.
        """
        if self._closed:
            raise ChannelClosed()
        self._validate_type(signal)
        if timeout is not None:
            await asyncio.wait_for(self._queue.put(signal), timeout=timeout)
        else:
            await self._queue.put(signal)

    async def receive(self) -> T:
        """Receive the next signal. Blocks until one is available or the channel closes."""
        # Fast path: item already available.
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        if self._closed:
            raise ChannelClosed()

        # Slow path: race a get against close. Either the producer hands us
        # an item, or close() fires and we raise. No sentinel; no silent strands.
        get_task: asyncio.Task[T] = asyncio.ensure_future(self._queue.get())
        close_task = asyncio.ensure_future(self._close_event.wait())
        try:
            done, _ = await asyncio.wait(
                {get_task, close_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                return get_task.result()
            # Close fired. Drain anything that landed concurrently.
            if get_task.done():
                return get_task.result()
            try:
                return self._queue.get_nowait()
            except asyncio.QueueEmpty:
                raise ChannelClosed() from None
        finally:
            for t in (get_task, close_task):
                if not t.done():
                    t.cancel()

    def try_receive(self) -> T | None:
        """Non-blocking receive. Returns None if no signal is available.

        To distinguish "empty" from "closed and drained", check `closed`.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        """Close the channel. Pending signals can still be drained.

        Wakes all blocked receivers. Future sends raise ChannelClosed.
        """
        if self._closed:
            return
        self._closed = True
        self._close_event.set()

    def __aiter__(self) -> Channel[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive()
        except ChannelClosed:
            raise StopAsyncIteration from None

    async def drain(self) -> list[T]:
        """Drain all pending signals from the channel."""
        signals: list[T] = []
        while True:
            try:
                signals.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return signals

    def _validate_type(self, signal: T) -> None:
        if self.signal_type is not None and not isinstance(signal, self.signal_type):
            raise SignalValidationError(
                f"Channel expected {self.signal_type.__name__}, got {type(signal).__name__}"
            )

    @staticmethod
    async def merge(*channels: Channel[T]) -> AsyncIterator[T]:
        """Merge multiple channels into a single async stream.

        Yields signals from any channel as they arrive. When all channels
        are closed, the iterator ends. Useful when an agent consumes from
        multiple sources without dedicating a task to each.

            async for signal in Channel.merge(inbox_a, inbox_b, inbox_c):
                process(signal)
        """
        output: asyncio.Queue[T | None] = asyncio.Queue()
        remaining = len(channels)
        lock = asyncio.Lock()

        async def _reader(ch: Channel[T]) -> None:
            nonlocal remaining
            try:
                async for signal in ch:
                    await output.put(signal)
            except (ChannelClosed, StopAsyncIteration):
                pass
            finally:
                async with lock:
                    remaining -= 1
                    if remaining <= 0:
                        await output.put(None)

        tasks = [asyncio.create_task(_reader(ch)) for ch in channels]
        try:
            while True:
                item = await output.get()
                if item is None:
                    break
                yield item
        finally:
            for t in tasks:
                t.cancel()


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
        self._has_space = asyncio.Event()
        self._has_space.set()  # Initially has space
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
        self._validate_type(signal)
        async with self._lock:
            if self._max_size and len(self._heap) >= self._max_size:
                raise ChannelFull()
            # Negate priority for max-heap behavior (heapq is a min-heap)
            heapq.heappush(self._heap, (-signal.priority, self._counter, signal))
            self._counter += 1
            self._has_items.set()
            if self._max_size and len(self._heap) >= self._max_size:
                self._has_space.clear()

    async def receive(self) -> T:
        """Receive the highest-priority signal. Blocks until one is available."""
        while True:
            async with self._lock:
                if self._heap:
                    _, _, signal = heapq.heappop(self._heap)
                    if not self._heap:
                        self._has_items.clear()
                    self._has_space.set()
                    return signal
                if self._closed:
                    raise ChannelClosed()
                self._has_items.clear()
            await self._has_items.wait()

    async def send_wait(self, signal: T, timeout: float | None = None) -> None:
        """Send a signal, waiting for space if the channel is full.

        Unlike `send()`, this method applies backpressure by blocking until
        buffer space is available instead of raising ChannelFull.
        Uses event-driven notification (no polling).
        """
        if self._closed:
            raise ChannelClosed()
        self._validate_type(signal)

        async def _wait_and_send() -> None:
            while True:
                await self._has_space.wait()
                async with self._lock:
                    if not self._max_size or len(self._heap) < self._max_size:
                        heapq.heappush(
                            self._heap, (-signal.priority, self._counter, signal)
                        )
                        self._counter += 1
                        self._has_items.set()
                        if self._max_size and len(self._heap) >= self._max_size:
                            self._has_space.clear()
                        return

        if timeout is not None:
            await asyncio.wait_for(_wait_and_send(), timeout=timeout)
        else:
            await _wait_and_send()

    def try_receive(self) -> T | None:
        """Non-blocking receive of the highest-priority signal.

        Note: not safe to interleave with concurrent receive(); locking would
        require an async API. Use receive() in concurrent code paths.
        """
        if not self._heap:
            return None
        _, _, signal = heapq.heappop(self._heap)
        if not self._heap:
            self._has_items.clear()
        self._has_space.set()
        return signal

    def close(self) -> None:
        """Close the channel. Pending signals can still be drained."""
        if self._closed:
            return
        self._closed = True
        self._has_items.set()

    def __aiter__(self) -> PriorityChannel[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive()
        except ChannelClosed:
            raise StopAsyncIteration from None

    async def drain(self) -> list[T]:
        """Drain all pending signals in priority order (highest first)."""
        async with self._lock:
            signals: list[T] = []
            while self._heap:
                _, _, signal = heapq.heappop(self._heap)
                signals.append(signal)
            self._has_items.clear()
            self._has_space.set()
            return signals

    def _validate_type(self, signal: T) -> None:
        if self.signal_type is not None and not isinstance(signal, self.signal_type):
            raise SignalValidationError(
                f"PriorityChannel expected {self.signal_type.__name__}, "
                f"got {type(signal).__name__}"
            )
