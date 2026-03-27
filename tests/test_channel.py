"""Tests for Channel async transport."""

import asyncio

import pytest

from signal_gating import Channel, ChannelClosed, ChannelFull, Signal
from signal_gating.channel import PriorityChannel


async def test_send_receive():
    ch: Channel[Signal] = Channel(Signal)
    s = Signal(priority=1)
    await ch.send(s)
    received = await ch.receive()
    assert received.id == s.id


async def test_channel_iteration():
    ch: Channel[Signal] = Channel(Signal)
    signals = [Signal(priority=i) for i in range(3)]
    for s in signals:
        await ch.send(s)
    ch.close()

    received = []
    async for s in ch:
        received.append(s)
    assert len(received) == 3


async def test_bounded_channel():
    ch: Channel[Signal] = Channel(Signal, buffer_size=2)
    await ch.send(Signal())
    await ch.send(Signal())
    with pytest.raises(ChannelFull):
        await ch.send(Signal())


async def test_closed_channel_send():
    ch: Channel[Signal] = Channel(Signal)
    ch.close()
    with pytest.raises(ChannelClosed):
        await ch.send(Signal())


async def test_try_receive_empty():
    ch: Channel[Signal] = Channel(Signal)
    assert ch.try_receive() is None


async def test_try_receive_with_signal():
    ch: Channel[Signal] = Channel(Signal)
    s = Signal()
    await ch.send(s)
    received = ch.try_receive()
    assert received is not None
    assert received.id == s.id


async def test_drain():
    ch: Channel[Signal] = Channel(Signal)
    for i in range(5):
        await ch.send(Signal(priority=i))
    drained = await ch.drain()
    assert len(drained) == 5
    assert ch.pending == 0


async def test_pending_count():
    ch: Channel[Signal] = Channel(Signal)
    assert ch.pending == 0
    await ch.send(Signal())
    assert ch.pending == 1
    await ch.receive()
    assert ch.pending == 0


# --- Backpressure: send_wait ---


async def test_send_wait_when_space_available():
    ch: Channel[Signal] = Channel(Signal, buffer_size=5)
    s = Signal(priority=1)
    await ch.send_wait(s)
    received = await ch.receive()
    assert received.id == s.id


async def test_send_wait_blocks_until_space():
    ch: Channel[Signal] = Channel(Signal, buffer_size=1)
    await ch.send(Signal(priority=1))  # Fill the buffer

    # send_wait should block; use a task to drain after a short delay
    async def drain_later():
        await asyncio.sleep(0.02)
        await ch.receive()

    asyncio.create_task(drain_later())
    await ch.send_wait(Signal(priority=2), timeout=1.0)
    assert ch.pending == 1


async def test_send_wait_on_closed_channel():
    ch: Channel[Signal] = Channel(Signal)
    ch.close()
    with pytest.raises(ChannelClosed):
        await ch.send_wait(Signal())


# --- PriorityChannel ---


async def test_priority_channel_ordering():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=10)
    await ch.send(Signal(priority=1))
    await ch.send(Signal(priority=10))
    await ch.send(Signal(priority=5))

    s1 = await ch.receive()
    s2 = await ch.receive()
    s3 = await ch.receive()

    assert s1.priority == 10
    assert s2.priority == 5
    assert s3.priority == 1


async def test_priority_channel_bounded():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal, buffer_size=2)
    await ch.send(Signal())
    await ch.send(Signal())
    with pytest.raises(ChannelFull):
        await ch.send(Signal())


async def test_priority_channel_closed():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal)
    ch.close()
    with pytest.raises(ChannelClosed):
        await ch.send(Signal())


async def test_priority_channel_iteration():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal)
    await ch.send(Signal(priority=1))
    await ch.send(Signal(priority=3))
    await ch.send(Signal(priority=2))
    ch.close()

    received = []
    async for s in ch:
        received.append(s.priority)

    assert received == [3, 2, 1]


async def test_priority_channel_drain():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal)
    await ch.send(Signal(priority=1))
    await ch.send(Signal(priority=5))
    await ch.send(Signal(priority=3))

    drained = await ch.drain()
    assert len(drained) == 3
    assert drained[0].priority == 5
    assert drained[1].priority == 3
    assert drained[2].priority == 1
    assert ch.pending == 0


async def test_priority_channel_try_receive():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal)
    assert ch.try_receive() is None
    await ch.send(Signal(priority=7))
    s = ch.try_receive()
    assert s is not None
    assert s.priority == 7


async def test_priority_channel_receive_blocks():
    ch: PriorityChannel[Signal] = PriorityChannel(Signal)

    async def send_later():
        await asyncio.sleep(0.02)
        await ch.send(Signal(priority=42))

    asyncio.create_task(send_later())
    s = await ch.receive()
    assert s.priority == 42
