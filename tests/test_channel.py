"""Tests for Channel async transport."""

import asyncio

import pytest

from signal_gating import Channel, ChannelClosed, ChannelFull, Signal


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
