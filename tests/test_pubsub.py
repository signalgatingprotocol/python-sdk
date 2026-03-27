"""Tests for Mesh Pub/Sub topic-based routing."""

import asyncio

import pytest

from signal_gating import Agent, Mesh, MeshError, Signal


class EventSignal(Signal):
    event: str


class AlertSignal(Signal):
    message: str


# === Topic Management ===


class TestTopicManagement:
    def test_create_topic(self):
        mesh = Mesh()
        mesh.create_topic("events")
        topics = mesh.list_topics()
        assert "events" in topics
        assert topics["events"] == []

    def test_create_duplicate_topic_raises(self):
        mesh = Mesh()
        mesh.create_topic("events")
        with pytest.raises(MeshError, match="already exists"):
            mesh.create_topic("events")

    def test_delete_topic(self):
        mesh = Mesh()
        mesh.create_topic("events")
        mesh.delete_topic("events")
        assert "events" not in mesh.list_topics()

    def test_delete_nonexistent_topic_raises(self):
        mesh = Mesh()
        with pytest.raises(MeshError, match="does not exist"):
            mesh.delete_topic("ghost")

    def test_list_topics_empty(self):
        mesh = Mesh()
        assert mesh.list_topics() == {}


# === Subscribe/Unsubscribe ===


class TestSubscription:
    def test_subscribe_agent(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe(agent, "events")
        assert mesh.list_topics()["events"] == ["logger"]

    def test_subscribe_by_name(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe("logger", "events")
        assert mesh.list_topics()["events"] == ["logger"]

    def test_subscribe_multiple_agents(self):
        a = Agent("a")
        b = Agent("b")
        mesh = Mesh([a, b])
        mesh.create_topic("events")
        mesh.subscribe(a, "events")
        mesh.subscribe(b, "events")
        assert mesh.list_topics()["events"] == ["a", "b"]

    def test_subscribe_to_nonexistent_topic_raises(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        with pytest.raises(MeshError, match="does not exist"):
            mesh.subscribe(agent, "nonexistent")

    def test_subscribe_idempotent(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe(agent, "events")
        mesh.subscribe(agent, "events")
        assert mesh.list_topics()["events"] == ["logger"]

    def test_unsubscribe(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe(agent, "events")
        mesh.unsubscribe(agent, "events")
        assert mesh.list_topics()["events"] == []

    def test_unsubscribe_nonexistent_topic_no_error(self):
        agent = Agent("logger")
        mesh = Mesh([agent])
        # Should not raise
        mesh.unsubscribe(agent, "nonexistent")


# === Publish ===


class TestPublish:
    async def test_publish_to_single_subscriber(self):
        agent = Agent("logger")
        received: list[str] = []

        @agent.on(EventSignal)
        async def handle(signal: EventSignal):
            received.append(signal.event)

        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe(agent, "events")

        async with mesh:
            count = await mesh.publish("events", EventSignal(event="click"))
            await asyncio.sleep(0.05)

        assert count == 1
        assert received == ["click"]

    async def test_publish_to_multiple_subscribers(self):
        a = Agent("a")
        b = Agent("b")
        a_received: list[str] = []
        b_received: list[str] = []

        @a.on(EventSignal)
        async def handle_a(signal: EventSignal):
            a_received.append(signal.event)

        @b.on(EventSignal)
        async def handle_b(signal: EventSignal):
            b_received.append(signal.event)

        mesh = Mesh([a, b])
        mesh.create_topic("events")
        mesh.subscribe(a, "events")
        mesh.subscribe(b, "events")

        async with mesh:
            count = await mesh.publish("events", EventSignal(event="broadcast"))
            await asyncio.sleep(0.05)

        assert count == 2
        assert a_received == ["broadcast"]
        assert b_received == ["broadcast"]

    async def test_publish_to_no_subscribers(self):
        mesh = Mesh()
        mesh.create_topic("events")

        # Should not raise, returns 0
        count = await mesh.publish("events", Signal())
        assert count == 0

    async def test_publish_to_nonexistent_topic_raises(self):
        mesh = Mesh()
        with pytest.raises(MeshError, match="does not exist"):
            await mesh.publish("ghost", Signal())

    async def test_publish_multiple_topics(self):
        agent = Agent("monitor")
        events: list[str] = []
        alerts: list[str] = []

        @agent.on(EventSignal)
        async def handle_event(signal: EventSignal):
            events.append(signal.event)

        @agent.on(AlertSignal)
        async def handle_alert(signal: AlertSignal):
            alerts.append(signal.message)

        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.create_topic("alerts")
        mesh.subscribe(agent, "events")
        mesh.subscribe(agent, "alerts")

        async with mesh:
            await mesh.publish("events", EventSignal(event="page_view"))
            await mesh.publish("alerts", AlertSignal(message="CPU high"))
            await asyncio.sleep(0.05)

        assert events == ["page_view"]
        assert alerts == ["CPU high"]

    async def test_publish_traces_signal_flow(self):
        agent = Agent("sub")

        @agent.on(Signal)
        async def handle(s: Signal):
            pass

        mesh = Mesh([agent])
        mesh.create_topic("events")
        mesh.subscribe(agent, "events")

        async with mesh:
            await mesh.publish("events", Signal())
            await asyncio.sleep(0.05)

        actions = mesh.tracer.summary().get("actions", {})
        assert "published" in actions

    async def test_pubsub_with_point_to_point(self):
        """Pub/sub works alongside normal mesh connections."""
        producer = Agent("producer")
        consumer = Agent("consumer")
        subscriber = Agent("subscriber")
        consumer_received: list[str] = []
        subscriber_received: list[str] = []

        @consumer.on(EventSignal)
        async def handle_consumer(signal: EventSignal):
            consumer_received.append(signal.event)

        @subscriber.on(EventSignal)
        async def handle_subscriber(signal: EventSignal):
            subscriber_received.append(signal.event)

        mesh = Mesh([producer, consumer, subscriber])
        mesh.connect(producer, consumer)  # Point-to-point
        mesh.create_topic("events")
        mesh.subscribe(subscriber, "events")  # Pub/sub

        async with mesh:
            # Point-to-point
            await producer.emit(EventSignal(event="direct"))
            # Pub/sub
            await mesh.publish("events", EventSignal(event="broadcast"))
            await asyncio.sleep(0.05)

        assert consumer_received == ["direct"]
        assert subscriber_received == ["broadcast"]
