"""
Tests for Message Bus
"""

import asyncio
import pytest

from app.core.message_bus import MessageBus
from app.models.schemas import ResultMessage, TaskMessage


@pytest.fixture
def message_bus():
    """Create a fresh message bus for each test."""
    return MessageBus()


@pytest.mark.asyncio
async def test_subscribe_and_publish(message_bus):
    """Test basic pub/sub functionality."""
    received_messages = []
    
    async def handler(msg):
        received_messages.append(msg)
    
    await message_bus.subscribe("test_topic", handler)
    
    task = TaskMessage(
        job_id="test-123",
        image_path="/path/to/image.jpg"
    )
    
    await message_bus.publish("test_topic", task)
    
    assert len(received_messages) == 1
    assert received_messages[0].job_id == "test-123"


@pytest.mark.asyncio
async def test_multiple_subscribers(message_bus):
    """Test multiple handlers for same topic."""
    received_1 = []
    received_2 = []
    
    async def handler1(msg):
        received_1.append(msg)
    
    async def handler2(msg):
        received_2.append(msg)
    
    await message_bus.subscribe("test_topic", handler1)
    await message_bus.subscribe("test_topic", handler2)
    
    task = TaskMessage(job_id="test-456", image_path="/path/to/image.jpg")
    await message_bus.publish("test_topic", task)
    
    assert len(received_1) == 1
    assert len(received_2) == 1


@pytest.mark.asyncio
async def test_idempotent_handling(message_bus):
    """Test that duplicate messages are ignored."""
    received = []
    
    async def handler(msg):
        received.append(msg)
    
    await message_bus.subscribe("results", handler)
    
    # Same message published twice
    result = ResultMessage(
        job_id="test-789",
        agent_name="TestAgent",
        score=0.75,
        explanation="Test",
        latency_ms=100.0
    )
    
    await message_bus.publish("results", result)
    await message_bus.publish("results", result)  # Duplicate
    
    # Should only receive once due to idempotence
    assert len(received) == 1


@pytest.mark.asyncio
async def test_event_queue_registration(message_bus):
    """Test job event queue registration."""
    queue = await message_bus.register_job_events("job-123")
    
    assert queue is not None
    
    await message_bus.emit_event("job-123", "test_event", {"data": "test"})
    
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event.event_type == "test_event"
    assert event.data["data"] == "test"


@pytest.mark.asyncio
async def test_unsubscribe(message_bus):
    """Test handler unsubscription."""
    received = []
    
    async def handler(msg):
        received.append(msg)
    
    await message_bus.subscribe("topic", handler)
    
    task = TaskMessage(job_id="test-1", image_path="/path")
    await message_bus.publish("topic", task)
    
    assert len(received) == 1
    
    await message_bus.unsubscribe("topic", handler)
    
    task2 = TaskMessage(job_id="test-2", image_path="/path")
    await message_bus.publish("topic", task2)
    
    # Should still be 1 since we unsubscribed
    assert len(received) == 1


@pytest.mark.asyncio
async def test_handler_error_isolation(message_bus):
    """Test that handler errors don't affect other handlers."""
    received = []
    
    async def failing_handler(msg):
        raise Exception("Handler error")
    
    async def working_handler(msg):
        received.append(msg)
    
    await message_bus.subscribe("topic", failing_handler)
    await message_bus.subscribe("topic", working_handler)
    
    task = TaskMessage(job_id="test", image_path="/path")
    await message_bus.publish("topic", task)
    
    # Working handler should still receive message
    assert len(received) == 1
