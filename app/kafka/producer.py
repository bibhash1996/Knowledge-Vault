import asyncio
from typing import Iterable, Optional

from aiokafka import AIOKafkaProducer

_producer: Optional[AIOKafkaProducer] = None


async def init_producer(bootstrap_servers: Iterable[str]):
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(bootstrap_servers=list(bootstrap_servers))
        await _producer.start()
    return _producer


async def stop_producer():
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None


async def send_message(topic: str, value: bytes, key: Optional[bytes] = None):
    if _producer is None:
        raise RuntimeError("Kafka producer is not initialized")

    await _producer.send_and_wait(topic, value=value, key=key)


def get_producer_sync():
    """Return underlying producer instance (may be None)."""
    return _producer
