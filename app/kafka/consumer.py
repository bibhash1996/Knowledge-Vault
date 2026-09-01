import asyncio
import logging
from typing import Awaitable, Callable, Iterable, Optional

from aiokafka import AIOKafkaConsumer


MessageHandler = Callable[[object], Awaitable[None]]


class KafkaConsumerManager:
    def __init__(
        self,
        bootstrap_servers: Iterable[str],
        topic: str,
        group_id: str,
        handler: MessageHandler,
    ):
        self.bootstrap_servers = list(bootstrap_servers)
        self.topic = topic
        self.group_id = group_id
        self.handler = handler

        self._consumer: Optional[AIOKafkaConsumer] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._consumer = AIOKafkaConsumer(
            self.topic, bootstrap_servers=self.bootstrap_servers, group_id=self.group_id
        )
        await self._consumer.start()
        self._task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        try:
            async for msg in self._consumer:
                try:
                    await self.handler(msg)
                except Exception:
                    logging.exception("Error handling Kafka message")
        except asyncio.CancelledError:
            pass
        finally:
            if self._consumer is not None:
                await self._consumer.stop()

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer is not None:
            await self._consumer.stop()
