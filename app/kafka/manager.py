import asyncio
import logging
from typing import Iterable, Optional

from app.core.config import get_settings

from .producer import init_producer, stop_producer, send_message, get_producer_sync
from .consumer import KafkaConsumerManager


class KafkaManager:
    def __init__(self):
        self.settings = get_settings()
        self.consumer_manager: Optional[KafkaConsumerManager] = None

    async def start(self, app=None):
        if not self.settings.kafka_bootstrap_servers:
            logging.info("Kafka bootstrap servers not configured; skipping Kafka startup")
            return

        servers = [s.strip() for s in self.settings.kafka_bootstrap_servers.split(",") if s.strip()]
        await init_producer(servers)

        if self.settings.kafka_consumer_topic:
            async def _default_handler(msg):
                logging.info("Received Kafka message on %s: %s", msg.topic, msg.value)

            self.consumer_manager = KafkaConsumerManager(
                bootstrap_servers=servers,
                topic=self.settings.kafka_consumer_topic,
                group_id=self.settings.kafka_consumer_group,
                handler=_default_handler,
            )
            await self.consumer_manager.start()

    async def stop(self):
        if self.consumer_manager:
            await self.consumer_manager.stop()
        await stop_producer()


kafka_manager = KafkaManager()
