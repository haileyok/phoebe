from atkafka_consumer import AtKafkaEvent, Consumer
import atproto

from src.clickhouse.clickhouse import Clickhouse


class Indexer:
    """
    Some indexer process that I started to write...but this isn't really necessary?
    We can just use the events from the Osprey Clickhouse itself so this feels kinda
    pointless atp. I'll leave it here just incase it proves useful later
    """

    def __init__(
        self,
        bootstrap_servers: list[str],
        input_topic: str,
        group_id: str,
        clickhouse: Clickhouse,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._input_topic = input_topic
        self._group_id = group_id
        self._clickhouse = clickhouse

        self._indexer: Consumer | None = None

    async def run(self) -> None:
        raise NotImplementedError()

        self._indexer = Consumer(
            bootstrap_servers=self._bootstrap_servers,
            input_topic=self._input_topic,
            group_id=self._group_id,
            on_event=self._on_event,
            max_concurrent_tasks=1_000,
        )

    async def _on_event(
        self, evt: AtKafkaEvent | atproto.models.ToolsOzoneModerationDefs.ModEventView
    ):
        pass
