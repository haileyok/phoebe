# Base model — fields present in every Sara routing event
EventType: str = JsonData(path='$.event_type')

EventId: Entity[str] = EntityJson(
    type='EventId',
    path='$.event_id',
)

UserId: Entity[str] = EntityJson(
    type='UserId',
    path='$.user_id_hash',
)

SessionId: Entity[str] = EntityJson(
    type='SessionId',
    path='$.session_id',
)

ModelId: str = JsonData(path='$.model_id')

TaskType: str = JsonData(path='$.task_type')

Domain: str = JsonData(path='$.domain')

QueryPreview: Optional[str] = JsonData(
    path='$.query_preview',
    required=False,
)

IsAgentic: Optional[bool] = JsonData(
    path='$.is_agentic',
    required=False,
)

HasToolCalls: Optional[bool] = JsonData(
    path='$.has_tool_calls',
    required=False,
)

RoutingConfidence: float = JsonData(path='$.routing_confidence')
