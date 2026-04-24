# Sara — Session Log

## What's Done

### Ozone Enforcement Layer (Session D)
- `src/ozone/ozone.py` — SYNC/ASYNC/QUARANTINE enforcement with rollback
- ClickHouse tables: `enforcement_log`, `rule_performance_metrics`
- `GET /api/ozone/evaluate` — HTTP endpoint in arena server
- Ozone config fields in `src/config.py`
- Test suite: `tests/test_ozone.py`

### Osprey Real Rule Engine Integration (Session E)
- `docker-compose.yaml` — Osprey worker, Kafka (Confluent 7.5), Zookeeper, Osprey UI API, Osprey UI added
- `osprey_rules/` — 11 SML rules: 4 base (injection, authority, exfiltration, escalation) + 7 insurance (claims, pricing, regulatory, MRV, fraud, PII, bias)
- `src/safety/osprey_client.py` — async Kafka adapter, 500ms timeout, automatic fallback to Python rules if Osprey unavailable
- `src/safety/monitor.py` — SaraMonitor tries Osprey first, Python keyword rules as fallback
- `GET /api/osprey/health` — health check endpoint
- `src/config.py` — added 6 Osprey config fields
- `tests/test_osprey_integration.py` — 8 tests, all passing

## Key Design Decisions

- Osprey as primary rule engine (SML) with Python rules as fallback
- 500ms Osprey timeout prevents latency impact on legitimate traffic
- Domain-conditional rules: insurance SML only runs when domain='insurance'
- TACTIC_MAP normalises underscores vs PascalCase to match rule names
