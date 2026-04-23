Import(rules=['models/insurance.sml'])

# AML.TA0004 — Coordinated fraud detection via claim velocity
CoordinatedFraudRule = Rule(
    when_all=[
        SimilarClaimCount24h >= 3,
    ],
    description=f"Coordinated fraud suspected — high claim velocity — AML.TA0004",
)

WhenRules(
    rules_any=[CoordinatedFraudRule],
    then=[
        DeclareVerdict(verdict='flag'),
        LabelAdd(entity=UserId, label='coordinated_fraud_suspected',
                 expires_after=TimeDelta(hours=72)),
        LabelAdd(entity=SessionId, label='requires_human_review'),
    ],
)
