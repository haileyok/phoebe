Import(rules=['models/insurance.sml'])

# AML.TA0006 — PII exfiltration attempt
PiiExposureRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='show me all claims for') or
            TextContains(text=QueryPreview, phrase='list all policyholders') or
            TextContains(text=QueryPreview, phrase='export customer data') or
            TextContains(text=QueryPreview, phrase='give me the medical history of') or
            TextContains(text=QueryPreview, phrase='dump the claims database')
        ),
    ],
    description=f"PII exfiltration attempt detected — AML.TA0006",
)

WhenRules(
    rules_any=[PiiExposureRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='pii_exfiltration_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
