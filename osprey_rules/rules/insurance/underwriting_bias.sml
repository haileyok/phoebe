Import(rules=['models/insurance.sml'])

# AML.TA0007 — Underwriting bias / discriminatory risk loading
UnderwritingBiasRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='assume higher risk because of age') or
            TextContains(text=QueryPreview, phrase='treat this demographic') or
            TextContains(text=QueryPreview, phrase='apply the standard loading for') or
            TextContains(text=QueryPreview, phrase='this postcode group should be')
        ),
    ],
    description=f"Underwriting bias / discriminatory risk loading detected — AML.TA0007",
)

WhenRules(
    rules_any=[UnderwritingBiasRule],
    then=[
        DeclareVerdict(verdict='flag'),
        LabelAdd(entity=UserId, label='underwriting_bias_flag',
                 expires_after=TimeDelta(hours=72)),
        LabelAdd(entity=SessionId, label='requires_human_review'),
    ],
)
