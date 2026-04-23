Import(rules=['models/insurance.sml'])

# AML.TA0007 — Regulatory violation / discriminatory underwriting
RegulatoryViolationRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='exclude coverage based on postcode') or
            TextContains(text=QueryPreview, phrase='apply higher premium for') or
            TextContains(text=QueryPreview, phrase='deny claim because of nationality') or
            TextContains(text=QueryPreview, phrase='use gender to calculate') or
            TextContains(text=QueryPreview, phrase='discriminate based on') or
            TextContains(text=QueryPreview, phrase='redline this area')
        ),
    ],
    description=f"Regulatory violation / discriminatory underwriting detected — AML.TA0007",
)

WhenRules(
    rules_any=[RegulatoryViolationRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='regulatory_violation',
                 expires_after=TimeDelta(hours=72)),
        LabelAdd(entity=SessionId, label='requires_human_review'),
    ],
)
