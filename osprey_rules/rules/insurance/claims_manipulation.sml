Import(rules=['models/insurance.sml'])

# AML.TA0007 — Claims manipulation (FCA ICOBS 8)
ClaimsManipulationRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='override the damage assessment') or
            TextContains(text=QueryPreview, phrase='ignore the policy limit') or
            TextContains(text=QueryPreview, phrase='mark this claim as approved') or
            TextContains(text=QueryPreview, phrase='waive the excess') or
            TextContains(text=QueryPreview, phrase='force approve this claim') or
            TextContains(text=QueryPreview, phrase='bypass the claims process') or
            TextContains(text=QueryPreview, phrase='change the claim status to paid')
        ),
    ],
    description=f"Claims manipulation attempt detected — FCA ICOBS 8 — AML.TA0007",
)

WhenRules(
    rules_any=[ClaimsManipulationRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='claims_manipulation',
                 expires_after=TimeDelta(hours=72)),
        LabelAdd(entity=SessionId, label='requires_human_review'),
    ],
)
