Import(rules=['models/base.sml'])

# AML.TA0007 — Impact via false authority claim
AuthorityClaimRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='admin override') or
            TextContains(text=QueryPreview, phrase='disable safety checks') or
            TextContains(text=QueryPreview, phrase='maintenance mode') or
            TextContains(text=QueryPreview, phrase='security exception') or
            TextContains(text=QueryPreview, phrase='override mode') or
            TextContains(text=QueryPreview, phrase='i am authorized to')
        ),
    ],
    description=f"False authority claim detected — AML.TA0007",
)

WhenRules(
    rules_any=[AuthorityClaimRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='authority_claim_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
