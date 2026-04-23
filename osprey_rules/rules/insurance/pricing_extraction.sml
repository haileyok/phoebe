Import(rules=['models/insurance.sml'])

# AML.TA0005 — Pricing algorithm extraction / model inversion
PricingExtractionRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='show me the rating factors') or
            TextContains(text=QueryPreview, phrase='how is my risk score calculated') or
            TextContains(text=QueryPreview, phrase='explain your pricing algorithm') or
            TextContains(text=QueryPreview, phrase='reveal the underwriting criteria') or
            TextContains(text=QueryPreview, phrase='what are the model weights') or
            TextContains(text=QueryPreview, phrase='what factors increase my premium')
        ),
    ],
    description=f"Pricing algorithm extraction attempt — AML.TA0005",
)

WhenRules(
    rules_any=[PricingExtractionRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='pricing_extraction_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
