Import(rules=['models/insurance.sml'])

Require(
    rule='rules/insurance/claims_manipulation.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/pricing_extraction.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/regulatory_violation.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/mrv_manipulation.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/coordinated_fraud.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/pii_exposure.sml',
    require_if=Domain == 'insurance',
)

Require(
    rule='rules/insurance/underwriting_bias.sml',
    require_if=Domain == 'insurance',
)
