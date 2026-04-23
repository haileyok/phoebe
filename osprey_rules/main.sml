Import(rules=['models/base.sml'])

# Always run base safety rules
Require(rule='rules/base/index.sml')

# Run insurance rules when domain is insurance
Require(
    rule='rules/insurance/index.sml',
    require_if=Domain == 'insurance',
)
