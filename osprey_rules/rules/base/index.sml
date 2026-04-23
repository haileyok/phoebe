Import(rules=['models/base.sml'])

Require(
    rule='rules/base/prompt_injection.sml',
    require_if=QueryPreview != None,
)

Require(
    rule='rules/base/authority_claim.sml',
    require_if=QueryPreview != None,
)

Require(
    rule='rules/base/data_exfiltration.sml',
    require_if=QueryPreview != None,
)

Require(
    rule='rules/base/privilege_escalation.sml',
    require_if=IsAgentic == True,
)
