# Insurance event model
Import(rules=['models/base.sml'])

SimilarClaimCount24h: Optional[int] = JsonData(
    path='$.similar_claim_count_24h',
    required=False,
)

ClaimValue: Optional[float] = JsonData(
    path='$.claim_value',
    required=False,
)

RequestedTools: Optional[str] = JsonData(
    path='$.requested_tools_json',
    required=False,
)

GrantedTools: Optional[str] = JsonData(
    path='$.granted_tools_json',
    required=False,
)
