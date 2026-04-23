Import(rules=['models/base.sml'])

# AML.TA0003 — Privilege escalation via agentic tool calls
PrivilegeEscalationRule = Rule(
    when_all=[
        IsAgentic == True,
        HasToolCalls == True,
        (
            TextContains(text=RequestedTools, phrase='filesystem') or
            TextContains(text=RequestedTools, phrase='network') or
            TextContains(text=RequestedTools, phrase='shell') or
            TextContains(text=RequestedTools, phrase='database_write') or
            TextContains(text=RequestedTools, phrase='email_send')
        ),
    ],
    description=f"Privilege escalation via agentic tool request — AML.TA0003",
)

WhenRules(
    rules_any=[PrivilegeEscalationRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='escalation_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
