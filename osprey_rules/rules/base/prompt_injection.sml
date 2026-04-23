Import(rules=['models/base.sml'])

# AML.TA0004 — Model Evasion via Prompt Injection
PromptInjectionRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='ignore previous instructions') or
            TextContains(text=QueryPreview, phrase='disregard your rules') or
            TextContains(text=QueryPreview, phrase='bypass filter') or
            TextContains(text=QueryPreview, phrase='jailbreak') or
            TextContains(text=QueryPreview, phrase='override safety') or
            TextContains(text=QueryPreview, phrase='as an ai with no restrictions') or
            TextContains(text=QueryPreview, phrase='ignore all prior') or
            TextContains(text=QueryPreview, phrase='do anything now')
        ),
    ],
    description=f"Prompt injection detected in query — AML.TA0004",
)

WhenRules(
    rules_any=[PromptInjectionRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='prompt_injection_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
