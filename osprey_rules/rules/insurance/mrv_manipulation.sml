Import(rules=['models/insurance.sml'])

# AML.TA0004 — MRV (Measurement, Reporting, Verification) data manipulation
MrvManipulationRule = Rule(
    when_all=[
        (
            TextContains(text=QueryPreview, phrase='override sensor reading') or
            TextContains(text=QueryPreview, phrase='adjust baseline manually') or
            TextContains(text=QueryPreview, phrase='ignore weather station data') or
            TextContains(text=QueryPreview, phrase='modify the catastrophe model input') or
            TextContains(text=QueryPreview, phrase='override the flood zone classification') or
            TextContains(text=QueryPreview, phrase='override ecomonitor data')
        ),
    ],
    description=f"MRV data manipulation attempt — AML.TA0004",
)

WhenRules(
    rules_any=[MrvManipulationRule],
    then=[
        DeclareVerdict(verdict='block'),
        LabelAdd(entity=UserId, label='mrv_manipulation_attempt',
                 expires_after=TimeDelta(hours=24)),
    ],
)
