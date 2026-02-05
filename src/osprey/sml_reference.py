SYNTAX_DATA_EXTRACTION = """\
## Data Extraction

### JsonData - Extract values from event JSON
```python
FieldName: type = JsonData(path='$.json.path')
FieldName: Optional[type] = JsonData(path='$.path', required=False)
```

### EntityJson - Create typed entities from JSON (for IDs that can have labels)
```python
UserId: Entity[str] = EntityJson(type='UserId', path='$.user.id')
PostId: Entity[str] = EntityJson(type='PostId', path='$.postId')
```

### Supported types
- `str`, `int`, `float`, `bool`
- `List[str]`, `List[int]`, etc.
- `Optional[type]` for nullable fields
- `Entity[type]` for identifiers that can have labels attached
"""

SYNTAX_RULES = """\
## Rules

### Defining a Rule
```python
RuleName = Rule(
    when_all=[
        Condition1,
        Condition2,  # ALL conditions must be True
    ],
    description='What this rule detects'
)
```

### Rule naming
- Must be PascalCase (e.g., `SpamDetectionRule`)
- Cannot start with underscore
- Description must be a string literal or f-string
"""

SYNTAX_EFFECTS = """\
## Effects and WhenRules

### Wiring rules to effects
```python
WhenRules(
    rules_any=[Rule1, Rule2],  # ANY rule triggers ALL effects
    then=[
        DeclareVerdict(verdict='reject'),
        LabelAdd(entity=UserId, label='spam'),
    ],
)
```

### Available effects
- `DeclareVerdict(verdict='...')` - Return verdict to caller
- `LabelAdd(entity=E, label='L')` - Add label to entity
- `LabelRemove(entity=E, label='L')` - Remove label from entity
"""

SYNTAX_OPERATORS = """\
## Operators

### Comparison
- `==`, `!=`, `>`, `>=`, `<`, `<=`
- `in`, `not in` (for list membership)

### Boolean
- `and`, `or`, `not`
- All conditions in `when_all` are implicitly AND-ed
- Use parentheses for OR: `(Cond1 or Cond2)`

### Null handling
- `Value != None` - check if not null
- `Value == None` - check if null
- Always check for null before using optional fields
"""

SYNTAX_IMPORTS = """\
## File Organization

### Import - Include models/features from other files
```python
Import(rules=[
    'models/base.sml',
    'models/record/post.sml',
])
```
- Paths must be relative and sorted lexicographically
- Imported features become available in current file

### Require - Conditionally include rule files
```python
Require(rule='rules/spam/check.sml')
Require(rule='rules/post/links.sml', require_if=EventType == 'post')
```
- Use for conditional rule execution based on event type
- Required file outputs are NOT available in parent file
"""

PROJECT_STRUCTURE = """\
## Project Structure

Osprey rules projects follow this structure:

```
rules/
├── main.sml                    # Entry point - requires index.sml
├── models/
│   ├── base.sml               # Common features (UserId, Handle, etc.)
│   └── record/
│       ├── post.sml           # Post-specific features
│       ├── like.sml           # Like-specific features
│       └── profile.sml        # Profile-specific features
└── rules/
    ├── index.sml              # Routes to event-specific rules
    └── record/
        ├── post/
        │   ├── index.sml      # Requires post rules
        │   ├── spam.sml       # Spam detection
        │   └── links.sml      # Link abuse detection
        └── profile/
            ├── index.sml
            └── impersonation.sml
```

### Key principles
1. **models/** - Feature definitions only, no rules
2. **rules/** - Rule logic with WhenRules effects
3. **index.sml** - Conditional routing based on event type
4. Each rule file imports the models it needs
"""

PATTERN_BASIC_RULE = """\
### Basic Rule Pattern
```python
Import(rules=[
    'models/base.sml',
    'models/record/post.sml',
])

# Define the rule
SpamLinkRule = Rule(
    when_all=[
        AccountAgeSecondsUnwrapped < Day,
        PostHasExternal,
        PostIsReply,
    ],
    description='New account posting external links in replies',
)

# Wire to effects
WhenRules(
    rules_any=[SpamLinkRule],
    then=[
        LabelAdd(entity=UserId, label='reply-link-spam'),
    ],
)
```
"""

PATTERN_MULTIPLE_RULES = """\
### Multiple Rules with Tiered Response
```python
# Low severity
LowRiskRule = Rule(
    when_all=[Signal1],
    description='Single signal detected',
)

# High severity - multiple signals
HighRiskRule = Rule(
    when_all=[Signal1, Signal2, Signal3],
    description='Multiple signals detected',
)

WhenRules(
    rules_any=[LowRiskRule, HighRiskRule],
    then=[
        LabelAdd(entity=UserId, label='flagged'),
        LabelAdd(entity=UserId, label='high-risk', apply_if=HighRiskRule),
    ],
)
```
"""

PATTERN_COMPUTED_FEATURES = """\
### Computed Features
```python
# Compute intermediate values
FollowRatio = FollowingCount / (FollowersCount + 1)  # +1 to avoid division by zero
IsHighFollowRatio = FollowRatio > 10.0

MessageLength = StringLength(s=PostText)
IsShortMessage = MessageLength < 10

HasManyMentions = ListLength(list=FacetMentionList) > 5

# Use in rules
SuspiciousActivity = Rule(
    when_all=[
        IsHighFollowRatio,
        IsShortMessage,
        HasManyMentions,
    ],
    description='Suspicious activity pattern',
)
```
"""

PATTERN_NULL_SAFETY = """\
### Null-Safe Patterns
```python
# For optional fields, always check null first
OptionalField: Optional[str] = JsonData(path='$.maybe', required=False)

SafeRule = Rule(
    when_all=[
        OptionalField != None,  # Guard clause
        StringLength(s=OptionalField) > 10,
    ],
    description='Checks optional field safely',
)

# Or use ResolveOptional for defaults
SafeValue: str = ResolveOptional(
    optional_value=OptionalField,
    default_value='',
)
```
"""


def get_syntax_reference() -> str:
    return "\n\n".join(
        [
            "# SML Syntax Reference",
            SYNTAX_DATA_EXTRACTION,
            SYNTAX_RULES,
            SYNTAX_EFFECTS,
            SYNTAX_OPERATORS,
            SYNTAX_IMPORTS,
        ]
    )


def get_project_structure() -> str:
    return PROJECT_STRUCTURE


def get_patterns_reference() -> str:
    return "\n\n".join(
        [
            "# Common SML Patterns",
            PATTERN_BASIC_RULE,
            PATTERN_MULTIPLE_RULES,
            PATTERN_COMPUTED_FEATURES,
            PATTERN_NULL_SAFETY,
        ]
    )
