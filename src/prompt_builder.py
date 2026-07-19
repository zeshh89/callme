from src.models import (
    FunctionDefinition,
    ParameterDefinition,
    ParameterValue
)
from typing import Mapping

REGEX_CATEGORY_HINTS = """
Some regex parameters describe a CATEGORY of characters instead
of literal text.
When that happens, translate the category into the matching regex pattern:
- "vowels"       -> [aeiouAEIOU]
- "consonants"   -> [^aeiouAEIOU\\s]
- "digits" / "numbers" -> [0-9]
- "whitespace" / "spaces" -> \\s
- "punctuation"  -> [.,!?;:]

Example:
User request: "Replace all vowels in 'Programming is fun' with asterisks"
Parameter "regex" -> Value: [aeiouAEIOU]
(NOT the whole sentence, and NOT the literal word "vowels" —
use the character class.)

Example:
User request: "Substitute the word 'cat' with 'dog'
in 'The cat sat on the mat'"
Parameter "regex" -> Value: cat
(Here the request names a literal word, so copy it as-is
instead of using a character class.)
"""

SUBSTITUTION_HINTS = """
For substitution/replace tasks, the request usually has this shape:
"Substitute/Replace <TARGET> with <REPLACEMENT> in <SOURCE_TEXT>"

- <TARGET>      -> goes into "regex" (or similar target parameter)
- <REPLACEMENT> -> the word/phrase that comes right after "with"
- <SOURCE_TEXT> -> the full text after "in"

FORMAT EXAMPLE ONLY (do not reuse these words,
they are illustrative placeholders):
User request: "Substitute the word 'XWORDX' with 'YWORDY'
in 'a text with XWORDX inside'"
Parameter "source_string" -> Value: a text with XWORDX inside
Parameter "regex"         -> Value: XWORDX
Parameter "replacement"   -> Value: YWORDY

IMPORTANT: The example above is ONLY to show the expected FORMAT.
NEVER copy "XWORDX",
"YWORDY", "cat", or "dog" into your answer unless those
exact words appear in the
ACTUAL user request below. Always re-read the actual request before answering.
"""


def build_value_prompt(
        user_prompt: str,
        function: FunctionDefinition,
        param_name: str,
        param: ParameterDefinition,
        already_filled: Mapping[str, ParameterValue]
) -> str:
    filled = "\n".join(
        f"- {k}: {v}" for k, v in already_filled.items()
    ) or "(none yet)"
    already_note = (
        f'\nNote: the value for "{param_name}" must be DIFFERENT from the '
        'values already extracted above.'
        if already_filled else ""
    )

    extra_hints = ""
    if param_name in ("regex", "replacement", "source_string"):
        extra_hints = SUBSTITUTION_HINTS
    if param_name == "regex":
        extra_hints += REGEX_CATEGORY_HINTS

    return f"""You are a PARAMETER EXTRACTOR, not a calculator.
Copy the value exactly as it appears in the user request. NEVER compute, solve,
reverse, or transform it,
UNLESS a specific rule below tells you otherwise.
{extra_hints}
User request: "{user_prompt}"
Function: {function.name} — {function.description}
Already extracted:
{filled}{already_note}

Now output ONLY the raw value (no quotes, no explanation)
for parameter "{param_name}" (type: {param.type}).
Value:"""
