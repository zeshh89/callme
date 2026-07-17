*This project has been created as part of the 42 curriculum by Jose-an2.

# call me maybe

## Description

**call me maybe** is a function calling system that translates natural language prompts into structured, machine-executable function calls, using a small (0.6B parameter) local LLM (`Qwen/Qwen3-0.6B`).

Given a prompt like:

```
"What is the sum of 40 and 2?"
```

the program does not answer the question directly. Instead, it identifies which function should be called and with which arguments, producing:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The core challenge of the project is that small language models are notoriously unreliable at producing structured output when simply prompted to do so (success rates as low as ~30%). To solve this reliably, the project implements **constrained decoding**: a technique that restricts, at each generation step, which tokens the model is allowed to produce, guaranteeing syntactically valid and schema-compliant output regardless of the model's raw uncertainty.

The pipeline is split into two stages:

1. **Function selection** — choosing which function best matches the user's request, constrained to only ever produce the name of a function that actually exists.
2. **Parameter extraction** — extracting each argument's value from the prompt, one parameter at a time, with generation constrained according to the expected type (`string`, `number`, `integer`, `boolean`).

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) as package/dependency manager
- The `llm_sdk` package (provided alongside the project, copied into the project root)

### Installation

```bash
git clone <repo_url>
cd callme
uv sync
```

This installs all dependencies declared in `pyproject.toml` / `uv.lock` (including `pydantic` and `numpy`), inside an isolated virtual environment managed by `uv`.

### Running the program

The program reads prompts and function definitions from `data/input/`, and writes results to `data/output/` by default:

```bash
uv run python -m src
```

You can override any of the input/output paths explicitly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

On first run, the model weights and tokenizer files for `Qwen/Qwen3-0.6B` are downloaded automatically from the Hugging Face Hub (~1.5 GB) and cached locally for subsequent runs.

### Development commands (Makefile)

```bash
make install       # install dependencies
make run           # run the program with default paths
make debug         # run the program under pdb
make lint          # run flake8 + mypy
make lint-strict   # run flake8 + mypy --strict
make clean         # remove __pycache__, .mypy_cache, etc.
```

## Example usage

Given `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

and `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 40 and 2?" }
]
```

running:

```bash
uv run python -m src
```

produces console output such as:

```
Loading model...
Model loaded

--- Processing 1/1 ---
Predicted function: fn_add_numbers
OK: fn_add_numbers -> {'a': 40.0, 'b': 2.0}

DONE. Results saved.
```

and writes `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": { "a": 40.0, "b": 2.0 }
  }
]
```

## Algorithm explanation

### Overview of constrained decoding

At each generation step, an LLM produces a probability distribution (logits) over its entire vocabulary. Left unconstrained, the model is free to output any token — including ones that would break JSON syntax, invent a parameter name, or hallucinate a function that doesn't exist. Constrained decoding intervenes **before** a token is selected:

1. The model produces logits for the next token, given everything generated so far.
2. Based on the current generation state, the set of *valid* next tokens is computed.
3. Every token outside that valid set has its logit forced to `-inf`.
4. The token is picked (greedily, via `argmax`) only among the remaining valid candidates.

Because invalid tokens are made mathematically impossible to select, the output is guaranteed to respect the intended structure at every step — not just "usually", but always.

### Function selection: trie-constrained decoding

Function names are known in advance (they come from `functions_definition.json`). Instead of letting the model generate a name freely and hoping it matches, all valid function names are pre-encoded into a **token-level trie** (`FunctionTrie`):

- Each function name is tokenized and inserted into the trie, so that shared prefixes across function names collapse into shared trie paths.
- At each decoding step, `get_allowed_tokens(generated_so_far)` walks the trie with the tokens generated so far and returns only the children of the current node — i.e. only tokens that could still lead to a valid, existing function name.
- `mask_logits` sets every other token's logit to `-inf`, and the model picks greedily among what remains.
- Decoding stops once the trie marks the current path as a complete function name (`is_complete`).

This makes it **structurally impossible** for the model to output a function name that isn't in `functions_definition.json`.

### Parameter extraction: type-constrained decoding + prompting

Parameter *names* are never generated by the model — they are already known from the schema, so each value is generated independently, one parameter at a time, and assembled into the final `parameters` dict in code. This eliminates two classes of errors that occurred in early iterations: invented parameter names, and parameters copied into the wrong slot.

The values themselves are constrained according to their declared type:

- **`number` / `integer`** — decoding accepts only tokens whose decoded text consists exclusively of digits, `.` and `-`. Leading whitespace/newline tokens (which a model naturally emits right after a prompt like `"Value:"`) are skipped before accumulation starts; generation stops as soon as a non-numeric token appears after digits have started accumulating.
- **`boolean`** — reuses the same trie mechanism as function selection, restricted to the two-token set `{"true", "false"}`.
- **`string`** — generation proceeds token by token and stops at the first newline, discarding any surrounding quotes the model may have added.

This guarantees **syntactic and type validity** (a `number` field can never contain a quoted string, a `boolean` can never be anything other than `true`/`false`). It does **not**, by itself, guarantee that the *value* is the semantically correct one — that responsibility falls on prompting (see Design decisions and Challenges faced below), since no grammar can encode "this must be the actual number the user meant" without understanding the sentence.

### Simplified pipeline diagram

```
Prompt
  │
  ▼
[1] Function selection prompt  →  trie-constrained greedy decoding  →  function name (guaranteed valid)
  │
  ▼
[2] For each parameter in the selected function's schema:
        build a per-parameter extraction prompt
              │
              ▼
        type-constrained decoding (number / integer / boolean / string)
              │
              ▼
        parameter value (guaranteed valid type)
  │
  ▼
Assemble { prompt, name, parameters } → append to output array
```

## Design decisions

- **One parameter per call, not one JSON blob per call.** An early version asked the model to emit the entire `parameters` object as free-form JSON in a single generation pass. This repeatedly produced invented parameter names and misassigned values when a function had several parameters of the same type (e.g. mixing up `regex` and `replacement`). Generating one value per parameter, with the parameter name supplied by code rather than by the model, removed an entire class of structural errors and let each generation step focus on a single, well-scoped extraction task.
- **Keys and structure are never left to the model.** JSON keys (`prompt`, `name`, `parameters`, and every parameter name) are always written by the program itself, using `pydantic` models (`FunctionCallResult`) to serialize the final result. The model is only ever asked to produce *values*, which are the one part of the output that genuinely requires natural-language understanding.
- **Few-shot examples with non-colliding placeholders.** Early prompts used realistic example words (e.g. `"cat"` / `"dog"`) to demonstrate the expected extraction format. This caused the model to occasionally leak those exact words into unrelated real prompts. Switching example placeholders to tokens that could never legitimately appear in a real request (e.g. `XWORDX` / `YWORDY`), combined with an explicit `=== ACTUAL REQUEST ===` separator, reduced this cross-contamination significantly.
- **"Extractor, not calculator" framing.** The single biggest source of semantic errors was the model *answering* the question instead of extracting the raw values needed to compute the answer (e.g. returning `12` instead of `144` for "square root of 144", or the already-reversed string instead of the original one for "reverse 'hello'"). Explicitly framing the model's role as a "PARAMETER EXTRACTOR, not a calculator", paired with contrastive examples showing the wrong behaviour next to the right one, was the single change with the largest measured impact on accuracy.
- **`pydantic` for all data validation.** `FunctionDefinition`, `ParameterDefinition`, `PromptInput`, and `FunctionCallResult` are all `pydantic` models. This gives free, descriptive validation errors for malformed input files (missing fields, wrong types) and guarantees the final serialized output always matches the expected schema shape.
- **Graceful per-prompt error handling.** Each prompt is processed inside its own `try/except` in `eval_runner.py`; a failure on one prompt (e.g. malformed generation, unknown function) is caught, logged, and recorded as an `"ERROR"` entry rather than crashing the entire run, so that a single hard prompt never prevents the rest of the batch from completing.

## Performance and reliability analysis

- **JSON / schema validity: 100%.** Because constrained decoding restricts token choice at the grammar level (trie for function names, type-restricted alphabets for parameter values), every entry in `function_calling_results.json` is guaranteed to be parseable JSON with the exact keys and types declared in `functions_definition.json`. This is verified automatically with `validate_results.py` (see Testing strategy below), which checks every entry against the schema and reports a `schema-compliant / total` percentage.
- **Function selection accuracy: high, close to 100% on tested sets.** Because the trie constrains the model to only ever output existing function names, the only failure mode is picking the *wrong* (but valid) function — this was observed only on a deliberately ambiguous prompt ("replace all numbers in ... with NUMBERS"), which could plausibly map to more than one function depending on interpretation.
- **Parameter value accuracy: generally 90%+ on literal extraction, lower on inferred values.** Prompts that require literal copying of text or numbers from the request (names, strings to reverse, numbers to sum) are extracted correctly essentially every time after the "extractor, not calculator" prompt redesign. Prompts that require the model to *synthesize* a value not literally present in the text (e.g. turning "replace all vowels" into a `[aeiouAEIOU]` regex pattern) are noticeably less reliable — this is a limitation of model size and reasoning capacity rather than of the decoding mechanism (see Challenges faced).
- **Speed:** on standard hardware (CPU, no GPU acceleration used), the full battery of test prompts processes well within the 5-minute budget required by the subject; most of the wall-clock time on a cold run is spent downloading and loading the model weights rather than generating output.
- **Robustness:** malformed or missing input files are caught explicitly (`FileNotFoundError` / `json.JSONDecodeError` / `pydantic.ValidationError`) and reported with a clear message instead of crashing with a raw traceback; a single failing prompt does not abort the batch.

## Challenges faced

- **The model "answering" instead of "extracting".** The most persistent issue throughout development was the model solving the user's question (e.g. reversing a string, computing a square root) instead of copying the literal input value the function needed. This was mitigated with explicit "extractor, not calculator" framing and contrastive few-shot examples showing the exact wrong-vs-right behaviour, though it can never be fully eliminated with certainty on a model this small — it is documented here as a known residual risk rather than a solved problem.
- **Few-shot example leakage.** Using realistic words in few-shot examples (`"cat"`, `"dog"`) caused the model to sometimes copy those exact words into unrelated prompts about different topics (e.g. producing `regex: "cat"` for a prompt about vowels). Switching to clearly artificial placeholders and an explicit "ACTUAL REQUEST" marker reduced, but did not entirely eliminate, this behaviour.
- **Multi-parameter disambiguation.** When a function has several parameters of the same type (e.g. `source_string`, `regex`, `replacement`, all strings), the model initially had no way to know which part of the sentence belonged to which slot, and would often copy the entire sentence into every parameter. This was solved by extracting one parameter at a time, passing already-extracted values back into the prompt, and adding an explicit rule ("the value must be different from the ones already extracted") plus a positional rule specific to substitution-style prompts ("the word after 'with' is the replacement").
- **Digits vs. quoted digits.** An early version of the number decoder produced quoted numeric strings (e.g. `"2"` instead of `2`) because nothing constrained the *type* of the generated tokens, only their JSON validity. This was fixed by writing a dedicated, type-aware value decoder per parameter type instead of relying on a single generic JSON generator.
- **Leading whitespace tokens breaking number extraction.** After switching to per-parameter extraction, the number decoder initially aborted immediately because the very first token generated after `"Value:"` was a whitespace/newline token, which was (incorrectly) treated as "not a digit, therefore no number present." Fixing this required distinguishing between "skippable whitespace before the value starts" and "a genuine non-numeric character after the value has started."
- **Evolving input schema.** During peer-review-style testing, the function definitions file used a `"integer"` type that the original schema (`Literal["string", "number", "boolean"]`) did not accept, causing a `pydantic.ValidationError` at load time. This confirmed the subject's warning that input files may change during review, and led to broadening the schema (and the corresponding value decoder) to explicitly support `integer` as distinct from `number`.
- **Circular imports.** Refactoring `greedy_decode` to be shared between the function-selection and parameter-extraction code paths initially caused a circular import between `pipeline.py` and `value_decoder.py`. This was resolved by extracting `greedy_decode` into its own module (`decoding.py`) that both other modules depend on, rather than depending on each other.

## Testing strategy

- **Manual end-to-end runs** against the provided example prompts and function definitions, inspecting both the console log (`Predicted function: ... -> {...}`) and the resulting `data/output/function_calling_results.json` after every significant prompt/code change, to catch regressions immediately.
- **Automated schema validation** via `validate_results.py`, a standalone script that checks, for every entry in the results file:
  - the file as a whole is valid JSON;
  - each entry has exactly the required top-level keys (`prompt`, `name`, `parameters`), no more, no less;
  - `name` matches an existing function in `functions_definition.json`;
  - `parameters` has exactly the keys declared for that function — catching both missing and invented parameters;
  - each parameter's value matches its declared type (`number`, `integer`, `string`, `boolean`).

  Run it with:

  ```bash
  uv run python validate_results.py \
      --functions data/input/functions_definition.json \
      --results data/output/function_calling_results.json
  ```

  It prints a per-entry pass/fail report plus a summary percentage of schema-compliant entries, which is the figure quoted under Performance analysis above.

- **Edge case prompts**, deliberately including: numbers embedded inside narrative text rather than stated plainly, ambiguous prompts that could map to more than one function, requests requiring an inferred (non-literal) value such as a regex character class, and functions with several parameters of the same type — to specifically probe the failure modes described in Challenges faced.
- **Testing against a second, independently generated set of function definitions and prompts** (generated via the project's grading tool, `moulinette generate_tests_and_corrections`), to confirm the solution generalizes beyond the example inputs shipped with the subject rather than being tuned specifically to them, in line with the subject's explicit warning that input files may change during peer review.

## Resources

- [Anthropic — Tool use / function calling overview](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Guidance / Outlines — structured generation and constrained decoding concepts](https://github.com/dottxt-ai/outlines)
- [Hugging Face — Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [BPE tokenization overview — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)

### AI usage

AI assistance (Claude) was used throughout this project as a **debugging and design-review partner**, not as a code generator for the core logic:

- Diagnosing runtime bugs from tracebacks and console output (circular imports, path resolution issues, `pydantic` schema mismatches when the input files introduced an `"integer"` type not originally handled).
- Iteratively reviewing and refining the prompting strategy for parameter extraction, after observing specific failure patterns in real output (e.g. the model computing answers instead of extracting inputs, or copying few-shot example words into unrelated prompts).
- Reviewing the constrained-decoding logic (trie-based function name masking, type-restricted value decoding) for correctness against the observed model behaviour.
- Drafting the structure and wording of this README, based on the actual implementation and the issues genuinely encountered during development.

Every suggestion was tested against the real model and real output before being kept; several early AI-suggested approaches (e.g. a single generic JSON generator for all parameters at once) were abandoned after testing revealed they did not solve the underlying problem, in favor of the type-specific, per-parameter decoding approach described above.