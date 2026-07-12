"""
Validation script for function_calling_results.json

Checks:
  1. The output file is valid JSON (parseable).
  2. Each entry has exactly the required top-level keys: prompt, name, parameters.
  3. `name` matches an existing function in functions_definition.json.
  4. `parameters` has exactly the parameter names defined for that function
     (no missing keys, no extra/invented keys).
  5. Each parameter value matches the expected type from the schema
     (number -> int/float, string -> str, boolean -> bool).

Usage:
    uv run python validate_results.py \
        --functions data/input/functions_definition.json \
        --results data/output/function_calling_results.json

Exit code is 0 if everything passes, 1 otherwise. A per-entry report is
printed, plus a summary with the metrics requested in the project README
(README > Performance analysis): % valid JSON, % schema-compliant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TYPE_CHECKS = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
}

REQUIRED_TOP_KEYS = {"prompt", "name", "parameters"}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_function_index(functions: list[dict]) -> dict[str, dict]:
    return {fn["name"]: fn for fn in functions}


def validate_entry(entry: dict, functions_by_name: dict[str, dict]) -> list[str]:
    """Return a list of error strings for a single result entry (empty = OK)."""
    errors: list[str] = []

    # 1. Top-level keys
    entry_keys = set(entry.keys())
    if entry_keys != REQUIRED_TOP_KEYS:
        missing = REQUIRED_TOP_KEYS - entry_keys
        extra = entry_keys - REQUIRED_TOP_KEYS
        if missing:
            errors.append(f"missing top-level keys: {sorted(missing)}")
        if extra:
            errors.append(f"unexpected top-level keys: {sorted(extra)}")
        # If keys are broken we can still try to continue checking what exists.

    if not isinstance(entry.get("prompt"), str):
        errors.append("`prompt` is not a string")

    name = entry.get("name")
    if not isinstance(name, str):
        errors.append("`name` is not a string")
        return errors  # can't check further without a valid name

    function_def = functions_by_name.get(name)
    if function_def is None:
        errors.append(f"`name` = {name!r} does not match any known function")
        return errors

    parameters = entry.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("`parameters` is not an object")
        return errors

    expected_params: dict = function_def.get("parameters", {})
    expected_names = set(expected_params.keys())
    actual_names = set(parameters.keys())

    missing_params = expected_names - actual_names
    extra_params = actual_names - expected_names

    if missing_params:
        errors.append(f"missing parameters: {sorted(missing_params)}")
    if extra_params:
        errors.append(f"unexpected/invented parameters: {sorted(extra_params)}")

    for param_name in expected_names & actual_names:
        expected_type = expected_params[param_name]["type"]
        value = parameters[param_name]
        check = TYPE_CHECKS.get(expected_type)
        if check is None:
            errors.append(f"unknown expected type {expected_type!r} for '{param_name}'")
            continue
        if not check(value):
            errors.append(
                f"parameter '{param_name}' expected type {expected_type!r}, "
                f"got {type(value).__name__} ({value!r})"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate function calling results.")
    parser.add_argument(
        "--functions",
        default="data/input/functions_definition.json",
        help="Path to functions_definition.json",
    )
    parser.add_argument(
        "--results",
        default="data/output/function_calling_results.json",
        help="Path to function_calling_results.json",
    )
    args = parser.parse_args()

    functions_path = Path(args.functions)
    results_path = Path(args.results)

    if not functions_path.exists():
        print(f"ERROR: functions definition file not found: {functions_path}")
        return 1
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}")
        return 1

    # --- 1. Valid JSON check ---
    try:
        functions = load_json(functions_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: functions_definition.json is not valid JSON: {exc}")
        return 1

    try:
        results = load_json(results_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: results file is not valid JSON: {exc}")
        print("=> 0% valid JSON. Aborting further checks.")
        return 1

    if not isinstance(results, list):
        print("ERROR: results file must contain a top-level JSON array.")
        return 1

    functions_by_name = build_function_index(functions)

    total = len(results)
    if total == 0:
        print("WARNING: results file is empty, nothing to validate.")
        return 0

    valid_count = 0
    print(f"Validating {total} entries against schema...\n")

    for i, entry in enumerate(results):
        if not isinstance(entry, dict):
            print(f"[{i}] FAIL — entry is not a JSON object")
            continue

        errors = validate_entry(entry, functions_by_name)
        prompt_preview = str(entry.get("prompt", ""))[:60]

        if errors:
            print(f"[{i}] FAIL — prompt: {prompt_preview!r}")
            for err in errors:
                print(f"      - {err}")
        else:
            valid_count += 1
            print(f"[{i}] OK   — prompt: {prompt_preview!r} -> {entry['name']}")

    pct = (valid_count / total) * 100
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Valid JSON file:            YES")
    print(f"Total entries:              {total}")
    print(f"Schema-compliant entries:   {valid_count} / {total} ({pct:.1f}%)")
    print("=" * 50)

    return 0 if valid_count == total else 1


if __name__ == "__main__":
    sys.exit(main())