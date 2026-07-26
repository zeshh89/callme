import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.models import (
    FunctionDefinition,
    PromptInput,
    FunctionCallResult,
)


def load_function_definitions(
    filepath: str,
) -> List[FunctionDefinition]:
    """
    Load and validate function definitions from JSON file.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Function definition file not found: {filepath}"
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        functions = []

        for index, item in enumerate(data):
            try:
                functions.append(
                    FunctionDefinition(**item)
                )
            except ValidationError:
                raise ValueError(
                    f"Invalid function definition at entry {index + 1}.\n\n"
                    "Please check the format of this function."
                ) from None

        return functions

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in function definitions: {exc}"
        ) from exc

    except ValidationError:
        raise ValueError(
            "Invalid function definition file."
        ) from None


def load_prompts(
    filepath: str,
) -> List[PromptInput]:
    """
    Load and validate prompts from JSON file.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {filepath}"
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompts = []

        for index, item in enumerate(data):
            try:
                prompts.append(
                    PromptInput(**item)
                )
            except ValidationError:
                raise ValueError(
                    f"Invalid prompt at entry {index + 1}.\n\n"
                    "Expected format:\n"
                    '{\n'
                    '    "prompt": "Your prompt here"\n'
                    '}'
                ) from None

        return prompts

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in prompts file: {exc}"
        ) from exc

    except ValidationError:
        raise ValueError(
            "Invalid prompt file."
        ) from None


def save_results(
    filepath: str,
    results: List[FunctionCallResult],
) -> None:
    """
    Save function calling results to JSON file.
    """

    path = Path(filepath)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            [
                result.model_dump()
                for result in results
            ],
            file,
            indent=4,
        )
