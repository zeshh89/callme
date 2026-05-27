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

        return [
            FunctionDefinition(**item)
            for item in data
        ]

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in function definitions: {exc}"
        ) from exc

    except ValidationError as exc:
        raise ValueError(
            f"Invalid function definition schema: {exc}"
        ) from exc


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

        return [
            PromptInput(**item)
            for item in data
        ]

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in prompts file: {exc}"
        ) from exc

    except ValidationError as exc:
        raise ValueError(
            f"Invalid prompt schema: {exc}"
        ) from exc


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
