from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    ParameterValue,
    PromptInput
)
from src.registry import FunctionRegistry
from src.function_trie import FunctionTrie
from src.decoding import greedy_decode
from src.function_prompt_builder import (
    build_function_prompt,
)
from src.value_decoder import (
    generate_number,
    generate_boolean,
    generate_string
)
from src.prompt_builder import build_value_prompt
from src.llm import LLM


def generate_parameters(
    llm: LLM,
    user_prompt: str,
    function: FunctionDefinition
) -> dict:
    parameters: dict[str, ParameterValue] = {}
    for name, param in function.parameters.items():
        value_prompt = build_value_prompt(
            user_prompt, function, name, param, parameters
        )
        if param.type in ("number", "integer"):
            parameters[name] = generate_number(
                llm,
                value_prompt,
                as_integer=(param.type == "integer")
            )
        elif param.type == "boolean":
            parameters[name] = generate_boolean(llm, value_prompt)
        else:
            parameters[name] = generate_string(llm, value_prompt)
    return parameters


def run_pipeline(
    prompt_obj: PromptInput,
    llm: LLM,
    trie: FunctionTrie,
    registry: FunctionRegistry,
) -> FunctionCallResult:
    prompt = prompt_obj.prompt

    # Seleccionar función
    function_prompt = build_function_prompt(
        prompt,
        registry.functions(),
    )

    function_tokens = greedy_decode(
        function_prompt,
        llm,
        trie,
    )

    function_name = llm.decode(function_tokens)

    function = registry.get(function_name)

    if function is None:
        return FunctionCallResult(
            prompt=prompt,
            name="UNKNOWN",
            parameters={},
        )

    print(f"\nPredicted function: {function.name}")

    parameters = generate_parameters(
        llm,
        prompt,
        function
    )

    return FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=parameters,
    )
