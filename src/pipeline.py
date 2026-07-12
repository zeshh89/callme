from src.prompt_builder import build_parameter_prompt
from src.json_decoder import generate_json
from src.models import FunctionCallResult
from src.decoding import greedy_decode
from src.function_prompt_builder import (
    build_function_prompt,
)
from src.value_decoder import generate_number, generate_boolean, generate_string
from src.prompt_builder import build_value_prompt


def generate_parameters(llm, user_prompt, function) -> dict:
    parameters = {}
    for name, param in function.parameters.items():
        value_prompt = build_value_prompt(
            user_prompt, function, name, param, parameters
        )
        if param.type in ("number", "integer"):
            parameters[name] = generate_number(llm, value_prompt, as_integer=(param.type == "integer"))
        elif param.type == "boolean":
            parameters[name] = generate_boolean(llm, value_prompt)
        else:
            parameters[name] = generate_string(llm, value_prompt)
    return parameters


def run_pipeline(
    prompt_obj,
    llm,
    trie,
    registry,
):
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

    # Generar parámetros
    parameter_prompt = build_parameter_prompt(
        prompt,
        function,
    )

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