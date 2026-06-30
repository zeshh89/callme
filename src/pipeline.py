from src.masking import mask_logits
from src.prompt_builder import build_parameter_prompt
from src.json_decoder import generate_json
from src.models import FunctionCallResult
from src.registry import FunctionRegistry
from src.function_prompt_builder import (
    build_function_prompt,
)


def greedy_decode(prompt, llm, trie):

    prompt_ids = llm.encode(prompt)

    generated = []

    for _ in range(20):

        logits = llm.logits(
            prompt_ids + generated
        )

        allowed = trie.get_allowed_tokens(
            generated
        )

        if not allowed:
            break

        masked = mask_logits(
            logits,
            allowed,
        )

        token = int(
            masked.argmax().item()
        )

        generated.append(token)

        if trie.is_complete(generated):
            break

    return generated


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

    parameters = generate_json(
        llm,
        parameter_prompt,
    )

    return FunctionCallResult(
        prompt=prompt,
        name=function.name,
        parameters=parameters,
    )