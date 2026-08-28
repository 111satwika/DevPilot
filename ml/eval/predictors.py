"""Model predictors for ml/eval/harness.py -- each is a callable
`(example: dict) -> list[dict]` returning the tool calls a model produced
(empty list for a refusal/text response).

make_ollama_predictor is real and runnable right now: Ollama is
confirmed reachable natively at 127.0.0.1:11434 in this environment
(qwen2.5:7b-instruct loaded, "tools" capability present) -- called via a
direct HTTP request, deliberately NOT through llm/agent.py's
_ollama_chat(), which still bridges through a WSL "Ubuntu" distro that
doesn't exist in this environment (a real, separately-known, still-open
bug -- see the Ollama connectivity discussion earlier in this project's
history). The eval harness has no reason to inherit that bug.

make_hf_adapter_predictor is for the fine-tuned model and needs a real
GPU (loads the 4-bit base model + the trained LoRA adapter) -- written
correctly per the same chat-template/tool-call format verified for real
against the tokenizer in ml/train/format_example.py, but not executable
in this environment. Run it on the same Colab session that trained the
adapter.
"""

import json
import re

import httpx2

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
# Confirmed live (first real eval run): 120s wasn't enough and timed out
# on the first call. This model runs CPU-only on this hardware (Entry
# 34's finding, llm/agent.py's own REQUEST_TIMEOUT_SECONDS) -- matching
# that project-wide, empirically-tuned value here rather than guessing a
# smaller one again.
DEFAULT_TIMEOUT_SECONDS = 600


def _build_prompt_messages(example: dict) -> list[dict]:
    messages = [{"role": "system", "content": example["system_prompt"]}]
    messages.extend(example.get("messages", []))
    messages.append({"role": "user", "content": example["user_request"]})
    return messages


def make_ollama_predictor(
    model_name: str,
    host: str = OLLAMA_HOST,
    port: int = OLLAMA_PORT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
):
    """Real, runnable now -- direct HTTP call to a live Ollama instance."""

    def predict(example: dict) -> list[dict]:
        messages = _build_prompt_messages(example)
        with httpx2.Client(timeout=timeout) as client:
            response = client.post(
                f"http://{host}:{port}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "tools": example.get("tools") or [],
                    "stream": False,
                },
            )
        response.raise_for_status()
        message = response.json()["message"]
        tool_calls = message.get("tool_calls") or []
        return [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in tool_calls
        ]

    return predict


_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _extract_first_json_object(text: str, start: int = 0) -> dict | None:
    """Finds the first '{' at or after `start`, then its matching '}' via
    brace counting (correct for a nested arguments dict, unlike a naive
    non-greedy regex like `\\{.*?\\}`, which stops at the FIRST '}' it
    sees and would truncate e.g. {"name": "x", "arguments": {"a": 1}} at
    the inner brace). Returns the parsed dict, or None if no span
    starting with '{' parses as valid JSON."""
    idx = text.find("{", start)
    while idx != -1:
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except json.JSONDecodeError:
                        break  # this '{' doesn't start valid JSON -- try the next one
        idx = text.find("{", idx + 1)
    return None


def _load_adapter_model(base_model: str, adapter_path: str):  # pragma: no cover -- needs a real GPU
    """Loads the 4-bit base model plus the trained LoRA adapter. Split out
    of make_hf_adapter_predictor so ml/eval/debug_predictions.py can share
    the exact same loading path when a MISS needs to be inspected by hand
    (raw generated text, not just the parsed tool calls)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def _generate_raw(tokenizer, model, example: dict, max_new_tokens: int) -> str:  # pragma: no cover -- needs a real GPU
    """Runs the exact same chat-template format ml/train/format_example.py
    trained on and returns the model's raw decoded completion, before any
    <tool_call> parsing."""
    import torch

    messages = _build_prompt_messages(example)
    prompt_text = tokenizer.apply_chat_template(
        messages, tools=example.get("tools") or None, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_tool_call_blocks(generated_text: str) -> list[dict]:
    """Parses tool calls out of raw generated text. Primary format:
    Hermes-style <tool_call>{...}</tool_call> blocks -- the format
    confirmed live against the real tokenizer (see format_example.py's
    module docstring) and what the training data actually uses.

    Confirmed live (Entry 58, real Kaggle adapter after the Entry 57
    learning-rate/warmup fix): the model sometimes emits the exact
    correct {"name": ..., "arguments": ...} JSON WITHOUT the <tool_call>
    wrapper tags at all (e.g. '{"name": "git_log", "arguments": {"limit":
    10}}' with no tags), which the tag-only parser silently discarded as
    zero tool calls -- scoring a genuinely correct prediction as a total
    miss. Falls back to the first bare JSON object with "name"/
    "arguments" keys when no tagged block is found. Only the FIRST such
    object is taken in the fallback case: this project's real design is
    one tool call per turn (llm/agent.py's iterative loop), so a model
    that repeats several (sometimes hallucinated) calls back-to-back --
    also confirmed live in that same investigation -- is a distinct
    generation-quality issue this parser should surface as "predicted the
    wrong number of calls" (via exact_tool_match's own len(...) != 1
    check on the tag-block path), not paper over by picking one of many.

    Exposed at module level (not nested in make_hf_adapter_predictor) so
    debug_predictions.py can parse the same raw text it prints, with the
    real parsing logic, not a re-implementation of it."""
    calls = []
    for match in _TOOL_CALL_TAG_RE.finditer(generated_text):
        parsed = _extract_first_json_object(match.group(1))
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            calls.append({"name": parsed["name"], "arguments": parsed["arguments"]})
    if calls:
        return calls

    parsed = _extract_first_json_object(generated_text)
    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
        return [{"name": parsed["name"], "arguments": parsed["arguments"]}]
    return []


def make_hf_adapter_predictor(
    base_model: str, adapter_path: str, max_new_tokens: int = 100
):  # pragma: no cover -- needs a real GPU, see module docstring
    """The fine-tuned model's predictor. See _load_adapter_model /
    _generate_raw / parse_tool_call_blocks above for the pieces this
    composes.

    max_new_tokens default lowered 256 -> 100 (Entry 58): measured
    directly against every real completion in this project's dataset
    (train+test), the longest is 49 tokens, mean ~35 -- 100 is a
    generous margin over any real completion while cutting off, much
    earlier than before, the runaway repeated-tool-call generation a
    real Kaggle adapter was confirmed to sometimes produce (multiple
    back-to-back JSON objects, occasionally naming hallucinated tools,
    running until max_new_tokens was hit without ever emitting an
    end-of-turn token)."""
    tokenizer, model = _load_adapter_model(base_model, adapter_path)

    def predict(example: dict) -> list[dict]:
        generated_text = _generate_raw(tokenizer, model, example, max_new_tokens)
        return parse_tool_call_blocks(generated_text)

    return predict
