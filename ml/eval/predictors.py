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


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def make_hf_adapter_predictor(
    base_model: str, adapter_path: str, max_new_tokens: int = 256
):  # pragma: no cover -- needs a real GPU, see module docstring
    """The fine-tuned model's predictor. Loads the 4-bit base model plus
    the trained LoRA adapter, generates a completion for each example
    using the exact same chat-template format ml/train/format_example.py
    trained on, and parses any <tool_call>{...}</tool_call> blocks out of
    the generated text -- the same Hermes-style format confirmed live
    against the real tokenizer (see format_example.py's module
    docstring)."""
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

    def predict(example: dict) -> list[dict]:
        messages = _build_prompt_messages(example)
        prompt_text = tokenizer.apply_chat_template(
            messages, tools=example.get("tools") or None, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        calls = []
        for match in _TOOL_CALL_BLOCK_RE.finditer(generated_text):
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                calls.append({"name": parsed["name"], "arguments": parsed["arguments"]})
        return calls

    return predict
