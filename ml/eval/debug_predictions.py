"""Diagnostic tool, not part of the regular eval pipeline: prints the
fine-tuned model's RAW generated text (before <tool_call> parsing) for a
handful of tool_call-expected examples, alongside what was expected.

Written because run_eval.py's report only stores the derived booleans
(exact_match, schema_valid, ...), not the raw prediction -- enough to see
THAT an example missed, not WHY. When exact_tool_match_rate=0% and
schema_valid_rate=100% and mean_argument_f1 is n/a at the same time, that
combination is consistent with the model predicting zero tool calls on
every example (is_schema_valid is vacuously true for an empty
prediction), but it's also consistent with other failure shapes -- this
prints the actual text so that doesn't have to stay a guess.

Usage (Colab, same session the adapter was trained/evaluated in):
    !python -m ml.eval.debug_predictions --limit 3
"""

import argparse
from pathlib import Path

from ml.data.schema import read_jsonl


def main() -> None:  # pragma: no cover -- needs a real GPU, see module docstring
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("ml/train/out/qlora-adapter"))
    parser.add_argument("--examples", type=Path, default=Path("ml/data/out/test.jsonl"))
    parser.add_argument("--limit", type=int, default=3, help="how many tool_call-expected examples to inspect")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch

    from ml.eval.predictors import _build_prompt_messages, _load_adapter_model, parse_tool_call_blocks

    examples = [ex for ex in read_jsonl(args.examples) if ex["completion"]["type"] == "tool_call"]
    examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No tool_call-expected examples found in {args.examples}")

    print(f"Loading {args.base_model} + adapter {args.adapter} ...")
    tokenizer, model = _load_adapter_model(str(args.base_model), str(args.adapter))

    for i, ex in enumerate(examples, 1):
        expected = ex["completion"]["tool_calls"][0]

        messages = _build_prompt_messages(ex)
        prompt_text = tokenizer.apply_chat_template(
            messages, tools=ex.get("tools") or None, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        new_token_ids = output_ids[0][inputs["input_ids"].shape[1] :].tolist()
        # skip_special_tokens=True is what the real predictor decodes with
        # (and what looked empty before) -- also show the special-tokens-
        # kept version and the raw token count, so "generated literally
        # nothing" and "generated only special tokens, invisibly stripped"
        # can be told apart.
        text_stripped = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        text_with_specials = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        parsed = parse_tool_call_blocks(text_stripped)

        print(f"\n=== [{i}/{len(examples)}] mode={ex['mode']} ===")
        print(f"request:  {ex['user_request']!r}")
        print(f"expected: {expected['name']}({expected['arguments']})")
        print(f"parsed predicted tool calls: {parsed}")
        print(f"prompt token count: {inputs['input_ids'].shape[1]}")
        print(f"new tokens generated: {len(new_token_ids)}  ids: {new_token_ids[:20]}")
        print(f"raw generated text (skip_special_tokens=True):  {text_stripped!r}")
        print(f"raw generated text (skip_special_tokens=False): {text_with_specials!r}")


if __name__ == "__main__":
    main()
