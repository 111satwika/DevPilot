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

    from ml.eval.predictors import _generate_raw, _load_adapter_model, parse_tool_call_blocks

    examples = [ex for ex in read_jsonl(args.examples) if ex["completion"]["type"] == "tool_call"]
    examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No tool_call-expected examples found in {args.examples}")

    print(f"Loading {args.base_model} + adapter {args.adapter} ...")
    tokenizer, model = _load_adapter_model(str(args.base_model), str(args.adapter))

    for i, ex in enumerate(examples, 1):
        expected = ex["completion"]["tool_calls"][0]
        generated_text = _generate_raw(tokenizer, model, ex, args.max_new_tokens)
        parsed = parse_tool_call_blocks(generated_text)

        print(f"\n=== [{i}/{len(examples)}] mode={ex['mode']} ===")
        print(f"request:  {ex['user_request']!r}")
        print(f"expected: {expected['name']}({expected['arguments']})")
        print(f"parsed predicted tool calls: {parsed}")
        print(f"raw generated text:\n{generated_text!r}")


if __name__ == "__main__":
    main()
