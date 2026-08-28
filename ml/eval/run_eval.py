"""CLI: run the eval harness against the held-out test set with a chosen
predictor, print the summary line, and write a full JSON report.

Usage:
    # Base model, live against Ollama (runnable right now):
    python -m ml.eval.run_eval --model ollama --model-name qwen2.5:7b-instruct

    # Fine-tuned model, needs a real GPU (run on the training Colab session):
    python -m ml.eval.run_eval --model hf-adapter \
        --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter-path ml/train/out/qlora-adapter
"""

import argparse
import json
from pathlib import Path

from ml.data.schema import read_jsonl
from ml.eval.harness import evaluate
from ml.eval.predictors import make_hf_adapter_predictor, make_ollama_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=Path("ml/data/out/test.jsonl"))
    parser.add_argument("--model", choices=["ollama", "hf-adapter"], required=True)
    parser.add_argument("--model-name", default="qwen2.5:7b-instruct", help="Ollama model tag (--model ollama)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="--model hf-adapter")
    parser.add_argument("--adapter-path", type=Path, help="--model hf-adapter")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    parser.add_argument("--out", type=Path, default=None, help="write the full JSON report here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    examples = read_jsonl(args.test)
    if not examples:
        raise SystemExit(f"No examples found in {args.test} -- run ml.data.build_dataset first.")
    if args.limit:
        examples = examples[: args.limit]

    if args.model == "ollama":
        predictor = make_ollama_predictor(args.model_name)
        report_name = args.model_name
    else:
        if not args.adapter_path:
            raise SystemExit("--adapter-path is required for --model hf-adapter")
        predictor = make_hf_adapter_predictor(args.base_model, str(args.adapter_path))
        report_name = f"{args.base_model} + {args.adapter_path}"

    print(f"Evaluating {report_name} on {len(examples)} held-out examples...")
    report = evaluate(report_name, predictor, examples, verbose=not args.quiet)

    print("\n" + report.summary_line())

    out_path = args.out or Path(f"ml/eval/out/{args.model}_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
