"""Combines every data source (real traces, adversarial, template-
generated positive examples) into a final train/test split --
deliberately NOT a random per-example split.

Two things a random split would get wrong for this dataset specifically:

1. Template-generated examples (generate_adversarial.py,
   generate_positive_examples.py) produce near-duplicate phrasings for
   the same tool right next to each other. A random split can put
   near-identical examples on both sides of the train/test boundary,
   which inflates test accuracy without the model having learned
   anything general -- the exact failure mode the spec's own methodology
   note warns about.

2. Testing whether the model generalizes to a genuinely UNSEEN tool
   (schema-only generalization, one of Phase 2's stated goals) requires
   some tool families to never appear in training at all -- a random
   per-example split can never produce that, since it doesn't hold
   anything out at the group level.

So the split works in two stages: (a) entirely hold out a fixed set of
tool families from training -- HELD_OUT_TOOL_FAMILIES below -- so those
only ever appear in the test set; (b) for every other tool family, take a
stable (not shuffled) tail slice of that family's own examples as test,
so near-duplicates generated adjacently stay on the same side.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from ml.data.schema import read_jsonl

# Tool families never seen during training -- present only in the test
# set, to measure whether the fine-tune generalizes to an unseen tool
# from its schema alone (Phase 2's stated goal). Chosen to be a small,
# genuinely representative slice: one read-only tool, one gated/mutating
# tool, one from a server not otherwise dominant in the dataset.
HELD_OUT_TOOL_FAMILIES = {"list_pull_requests", "stop_container", "git_diff"}

TEST_FRACTION_PER_FAMILY = 0.2


def build_dataset(
    sources: list[Path], held_out_families: set[str] = HELD_OUT_TOOL_FAMILIES
) -> tuple[list[dict], list[dict], dict]:
    all_examples: list[dict] = []
    for path in sources:
        all_examples.extend(read_jsonl(path))

    by_family: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        family = ex.get("tool_family_holdout") or "(none)"
        by_family[family].append(ex)

    train: list[dict] = []
    test: list[dict] = []
    stats = {"held_out_families": sorted(held_out_families), "per_family_counts": {}}

    for family, examples in by_family.items():
        stats["per_family_counts"][family] = len(examples)
        if family in held_out_families:
            test.extend(examples)
            continue

        split_index = max(1, int(len(examples) * (1 - TEST_FRACTION_PER_FAMILY))) if len(examples) > 1 else len(examples)
        train.extend(examples[:split_index])
        test.extend(examples[split_index:])

    return train, test, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", type=Path,
        default=[
            Path("ml/data/out/real_traces.jsonl"),
            Path("ml/data/out/adversarial.jsonl"),
            # Entry 51's fix: the dataset's only tool_call examples used
            # to come entirely from adversarial.jsonl's explore-first
            # bucket (all list_directory) -- this generator adds 35 more,
            # across 21 different tools, to correct the ~4.5:1 class
            # imbalance implicated in the fine-tuned model's
            # exact_tool_match regression.
            Path("ml/data/out/positive.jsonl"),
        ],
    )
    parser.add_argument("--train-out", type=Path, default=Path("ml/data/out/train.jsonl"))
    parser.add_argument("--test-out", type=Path, default=Path("ml/data/out/test.jsonl"))
    args = parser.parse_args()

    existing_sources = [p for p in args.sources if p.is_file()]
    missing_sources = [p for p in args.sources if not p.is_file()]

    train, test, stats = build_dataset(existing_sources)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with args.train_out.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with args.test_out.open("w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    print(f"Sources used: {[str(p) for p in existing_sources]}")
    if missing_sources:
        print(f"Sources not found (skipped): {[str(p) for p in missing_sources]}")
    print(f"Train: {len(train)} examples -> {args.train_out}")
    print(f"Test:  {len(test)} examples -> {args.test_out}")
    print(f"Held-out tool families (test-only): {stats['held_out_families']}")
    print("Per-family counts:")
    for family, count in sorted(stats["per_family_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
