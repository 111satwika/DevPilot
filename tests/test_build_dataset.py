"""Tests for ml/data/build_dataset.py's train/test split logic --
deliberately not-random, at the tool-family level. Uses fabricated
JSONL fixtures directly rather than the real generators, so the split
logic is tested in isolation from what any generator happens to produce.
"""

import json

from ml.data.build_dataset import build_dataset


def _write(path, examples):
    path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")


def _example(tool_family, i):
    return {"user_request": f"{tool_family} request {i}", "tool_family_holdout": tool_family}


def test_held_out_families_never_appear_in_train(tmp_path):
    source = tmp_path / "source.jsonl"
    examples = [_example("held_out_tool", i) for i in range(5)] + [_example("normal_tool", i) for i in range(5)]
    _write(source, examples)

    train, test, stats = build_dataset([source], held_out_families={"held_out_tool"})

    assert all(e["tool_family_holdout"] != "held_out_tool" for e in train)
    assert all(e["tool_family_holdout"] == "held_out_tool" for e in test if e["tool_family_holdout"] == "held_out_tool")
    # All 5 held-out examples must land in test, none in train.
    held_out_in_test = [e for e in test if e["tool_family_holdout"] == "held_out_tool"]
    assert len(held_out_in_test) == 5


def test_non_held_out_family_gets_a_stable_tail_split(tmp_path):
    source = tmp_path / "source.jsonl"
    examples = [_example("normal_tool", i) for i in range(10)]
    _write(source, examples)

    train, test, _ = build_dataset([source], held_out_families=set())

    assert len(train) == 8  # first 80%
    assert len(test) == 2  # last 20%, not shuffled
    assert train == examples[:8]
    assert test == examples[8:]


def test_split_is_deterministic_across_runs(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_example("t", i) for i in range(20)])

    train_1, test_1, _ = build_dataset([source], held_out_families=set())
    train_2, test_2, _ = build_dataset([source], held_out_families=set())

    assert train_1 == train_2
    assert test_1 == test_2


def test_combines_multiple_source_files(tmp_path):
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    _write(source_a, [_example("tool_a", 0)])
    _write(source_b, [_example("tool_b", 0)])

    train, test, stats = build_dataset([source_a, source_b], held_out_families=set())

    assert len(train) + len(test) == 2
    assert set(stats["per_family_counts"]) == {"tool_a", "tool_b"}


def test_missing_source_files_are_silently_skipped(tmp_path):
    real_source = tmp_path / "real.jsonl"
    _write(real_source, [_example("tool_a", 0)])
    missing = tmp_path / "does_not_exist.jsonl"

    train, test, _ = build_dataset([real_source, missing], held_out_families=set())
    assert len(train) + len(test) == 1
