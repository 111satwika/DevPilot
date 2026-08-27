"""Tests for the CPU-testable parts of ml/train/train_qlora.py -- the
data-loading wrapper and the warmup-step computation. The actual QLoRA
training loop (_run_train) needs a real GPU + CUDA bitsandbytes and is
explicitly marked `# pragma: no cover` in the script itself; nothing here
pretends to test that part.
"""

import json

import pytest

from ml.train.train_qlora import _compute_warmup_steps, _load_tokenized_dataset

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    except Exception as exc:  # noqa: BLE001 -- no network / HF unreachable
        pytest.skip(f"Could not load tokenizer (no network?): {exc}")


def _write_example(path, mode="agent", request="hi", tools=None):
    example = {
        "mode": mode,
        "system_prompt": "You are DevPilot.",
        "tools": tools or [],
        "messages": [],
        "user_request": request,
        "completion": {"type": "text", "tool_calls": [], "text": "ok"},
        "source": "real_trace",
        "tool_family_holdout": None,
    }
    path.write_text(json.dumps(example), encoding="utf-8")


def test_load_tokenized_dataset_returns_correct_records(tmp_path, tokenizer):
    path = tmp_path / "data.jsonl"
    _write_example(path)

    records = _load_tokenized_dataset(path, tokenizer, max_seq_len=4096)

    assert len(records) == 1
    assert set(records[0].keys()) == {"input_ids", "labels", "attention_mask"}
    assert len(records[0]["input_ids"]) == len(records[0]["labels"]) == len(records[0]["attention_mask"])


def test_load_tokenized_dataset_drops_oversized_examples(tmp_path, tokenizer):
    path = tmp_path / "data.jsonl"
    _write_example(path, request="x " * 5000)

    records = _load_tokenized_dataset(path, tokenizer, max_seq_len=64)

    assert records == []


class TestComputeWarmupSteps:
    def test_is_roughly_three_percent_of_total_steps(self):
        # 100 examples, batch 4 * grad_accum 4 = effective 16 -> 7 steps/epoch (ceil),
        # 3 epochs -> 21 total steps -> 3% = 0.63 -> rounds to 1.
        assert _compute_warmup_steps(100, 4, 4, 3) == 1

    def test_scales_up_for_larger_datasets(self):
        # 10,000 examples, effective batch 16 -> 625 steps/epoch, 3 epochs -> 1875 total -> 3% ~ 56.
        assert _compute_warmup_steps(10_000, 4, 4, 3) == 56

    def test_never_returns_zero(self):
        assert _compute_warmup_steps(1, 4, 4, 1) >= 1

    def test_larger_effective_batch_size_means_fewer_steps(self):
        small_batch = _compute_warmup_steps(1000, 2, 2, 3)
        large_batch = _compute_warmup_steps(1000, 8, 8, 3)
        assert large_batch < small_batch
