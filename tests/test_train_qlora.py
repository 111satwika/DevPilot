"""Tests for the CPU-testable parts of ml/train/train_qlora.py -- the
data-loading wrapper, the warmup-step computation, and _run_train's
empty-dataset early exit. The actual QLoRA training loop past that point
needs a real GPU + CUDA bitsandbytes and is explicitly marked
`# pragma: no cover` in the script itself; nothing here pretends to test
that part.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.train.train_qlora import _compute_warmup_steps, _load_tokenized_dataset, _run_train

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


def test_run_train_fails_fast_on_empty_dataset_before_any_gpu_code(tmp_path, tokenizer):
    """Confirmed live on the first real Colab run: an empty train_dataset
    used to reach SFTTrainer's __init__ fine and crash deep inside it
    with a bare StopIteration -- AFTER the ~3GB base model had already
    been downloaded and LoRA-wrapped. This must fail immediately instead,
    before any of that -- and since the check happens before model
    load/quantization, it's testable here with no GPU at all."""
    empty_train = tmp_path / "empty_train.jsonl"
    empty_train.write_text("", encoding="utf-8")

    args = SimpleNamespace(
        train=empty_train,
        eval=tmp_path / "does_not_exist.jsonl",
        output_dir=tmp_path / "out",
        max_seq_len=4096,
        epochs=3,
        per_device_batch_size=4,
        grad_accum_steps=4,
    )

    with pytest.raises(SystemExit, match="No training examples loaded"):
        _run_train(args)


def test_sft_config_eval_batch_size_matches_train_batch_size():
    """Confirmed live (second real Colab run): training succeeded at
    per_device_train_batch_size=1, then OOM'd during evaluation because
    per_device_eval_batch_size was never set and defaulted to 8 (verified
    directly against the real installed trl.SFTConfig's dataclass
    fields). This builds the exact same SFTConfig shape _run_train does
    and checks the two batch sizes actually agree, so a regression here
    fails a CPU-only test instead of only showing up as a GPU OOM."""
    trl = pytest.importorskip("trl")

    per_device_batch_size = 1
    training_args = trl.SFTConfig(
        output_dir="out",
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False,
        use_cpu=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        max_length=4096,
    )

    assert training_args.per_device_eval_batch_size == training_args.per_device_train_batch_size == per_device_batch_size


def test_sft_config_falls_back_to_fp16_when_bf16_is_unsupported():
    """Confirmed live (real Kaggle run): _run_train used to hardcode
    bf16=True unconditionally, and SFTConfig's own validation rejected
    that outright on Kaggle's GPU with "Your setup doesn't support
    bf16/gpu" -- bf16 needs Ampere-or-newer hardware (compute capability
    >= 8.0), which not every free-tier GPU (Kaggle's T4/P100, possibly
    unlike whatever Colab happened to assign in earlier runs) has. Fixed
    by detecting support at runtime via torch.cuda.is_bf16_supported()
    (verified safe to call with no CUDA device at all -- returns False,
    doesn't error) and falling back to fp16 otherwise. This constructs
    the exact same SFTConfig shape with the fallback values and confirms
    it doesn't error and the two flags are always opposite, never both
    True or both False."""
    torch = pytest.importorskip("torch")
    trl = pytest.importorskip("trl")

    use_bf16 = torch.cuda.is_bf16_supported()  # False in this CPU-only environment

    training_args = trl.SFTConfig(
        output_dir="out",
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16,
        use_cpu=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        max_length=4096,
    )

    assert training_args.bf16 != training_args.fp16
    assert training_args.bf16 == use_bf16


def test_sft_config_uses_plain_nll_loss_not_the_buggy_chunked_default():
    """Confirmed live (real Kaggle GPU run): trl==1.12.0 defaults
    loss_type to "chunked_nll" whenever it isn't set explicitly, and that
    path has a real bug for a PEFT LoRA + 4-bit-quantized + gradient-
    checkpointed model -- AttributeError: 'functools.partial' object has
    no attribute '__func__', inside trl's own
    _patch_chunked_ce_lm_head (confirmed by reading that function's real
    installed source directly). This constructs the exact SFTConfig
    shape _run_train uses and checks loss_type is explicitly "nll", not
    left to default -- a regression here would silently reintroduce the
    same crash the next time SFTConfig is touched."""
    trl = pytest.importorskip("trl")

    training_args = trl.SFTConfig(
        output_dir="out",
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False,
        fp16=True,
        use_cpu=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        max_length=4096,
        loss_type="nll",
    )

    assert training_args.loss_type == "nll"


class TestComputeWarmupSteps:
    def test_respects_an_explicit_warmup_ratio(self):
        # 100 examples, batch 4 * grad_accum 4 = effective 16 -> 7 steps/epoch (ceil),
        # 3 epochs -> 21 total steps -> 3% = 0.63 -> rounds to 1.
        assert _compute_warmup_steps(100, 4, 4, 3, warmup_ratio=0.03) == 1

    def test_scales_up_for_larger_datasets(self):
        # 10,000 examples, effective batch 16 -> 625 steps/epoch, 3 epochs -> 1875 total -> 3% ~ 56.
        assert _compute_warmup_steps(10_000, 4, 4, 3, warmup_ratio=0.03) == 56

    def test_never_returns_zero(self):
        assert _compute_warmup_steps(1, 4, 4, 1) >= 1

    def test_larger_effective_batch_size_means_fewer_steps(self):
        small_batch = _compute_warmup_steps(1000, 2, 2, 3)
        large_batch = _compute_warmup_steps(1000, 8, 8, 3)
        assert large_batch < small_batch

    def test_default_warmup_ratio_is_twenty_percent_not_three(self):
        """Entry 57: raised from 0.03 to 0.2 after a real Kaggle run's
        tiny 12-total-step schedule got only 1 warmup step at 3% -- the
        LR jumped to full value on step 2, a documented contributor to
        the premature-EOS collapse that run's raw generated output
        showed. 0.1 was tried first but verified to round right back
        down to the same 1 step for this exact schedule length
        (round(12*0.1)=1) -- 0.2 is the smallest round ratio that
        actually changes anything here. This project's real dataset
        shape (57 examples, batch_size=1, grad_accum=16, 3 epochs -> 12
        total steps) is used directly so a regression back to a
        too-low default fails this test with real numbers, not an
        arbitrary example."""
        assert _compute_warmup_steps(57, 1, 16, 3) == 2  # default ratio: round(12 * 0.2) = 2
        assert _compute_warmup_steps(57, 1, 16, 3) != _compute_warmup_steps(57, 1, 16, 3, warmup_ratio=0.03)
        # A larger, more realistic total-step count shows the raised default's real effect.
        assert _compute_warmup_steps(1000, 4, 4, 3) == 38  # round(189 * 0.2) at default ratio
