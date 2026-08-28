"""QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on DevPilot's own real
tool-calling dataset (ml/data/build_dataset.py's output).

honesty note, read before trusting this end-to-end: this script was
written and structurally checked in an environment confirmed to have NO
GPU (torch.cuda.is_available() -> False) and no CUDA-enabled
bitsandbytes -- it cannot be run to completion here. What IS verified,
against the real base-model tokenizer, is everything data-side:
ml/train/format_example.py's tokenization and completion-only masking
(tests/test_format_example.py, 4 tests, real tokenizer download). The
PEFT/bitsandbytes/TRL wiring below follows each library's documented API
correctly but has not itself been executed against a live GPU. Run it on
a free Colab/Kaggle T4 and treat the first real run as the actual first
test of this part, not as "already proven" -- report back what actually
happens, good or bad, same discipline as every other entry in this
project's log.

Usage (Colab/Kaggle, T4 GPU runtime):
    !pip install transformers peft trl bitsandbytes accelerate datasets
    !python -m ml.train.train_qlora --train ml/data/out/train.jsonl --eval ml/data/out/test.jsonl

Use --dry-run to sanity-check data loading/tokenization on a CPU-only
machine (this one, right now) without ever touching the GPU-only parts.
"""

import argparse
import json
from pathlib import Path

from ml.data.schema import read_jsonl
from ml.train.format_example import format_example

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# Measured against the real dataset, not the original plan's 2048 --
# see Learning Log Concept #40 / Entry 48.
MAX_SEQ_LEN = 4096

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def _load_tokenized_dataset(path: Path, tokenizer, max_seq_len: int):
    """Returns a list of {input_ids, labels, attention_mask} dicts --
    plain Python here so --dry-run needs no `datasets` install; wrapped
    into a real datasets.Dataset only in the GPU path below."""
    examples = read_jsonl(path)
    records = []
    dropped = 0
    for ex in examples:
        tokenized = format_example(ex, tokenizer, max_seq_len=max_seq_len)
        if tokenized is None:
            dropped += 1
            continue
        records.append(
            {
                "input_ids": tokenized.input_ids,
                "labels": tokenized.labels,
                "attention_mask": tokenized.attention_mask,
            }
        )
    if dropped:
        print(f"  dropped {dropped}/{len(examples)} example(s) from {path} (exceeded max_seq_len={max_seq_len})")
    return records


def _compute_warmup_steps(num_examples: int, per_device_batch_size: int, grad_accum_steps: int, epochs: int) -> int:
    """3% warmup, expressed as steps -- SFTConfig (trl==1.12.0, verified
    by constructing one directly) only accepts warmup_STEPS, not
    warmup_ratio, so the step count has to be derived from dataset size
    here instead."""
    effective_batch_size = per_device_batch_size * grad_accum_steps
    steps_per_epoch = max(1, -(-num_examples // effective_batch_size))  # ceil div
    total_steps = steps_per_epoch * epochs
    return max(1, round(total_steps * 0.03))


def _run_dry(args) -> None:
    """CPU-only sanity check: load real data, tokenize it for real
    against the real base-model tokenizer, print stats, stop before
    anything that needs a GPU. This is the part of this script that
    genuinely has been run, in this environment, for real."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading + tokenizing {args.train} ...")
    train_records = _load_tokenized_dataset(args.train, tokenizer, args.max_seq_len)
    print(f"  {len(train_records)} usable training example(s)")

    eval_records = []
    if args.eval.is_file():
        print(f"Loading + tokenizing {args.eval} ...")
        eval_records = _load_tokenized_dataset(args.eval, tokenizer, args.max_seq_len)
        print(f"  {len(eval_records)} usable eval example(s)")

    if train_records:
        lengths = [len(r["input_ids"]) for r in train_records]
        print(f"Train sequence lengths: min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.0f}")

    print("\n--dry-run: stopping before model load / QLoRA / training (no GPU touched).")


def _run_train(args) -> None:  # pragma: no cover -- needs a real GPU + bitsandbytes CUDA build
    """The actual QLoRA training loop. Not executable in a CPU-only
    environment; see the module docstring."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
    )
    from trl import SFTConfig, SFTTrainer

    # Confirmed live (real Kaggle run): bf16 is an Ampere-or-newer hardware
    # feature (compute capability >= 8.0) -- unconditionally passing
    # bf16=True made SFTConfig's own validation reject the run outright on
    # Kaggle's GPU with "Your setup doesn't support bf16/gpu", well before
    # any training code ran. torch.cuda.is_bf16_supported() is the same
    # check transformers' own validation uses internally, confirmed by
    # constructing it directly in this CPU-only environment (returns False
    # safely, doesn't error, even with no CUDA device at all) -- so it's
    # used here to pick bf16 where the GPU actually supports it and fall
    # back to fp16 (with its own loss-scaling, handled by SFTConfig/
    # SFTTrainer automatically when fp16=True) everywhere else, rather
    # than assuming any given free-tier GPU (Colab's vs. Kaggle's may
    # differ) supports bf16.
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"bf16 supported on this GPU: {use_bf16} -- using compute dtype {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = _load_tokenized_dataset(args.train, tokenizer, args.max_seq_len)
    train_dataset = Dataset.from_list(train_records)

    eval_dataset = None
    if args.eval.is_file():
        eval_records = _load_tokenized_dataset(args.eval, tokenizer, args.max_seq_len)
        if eval_records:
            eval_dataset = Dataset.from_list(eval_records)

    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval examples: {len(eval_dataset)}")

    if len(train_dataset) == 0:
        # Confirmed live (first real Colab run): an empty train_dataset
        # reaches SFTTrainer's own __init__ fine, then crashes deep inside
        # it with a bare `StopIteration` from `next(iter(train_dataset))`
        # -- AFTER the base model has already been downloaded (~3GB) and
        # LoRA-wrapped. Failing here instead means an empty/missing
        # dataset (e.g. ml/data/build_dataset.py was never run in this
        # session, or ran in a Colab session that then disconnected and
        # wiped /content) costs a fast, clear error instead of minutes of
        # wasted download + a cryptic traceback.
        raise SystemExit(
            f"No training examples loaded from {args.train} -- did you run "
            f"ml.data.mine_real_traces / generate_adversarial / build_dataset "
            f"in this session? (Colab wipes /content on disconnect, so a "
            f"dataset built in an earlier session won't still be here.)"
        )

    warmup_steps = _compute_warmup_steps(
        len(train_dataset), args.per_device_batch_size, args.grad_accum_steps, args.epochs
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    # Confirmed live (first real Colab run): a batch of 4 real examples
    # OOM'd a T4 (16GB) on the very first backward pass -- 14.17/14.56 GB
    # already in use before the failed 2.64 GB allocation. Qwen's ~152k
    # vocabulary is the likely dominant cost here, not model weights (tiny
    # at 4-bit): the output logits tensor scales with
    # batch_size * seq_len * vocab_size, and real examples run up to
    # ~3,867 tokens (Concept #40) -- one batch of 4 long sequences can be
    # several GB just for logits. use_gradient_checkpointing explicit here
    # (not relying on this function's own default) plus
    # gradient_checkpointing_kwargs to avoid the reentrant-autograd/PEFT
    # interaction issue that setting persists.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        # Confirmed live (second real Colab run): training got through
        # 4/12 steps fine at batch_size=1, then OOM'd during the
        # end-of-epoch EVALUATION phase specifically (5.30 GiB alloc
        # failure inside compute_loss's forward pass). Root cause,
        # confirmed by constructing a real SFTConfig and inspecting its
        # dataclass fields directly: per_device_eval_batch_size defaults
        # to 8, independent of per_device_train_batch_size -- it was
        # never set here, so eval silently ran at 8x the batch size
        # training had already been cut down to. Same memory-scaling
        # argument as the training-side fix applies here too (batch_size
        # * seq_len * vocab_size), so it gets the same value.
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        # train_dataset already carries input_ids/labels/attention_mask
        # (ml/train/format_example.py). Confirmed directly against the
        # real installed trl==1.12.0 source
        # (site-packages/trl/trainer/sft_trainer.py, _prepare_dataset):
        # `is_processed = "input_ids" in column_names` -- a dataset with
        # that column is recognized as pre-tokenized and its own
        # text-formatting/re-tokenization step is skipped entirely, using
        # the existing "labels" column as-is. This is the one part of
        # this script that could otherwise have been silently wrong
        # (TRL's pre-tokenized-dataset handling has changed across
        # versions) -- if you're running a different TRL version on
        # Colab than 1.12.0, re-check this same source path before
        # trusting it.
        max_length=args.max_seq_len,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Adapter saved to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("ml/data/out/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("ml/data/out/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/train/out/qlora-adapter"))
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--epochs", type=int, default=3)
    # Defaults changed from the original plan's 4/4 after a real T4 OOM'd
    # on the very first backward pass at batch_size=4 (see the comment in
    # _run_train, near prepare_model_for_kbit_training). 1/16 keeps the
    # same effective batch size (16) while cutting the per-step memory
    # that scales with batch_size * seq_len * vocab_size by 4x.
    parser.add_argument("--per-device-batch-size", type=int, default=1, dest="per_device_batch_size")
    parser.add_argument("--grad-accum-steps", type=int, default=16, dest="grad_accum_steps")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load + tokenize real data against the real tokenizer and report stats; skip model load and training entirely (no GPU needed)",
    )
    args = parser.parse_args()

    if args.dry_run:
        _run_dry(args)
    else:
        _run_train(args)


if __name__ == "__main__":
    main()
