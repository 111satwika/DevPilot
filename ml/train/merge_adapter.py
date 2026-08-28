"""Phase 4: merges a trained LoRA adapter into the base model, producing
a standalone HF checkpoint that can be converted to GGUF and served
through Ollama on the same hardware as the (un-fine-tuned) base model --
the fair, same-hardware latency comparison Concept #45 flagged as
missing (the original comparison mixed this machine's CPU against a
Colab/Kaggle GPU).

Loads the base model in fp16 (NOT the 4-bit quantized form training
used) -- merging LoRA deltas into a 4-bit-quantized base isn't a
meaningful operation (the delta was learned against dequantized
values during the forward pass; PEFT's own merge_and_unload() expects
a normal-precision base). CPU-only is fine here: merging is a handful
of matrix additions over ~1.5B params, not a training loop.

Usage:
    python -m ml.train.merge_adapter --adapter path/to/qlora-adapter --out ml/train/out/merged
"""

import argparse
from pathlib import Path


def main() -> None:  # pragma: no cover -- loads real multi-GB model weights
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ml/train/out/merged"))
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {args.base_model} in fp16 (CPU) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)

    print(f"Attaching adapter from {args.adapter} ...")
    model = PeftModel.from_pretrained(model, str(args.adapter))

    print("Merging LoRA weights into the base model ...")
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.out} ...")
    model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    print("Done.")


if __name__ == "__main__":
    main()
