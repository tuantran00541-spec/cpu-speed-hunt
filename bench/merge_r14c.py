#!/usr/bin/env python3
"""Merge a LoRA adapter into the base Qwen3.5-0.8B model (runner version).

Same merge as qwen-sft-merge/merge_adapter.py on the VM, but all paths
are local CLI args: the base comes from the workflow's snapshot_download
(./hf-model) and the adapter from the r14c-fp16 release asset
(./r14c-adapter). No HF Hub access at merge time (offline flags set).
Writes the merged bf16 model to --out for convert_hf_to_gguf.py.
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="local base model dir")
    ap.add_argument("--adapter", required=True, help="local adapter dir")
    ap.add_argument("--out", required=True, help="output dir for merged model")
    args = ap.parse_args()

    print("loading base", args.base, flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu")
    print("loading adapter", args.adapter, flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging", flush=True)
    merged = model.merge_and_unload()
    print("saving to", args.out, flush=True)
    merged.save_pretrained(args.out)
    proc = AutoProcessor.from_pretrained(args.base)
    proc.save_pretrained(args.out)
    print("MERGE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
