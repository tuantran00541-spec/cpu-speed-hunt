#!/usr/bin/env python3
"""Bench: OpenVINO INT8 (optimum-intel OVModelForVisualCausalLM).

Runtime name: "openvino-int8"

Model dir (in cwd, produced by the job export step):
    ./ov_qwen3.5_0.8b_int8
Base model: huihui-ai/Huihui-Qwen3.5-0.8B-abliterated (BASE only, no LoRA).

Protocol (identical for every runtime):
  - Load the model once, record load_s.
  - Per image: 1 untimed warmup generation, then 3 timed generations;
    report the median. Timing covers model.generate() only.
  - Greedy decoding (do_sample=False), max_new_tokens=100.
  - Prompt (plain text inside the chat template, image first):
        "Dịch chữ trong ảnh sang tiếng Việt."

Inference pattern copied from the official OpenVINO Qwen3.5 model card
(OpenVINO/Qwen3.5-9B-int8-ov README): AutoProcessor + chat template with
{"type": "image"} / {"type": "text"} messages, then
OVModelForVisualCausalLM.from_pretrained(...) and model.generate(**inputs).

Usage (run from the repo root):
    python bench/openvino_int8.py [--model-dir ./ov_qwen3.5_0.8b_int8] [--output ./results-openvino-int8.json]

Writes ./results-openvino-int8.json (schema per the hunt spec).
No secrets, no network calls at bench time (model dir is local).
"""

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

# Verified import path (optimum-intel docs + OpenVINO Qwen3.5 model cards):
#   from optimum.intel.openvino import OVModelForVisualCausalLM
from optimum.intel.openvino import OVModelForVisualCausalLM

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."
RUNTIME = "openvino-int8"
MAX_NEW_TOKENS = 100
IMAGES = [
    "assets/img1_tmpt_ch155.png",
    "assets/img2_tmpt_ch136.png",
    "assets/img3_romcom.png",
]


def peak_rss_mb() -> float:
    """Process high-water mark, Linux: ru_maxrss is in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_inputs(processor, image: Image.Image):
    """Chat-template inputs for one image + the fixed prompt.

    Same construction as the official OpenVINO Qwen3.5 README:
    messages -> apply_chat_template(tokenize=False,
    add_generation_prompt=True) -> processor(text=[...], images=[...]).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], images=[image], return_tensors="pt")


def generate_once(model, processor, image: Image.Image):
    """One greedy generation. Returns (output_text, generated_token_count)."""
    inputs = build_inputs(processor, image)
    prompt_len = int(inputs.input_ids.shape[1])
    outputs = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
    )
    gen_ids = outputs[:, prompt_len:]
    n_tokens = int(gen_ids.shape[1])
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return text, n_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench OpenVINO INT8 VLM")
    ap.add_argument("--model-dir", default="./ov_qwen3.5_0.8b_int8")
    ap.add_argument("--output", default="./results-openvino-int8.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        print("Export it first, see bench/openvino_job_steps.md", file=sys.stderr)
        return 2

    torch.set_num_threads(os.cpu_count() or 4)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    # No ov_config here on purpose: this is the plain INT8 baseline.
    # Default device for OVModelForVisualCausalLM is CPU.
    model = OVModelForVisualCausalLM.from_pretrained(str(model_dir), device="CPU")
    load_s = time.perf_counter() - t0
    print(f"[{RUNTIME}] loaded {model_dir} in {load_s:.2f}s", flush=True)

    results_images = []
    for rel in IMAGES:
        path = Path(rel)
        image = Image.open(path).convert("RGB")

        # 1 untimed warmup generation.
        generate_once(model, processor, image)

        # 3 timed generations; timing covers generate() only.
        runs_s = []
        first_output = None
        first_tokens = 0
        for i in range(3):
            t1 = time.perf_counter()
            text, n_tokens = generate_once(model, processor, image)
            runs_s.append(time.perf_counter() - t1)
            if i == 0:
                first_output = text
                first_tokens = n_tokens
        median_s = float(statistics.median(runs_s))
        rec = {
            "name": Path(rel).name,
            "runs_s": [float(x) for x in runs_s],
            "median_s": median_s,
            "tokens": int(first_tokens),
            "tok_per_s": (float(first_tokens) / median_s) if median_s > 0 else 0.0,
            "peak_rss_mb": float(peak_rss_mb()),
            "output": first_output,
        }
        results_images.append(rec)
        print(
            f"[{RUNTIME}] {rec['name']}: median {median_s:.2f}s, "
            f"{first_tokens} tok, {rec['tok_per_s']:.2f} tok/s, "
            f"peak RSS {rec['peak_rss_mb']:.0f} MB",
            flush=True,
        )

    payload = {
        "runtime": RUNTIME,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": float(load_s),
        "images": results_images,
        "notes": (
            "optimum-intel OVModelForVisualCausalLM.from_pretrained(model_dir, "
            "device='CPU'), no ov_config (plain INT8 baseline). AutoProcessor + "
            "chat template (image message first, then text prompt), "
            "model.generate(max_new_tokens=100, do_sample=False). Timing covers "
            "generate() only; 1 warmup + 3 timed runs per image, median reported. "
            "tokens = generated ids after the prompt (outputs[:, prompt_len:]). "
            "peak_rss_mb via resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024 "
            "(Linux KiB -> MB), process-wide high-water mark at end of that "
            "image's runs. Weights: OpenVINO IR exported by optimum-cli "
            "--weight-format int8 from the base model (see "
            "bench/openvino_job_steps.md). Qwen3.5-0.8B runs in non-thinking mode "
            "by default (per Qwen/Qwen3.5-0.8B model card), no enable_thinking flag."
        ),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[{RUNTIME}] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
