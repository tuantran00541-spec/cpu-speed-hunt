#!/usr/bin/env python3
"""CPU inference bench — Intel Extension for PyTorch (IPEX).

Runtime name: "pytorch-ipex"

Same protocol as pytorch_eager.py, except the model is optimized with
IPEX after loading:

    import intel_extension_for_pytorch as ipex
    model = ipex.optimize(model, dtype=torch.bfloat16)

Verified signature (Intel IPEX API docs, CPU):
    ipex.optimize(model, dtype=None, optimizer=None, level="O1",
                  inplace=False, conv_bn_folding=None, weights_prepack=None,
                  replace_dropout_with_identity=None, optimize_lstm=None,
                  split_master_weight_for_bf16=None, fuse_update_step=None,
                  auto_kernel_selection=None, sample_input=None)
With optimizer=None this is the inference case and returns the optimized
model (a single model, not a tuple). No optimizer is passed here.

FAIL-LOUD RULE (by design, per the hunt spec): if the IPEX import fails,
if the IPEX version does not match the torch major.minor version, or if
ipex.optimize() raises, this script exits nonzero. It NEVER silently
falls back to eager — a fallback would mislabel an eager run as "ipex".

Notes:
  * ipex.optimize() deep-copies the model (per Intel docs), so peak RSS
    transiently holds ~2x weights; 16 GB runner RAM is ample for 0.8B.
  * ipex.llm.optimize / ipex.optimize_transformers are NOT used: they are
    LLM-focused APIs whose supported model list (Llama, GPT-J, OPT,
    Falcon, ...) does not include Qwen3.5-VL; plain ipex.optimize is the
    documented generic inference path for a transformers model.
  * No autocast context: weights are already bf16 from load, matching the
    other runtimes' numerics as closely as possible.

Run from the repo root:  python bench/pytorch_ipex.py
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."
RUNTIME = "pytorch-ipex"
IMAGES = ["img1_tmpt_ch155.png", "img2_tmpt_ch136.png", "img3_romcom.png"]
MAX_NEW_TOKENS = 100
N_TIMED_RUNS = 3

ROOT = Path(__file__).resolve().parent.parent  # repo root


def log(msg: str) -> None:
    print(f"[{RUNTIME}] {msg}", flush=True)


def fail(msg: str):
    """Fail loudly, nonzero exit, no fallback. This is the spec."""
    print(f"[{RUNTIME}] FATAL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def peak_rss_mb() -> float:
    # resource.getrusage(RUSAGE_SELF).ru_maxrss is KiB on Linux.
    # Process-wide peak so far (monotonic); best effort, per protocol.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_model_and_processor():
    """Load model + processor. Returns (model, processor).

    Caller measures load_s around this + any runtime prep.
    """
    log(f"loading {MODEL_ID} (dtype=bfloat16, device_map=cpu) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def build_inputs(processor, image: Image.Image):
    """One chat-templated input for (image, PROMPT). Prepared once per image
    and reused for the warmup + all timed runs."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.no_grad()
def generate(model, inputs, max_new_tokens: int = MAX_NEW_TOKENS):
    return model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )


def decode_output(processor, inputs, generated_ids):
    n_in = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[0][n_in:]
    text = processor.decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text, int(new_ids.shape[0])


def main() -> int:
    nproc = os.cpu_count() or 4
    torch.set_num_threads(nproc)
    import transformers

    log(f"torch={torch.__version__} transformers={transformers.__version__}")
    log(f"cpus={nproc} torch_threads={torch.get_num_threads()}")

    # --- IPEX import: fail loudly, never fall back to eager. ---
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError as exc:
        fail(
            "intel_extension_for_pytorch import failed "
            f"({type(exc).__name__}: {exc}). Refusing to fall back to eager; "
            "install a matching IPEX (see bench/pytorch_job_steps.md)."
        )
    log(f"ipex={ipex.__version__}")

    # IPEX must match the torch major.minor version exactly (Intel docs).
    torch_mm = ".".join(torch.__version__.split(".")[:2])
    if not ipex.__version__.startswith(torch_mm):
        fail(
            f"IPEX version {ipex.__version__} does not match "
            f"torch {torch.__version__} (need {torch_mm}.x). "
            "IPEX must match the torch major.minor version."
        )

    t_load0 = time.perf_counter()
    model, processor = load_model_and_processor()
    # --- runtime prep: IPEX optimize (inference case, no optimizer) ---
    t_ipex0 = time.perf_counter()
    try:
        optimized = ipex.optimize(model, dtype=torch.bfloat16)
    except Exception as exc:
        fail(f"ipex.optimize(model, dtype=torch.bfloat16) raised "
             f"({type(exc).__name__}: {exc}). No eager fallback.")
    # Docs: with optimizer=None returns the optimized model (single return).
    # Defensively unwrap a tuple if a future version returns (model, opt).
    model = optimized[0] if isinstance(optimized, tuple) else optimized
    ipex_s = time.perf_counter() - t_ipex0
    load_s = time.perf_counter() - t_load0
    log(f"IPEX optimize took {ipex_s:.1f}s; model ready in {load_s:.1f}s")

    results = []
    for name in IMAGES:
        image = Image.open(ROOT / "assets" / name).convert("RGB")
        inputs = build_inputs(processor, image)

        # 1 untimed warmup generation.
        generate(model, inputs)

        # 3 timed generations.
        runs_s = []
        first_text = None
        first_tokens = None
        for i in range(N_TIMED_RUNS):
            t0 = time.perf_counter()
            out = generate(model, inputs)
            runs_s.append(time.perf_counter() - t0)
            if i == 0:
                first_text, first_tokens = decode_output(processor, inputs, out)

        median_s = float(statistics.median(runs_s))
        results.append(
            {
                "name": name,
                "runs_s": [float(r) for r in runs_s],
                "median_s": median_s,
                "tokens": int(first_tokens),
                "tok_per_s": float(first_tokens / median_s) if median_s > 0 else 0.0,
                "peak_rss_mb": float(peak_rss_mb()),
                "output": first_text,
            }
        )
        log(
            f"{name}: median={median_s:.2f}s runs={[round(r, 2) for r in runs_s]} "
            f"tokens={first_tokens} tok/s={first_tokens / median_s:.2f}"
        )

    payload = {
        "runtime": RUNTIME,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": float(load_s),
        "images": results,
        "notes": (
            "IPEX: ipex.optimize(model, dtype=torch.bfloat16) after load, "
            "eval mode, no optimizer (inference case). IPEX "
            f"{ipex.__version__} matched to torch {torch.__version__}; "
            f"ipex.optimize took {ipex_s:.1f}s (included in load_s). "
            "ipex.llm.optimize / optimize_transformers NOT used: LLM-focused "
            "APIs whose supported list does not include Qwen3.5-VL. torch "
            f"{torch.__version__}, transformers {transformers.__version__}, "
            "dtype=bfloat16, device_map=cpu, attn_implementation=default (sdpa), "
            "greedy (do_sample=False), max_new_tokens=100, "
            f"torch.set_num_threads({nproc}). Import/optimize failure is fatal "
            "by design (no eager fallback). Inputs prepared once per image via "
            "processor.apply_chat_template (image+text) and reused for warmup + "
            "3 timed runs. Timing = model.generate() only (time.perf_counter). "
            "No autocast; weights loaded in bf16. peak_rss_mb via "
            "resource.getrusage(RUSAGE_SELF).ru_maxrss (KiB on Linux), "
            "process-wide peak so far."
        ),
    }
    out_path = Path.cwd() / f"results-{RUNTIME}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
