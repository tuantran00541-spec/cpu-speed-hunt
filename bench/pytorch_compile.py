#!/usr/bin/env python3
"""CPU inference bench — PyTorch torch.compile (default mode).

Runtime name: "pytorch-compile"

Same protocol as pytorch_eager.py, except the model is wrapped with
torch.compile(model, mode="default") (backend defaults to "inductor").
max-autotune is deliberately NOT used: compile time explodes on CPU.

torch.compile is lazy: no compilation happens at wrap time. The first
generate call (the untimed warmup on img1) triggers Dynamo/Inductor
compilation, so the warmup covers compilation per the bench protocol.
compile_s below is the wall time of that first generate call on the
compiled model — i.e. Inductor compile + one 100-token generation, NOT
pure compile time (the two cannot be separated from outside).

torch._dynamo.config.suppress_errors is left at its default (False), so
a Dynamo failure raises instead of silently falling back to eager.

Run from the repo root:  python bench/pytorch_compile.py
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
RUNTIME = "pytorch-compile"
IMAGES = ["img1_tmpt_ch155.png", "img2_tmpt_ch136.png", "img3_romcom.png"]
MAX_NEW_TOKENS = 100
N_TIMED_RUNS = 3

ROOT = Path(__file__).resolve().parent.parent  # repo root


def log(msg: str) -> None:
    print(f"[{RUNTIME}] {msg}", flush=True)


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

    t_load0 = time.perf_counter()
    model, processor = load_model_and_processor()
    # --- runtime prep: torch.compile (lazy; wrap is ~instant) ---
    # Verified signature (PyTorch docs, torch.compile):
    #   torch.compile(model, *, fullgraph=False, dynamic=None,
    #                 backend="inductor", mode=None, options=None, disable=False)
    # mode="default" is the default preset; max-autotune NOT used (see docstring).
    model = torch.compile(model, mode="default")
    load_s = time.perf_counter() - t_load0
    log(f"model ready in {load_s:.1f}s (compile wrapper applied; compilation is lazy)")

    results = []
    compile_s = None
    for idx, name in enumerate(IMAGES):
        image = Image.open(ROOT / "assets" / name).convert("RGB")
        inputs = build_inputs(processor, image)

        # 1 untimed warmup generation. On img1 this is the first call on the
        # compiled model, so it triggers (and therefore times) compilation.
        t0 = time.perf_counter()
        generate(model, inputs)
        warm_s = time.perf_counter() - t0
        if idx == 0:
            compile_s = warm_s
            log(f"first-call (compile+1 generation) wall time: {compile_s:.1f}s")

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
        "compile_s": float(compile_s),
        "images": results,
        "notes": (
            "torch.compile(model, mode='default'), backend default 'inductor'; "
            "max-autotune NOT used (compile time explodes on CPU). torch "
            f"{torch.__version__}, transformers {transformers.__version__}, "
            "dtype=bfloat16, device_map=cpu, attn_implementation=default (sdpa), "
            "greedy (do_sample=False), max_new_tokens=100, "
            f"torch.set_num_threads({nproc}). compile_s is the wall time of the "
            "first generate call on the compiled model (warmup on img1): "
            "Inductor compilation + one 100-token generation, NOT pure compile "
            "time. suppress_errors left at default (False): a Dynamo failure "
            "raises instead of silently falling back to eager. Inputs prepared "
            "once per image via processor.apply_chat_template (image+text) and "
            "reused for warmup + 3 timed runs. Timing = model.generate() only "
            "(time.perf_counter). No autocast; weights loaded in bf16. "
            "peak_rss_mb via resource.getrusage(RUSAGE_SELF).ru_maxrss (KiB on "
            "Linux), process-wide peak so far."
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
