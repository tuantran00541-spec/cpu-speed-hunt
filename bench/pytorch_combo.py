#!/usr/bin/env python3
"""CPU inference bench — PyTorch combo (stack everything that composes).

Runtime name: "pytorch-combo"

Same bench protocol as pytorch_eager.py, but the model prep stacks the
PyTorch-side speedups and keeps only the subset that actually works,
verified by a smoke generation at each step. The final stack is recorded
in "notes" — nothing incompatible is forced together.

Stacking order and compatibility rules (all honest, smoke-tested):
  1. INT8 (dynamic quantization) is attempted FIRST, because it conflicts
     with the bf16 IPEX path (INT8 weights vs bf16 weights) and with
     compile in ways that must be observed, not assumed:
         torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear},
                                                dtype=torch.qint8)
     Smoke: greedy generate, max_new_tokens=16, on img1. Keep only if it
     runs AND decodes to non-empty text. If it raises or returns empty,
     the attempt is dropped, the model is RELOADED fresh from the
     checkpoint, and the reason is recorded in "notes".
  2. If INT8 is kept, IPEX bf16 is SKIPPED (dtype conflict: a model cannot
     be both INT8-quantized and bf16-IPEX-optimized), and torch.compile is
     then attempted on the quantized model, again smoke-gated.
  3. If INT8 is dropped, the path is IPEX + torch.compile:
         model = ipex.optimize(model, dtype=torch.bfloat16,
                               weights_prepack=False)
     weights_prepack=False is REQUIRED under TorchDynamo per Intel's docs
     ("in TorchDynamo mode ... disable weights prepacking by setting
     weights_prepack=False in ipex.optimize()"). Then:
         model = torch.compile(model, mode="default")
     smoke-gated; if compile raises or the smoke output is empty, compile
     is dropped and the IPEX-optimized eager model is kept.
  4. If the IPEX import itself fails, IPEX is dropped (recorded) and
     torch.compile alone is attempted on the eager model. Unlike
     pytorch_ipex.py, this script does NOT fail on a missing IPEX: its job
     is to keep the fastest WORKING subset.
  5. Thread tuning throughout: torch.set_num_threads(nproc).

Smoke checks are deliberately weak (runs + non-empty decoded text) — they
catch crashes and empty/garbage-collapse, not quality loss. Compare the
"output" fields in results-pytorch-combo.json against
results-pytorch-eager.json by eye: if the combo output is degraded
relative to eager, that is a result, and the notes say which piece to
blame (INT8 quantization is the usual suspect).

compile_s (when compile is kept) is the wall time of the FIRST generate
call on the compiled model (smoke or warmup, whichever comes first):
compile + one short generation, NOT pure compile time.

Run from the repo root:  python bench/pytorch_combo.py
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
RUNTIME = "pytorch-combo"
IMAGES = ["img1_tmpt_ch155.png", "img2_tmpt_ch136.png", "img3_romcom.png"]
MAX_NEW_TOKENS = 100
N_TIMED_RUNS = 3
SMOKE_NEW_TOKENS = 16

ROOT = Path(__file__).resolve().parent.parent  # repo root


def log(msg: str) -> None:
    print(f"[{RUNTIME}] {msg}", flush=True)


def peak_rss_mb() -> float:
    # resource.getrusage(RUSAGE_SELF).ru_maxrss is KiB on Linux.
    # Process-wide peak so far (monotonic); best effort, per protocol.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_model_and_processor():
    """Load a fresh model + processor. Returns (model, processor)."""
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


def smoke_generate(model, processor, inputs):
    """Smoke-test a candidate model. Returns (ok, seconds, reason).

    ok is True only if greedy generation runs AND decodes to non-empty
    text. This catches crashes and empty/garbage collapse, not subtle
    quality loss (compare outputs vs eager by eye afterwards).
    """
    t0 = time.perf_counter()
    try:
        out = generate(model, inputs, max_new_tokens=SMOKE_NEW_TOKENS)
    except Exception as exc:  # noqa: BLE001 - any failure drops the piece
        return False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t0
    try:
        text, _ = decode_output(processor, inputs, out)
    except Exception as exc:  # noqa: BLE001
        return False, dt, f"decode failed: {type(exc).__name__}: {exc}"
    if not text.strip():
        return False, dt, "empty smoke output"
    return True, dt, "ok"


def try_compile(model, processor, inputs, label: str):
    """Attempt torch.compile(model, mode='default') + smoke test.

    Returns (model_to_use, kept: bool, first_call_s or None, reason).
    The original model object is kept as fallback if compile fails.
    """
    try:
        compiled = torch.compile(model, mode="default")
    except Exception as exc:  # noqa: BLE001
        return model, False, None, f"torch.compile() raised {type(exc).__name__}: {exc}"
    ok, dt, reason = smoke_generate(compiled, processor, inputs)
    if ok:
        log(f"combo: torch.compile ({label}) smoke OK in {dt:.1f}s")
        return compiled, True, dt, "ok"
    log(f"combo: torch.compile ({label}) smoke FAILED ({reason}); dropping compile")
    return model, False, None, f"smoke failed: {reason}"


def main() -> int:
    nproc = os.cpu_count() or 4
    torch.set_num_threads(nproc)
    import transformers

    log(f"torch={torch.__version__} transformers={transformers.__version__}")
    log(f"cpus={nproc} torch_threads={torch.get_num_threads()}")

    stack_notes = []
    compile_s = None

    t_load0 = time.perf_counter()
    model, processor = load_model_and_processor()
    img1 = Image.open(ROOT / "assets" / IMAGES[0]).convert("RGB")
    inputs1 = build_inputs(processor, img1)

    # --- Step 1: INT8 dynamic quantization attempt (first: it conflicts
    # --- with the bf16 IPEX path, so it must be decided before IPEX). ---
    int8_kept = False
    try:
        log("combo: attempting INT8 dynamic quantization "
            "(torch.ao.quantization.quantize_dynamic, {Linear}, qint8) ...")
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        ok, dt, reason = smoke_generate(qmodel, processor, inputs1)
        if ok:
            int8_kept = True
            model = qmodel
            stack_notes.append(
                "INT8 KEPT: torch.ao.quantization.quantize_dynamic "
                f"({{Linear}}, qint8); smoke OK ({dt:.1f}s, non-empty output)."
            )
            log("combo: INT8 smoke OK; INT8 kept")
        else:
            stack_notes.append(f"INT8 DROPPED: smoke failed ({reason}).")
            log(f"combo: INT8 smoke failed ({reason}); dropping INT8")
    except Exception as exc:  # noqa: BLE001 - any failure drops INT8
        stack_notes.append(
            f"INT8 DROPPED: quantize_dynamic raised "
            f"{type(exc).__name__}: {exc}."
        )
        log(f"combo: INT8 attempt raised ({exc}); dropping INT8")

    if int8_kept:
        # INT8 weights vs bf16 IPEX optimize: dtype conflict, skip IPEX.
        stack_notes.append(
            "IPEX SKIPPED: dtype conflict — model is INT8-quantized, cannot "
            "also be bf16 IPEX-optimized (ipex.optimize(dtype=bfloat16) "
            "would recast/duplicate the weights)."
        )
        log("combo: skipping IPEX (INT8 dtype conflict)")
        model, compile_kept, first_s, reason = try_compile(
            model, processor, inputs1, "on INT8 model"
        )
        if compile_kept:
            compile_s = first_s
            stack_notes.append(
                "torch.compile KEPT on INT8 model (mode='default', smoke OK)."
            )
        else:
            stack_notes.append(f"torch.compile DROPPED on INT8 model: {reason}.")
        ipex_applied = False
    else:
        # INT8 failed: reload a fresh bf16 model. quantize_dynamic may have
        # touched the original module, so a clean reload is the only safe
        # base for the IPEX+compile path. (HF cache makes this a reload,
        # not a re-download; the reload time is inside load_s.)
        log("combo: reloading fresh bf16 model for the IPEX+compile path ...")
        model, processor = load_model_and_processor()
        inputs1 = build_inputs(processor, img1)

        # --- Step 2: IPEX (lenient here, unlike pytorch_ipex.py). ---
        ipex_applied = False
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError as exc:
            stack_notes.append(
                f"IPEX DROPPED: import failed ({type(exc).__name__}: {exc})."
            )
            log(f"combo: IPEX import failed ({exc}); dropping IPEX")
            ipex = None
        if ipex is not None:
            torch_mm = ".".join(torch.__version__.split(".")[:2])
            if not ipex.__version__.startswith(torch_mm):
                stack_notes.append(
                    f"IPEX DROPPED: version {ipex.__version__} does not match "
                    f"torch {torch.__version__} (need {torch_mm}.x)."
                )
                log("combo: IPEX version mismatch; dropping IPEX")
            else:
                try:
                    # weights_prepack=False is required under TorchDynamo
                    # per Intel docs (see module docstring).
                    optimized = ipex.optimize(
                        model, dtype=torch.bfloat16, weights_prepack=False
                    )
                    model = optimized[0] if isinstance(optimized, tuple) else optimized
                    ipex_applied = True
                    stack_notes.append(
                        f"IPEX KEPT: ipex.optimize(dtype=bfloat16, "
                        f"weights_prepack=False), ipex {ipex.__version__}."
                    )
                    log("combo: IPEX optimize OK (weights_prepack=False)")
                except Exception as exc:  # noqa: BLE001
                    stack_notes.append(
                        f"IPEX DROPPED: ipex.optimize raised "
                        f"{type(exc).__name__}: {exc}."
                    )
                    log(f"combo: ipex.optimize raised ({exc}); dropping IPEX")

        # --- Step 3: torch.compile on top (of IPEX model, or eager). ---
        label = "on IPEX model" if ipex_applied else "on eager model (no IPEX)"
        model, compile_kept, first_s, reason = try_compile(
            model, processor, inputs1, label
        )
        if compile_kept:
            compile_s = first_s
            stack_notes.append(
                f"torch.compile KEPT ({label}, mode='default', smoke OK)."
            )
        else:
            stack_notes.append(f"torch.compile DROPPED ({label}): {reason}.")

    load_s = time.perf_counter() - t_load0
    log(f"final stack: {'; '.join(stack_notes)}")
    log(f"model ready in {load_s:.1f}s")

    results = []
    for idx, name in enumerate(IMAGES):
        image = Image.open(ROOT / "assets" / name).convert("RGB")
        inputs = build_inputs(processor, image)

        # 1 untimed warmup generation. If compile was kept and the smoke
        # test already triggered compilation, this is just a warmup; if
        # compile is kept but somehow not yet triggered, this triggers it.
        t0 = time.perf_counter()
        generate(model, inputs)
        warm_s = time.perf_counter() - t0
        if idx == 0 and compile_s is None and "torch.compile KEPT" in "; ".join(stack_notes):
            compile_s = warm_s

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

    notes = (
        "Combo: staged stacking, each piece smoke-gated (greedy generate, "
        "16 new tokens on img1; keep only if it runs AND decodes non-empty). "
        + " ".join(stack_notes)
        + f" Thread tuning: torch.set_num_threads({nproc}). torch "
        f"{torch.__version__}, transformers {transformers.__version__}, "
        "base dtype=bfloat16, device_map=cpu, attn_implementation=default "
        "(sdpa), greedy (do_sample=False), max_new_tokens=100. "
        "If nothing stacked, the combo is the best working single piece + "
        "thread tuning. INT8 quality check is weak by design (non-empty "
        "only): compare 'output' here against results-pytorch-eager.json "
        "by eye — degraded output means the stacking piece, usually INT8, "
        "cost quality for speed. "
        + (
            "compile_s is the wall time of the first generate call on the "
            "compiled model (smoke or warmup): compile + one short "
            "generation, NOT pure compile time."
            if compile_s is not None
            else "No compile_s: torch.compile was not kept."
        )
        + " Inputs prepared once per image via "
        "processor.apply_chat_template (image+text) and reused for warmup + "
        "3 timed runs. Timing = model.generate() only (time.perf_counter). "
        "peak_rss_mb via resource.getrusage(RUSAGE_SELF).ru_maxrss (KiB on "
        "Linux), process-wide peak so far. load_s includes the INT8 attempt "
        "and any fresh-model reload it forced."
    )
    payload = {
        "runtime": RUNTIME,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": float(load_s),
        "images": results,
        "notes": notes,
    }
    if compile_s is not None:
        payload["compile_s"] = float(compile_s)
    out_path = Path.cwd() / f"results-{RUNTIME}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
