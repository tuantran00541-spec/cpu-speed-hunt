#!/usr/bin/env python3
"""Bench: OpenVINO combo (INT4 + GenAI VLMPipeline + MTP draft + tuning).

Runtime name: "openvino-combo"

Model dir (in cwd, produced by the job export step):
    ./ov_qwen3.5_0.8b_int4   (same INT4 export as openvino_int4.py)
Base model: huihui-ai/Huihui-Qwen3.5-0.8B-abliterated (BASE only, no LoRA).

What is stacked here — and what is NOT, with reasons (all verified against
the OpenVINO GenAI 2026.4 / optimum-intel 2.2.0 docs before writing this):

1. INT4 weights: yes — this script loads the INT4 export, not FP16/INT8.
2. MTP speculative decoding: attempted, NOT forced. Qwen3.5 ships a
   trained multi-token-prediction head and optimum-intel 2.2.0 exports it
   as `openvino_mtp_model.xml` in the model dir (Optimum-Intel v2.2.0
   release notes; OpenVINO/Qwen3.5-9B-int8-ov card: "includes
   openvino_mtp_model.xml"). This script checks for that file at runtime:
   if present, it builds the draft with
   `ov_genai.draft_model(model_dir, "CPU")` and passes it as
   `draft_model=` to VLMPipeline, with `num_assistant_tokens = 2`
   (the value in the official Qwen3.5 MTP example). If the file is
   absent (e.g. this abliterated 0.8B checkpoint lost its MTP head),
   the script runs WITHOUT a draft model and records that in "notes" —
   it never fakes speculative decoding.
3. Prefix caching: MUTUALLY EXCLUSIVE with MTP. Official docs state
   "Prefix caching is not currently supported in MTP mode" and the
   official MTP examples set `scheduler_config.enable_prefix_caching
   = False`. So: MTP present -> prefix caching OFF; MTP absent ->
   prefix caching ON (`enable_prefix_caching = True`). The two are
   never enabled together.
4. Thread tuning: `INFERENCE_NUM_THREADS = <nproc>` and `NUM_STREAMS = 1`
   passed as VLMPipeline constructor kwargs (the constructor forwards
   extra kwargs as OpenVINO device properties to compile_model; see
   openvino_genai.VLMPipeline docs: "__init__(models_path, device,
   **kwargs) ... kwargs: Device properties"). If the constructor rejects
   them, the script retries without them and records the fallback in
   "notes". NUM_STREAMS=1 + latency-oriented single-sequence bench.
5. API choice: openvino-genai VLMPipeline, NOT optimum-intel OVModel —
   OVModelForVisualCausalLM exposes no draft-model / speculative
   decoding API, so MTP is only reachable through GenAI.

Protocol (identical for every runtime):
  - Load the model once, record load_s.
  - Per image: 1 untimed warmup generation, then 3 timed generations;
    report the median. Timing covers pipe.generate() only.
  - Greedy decoding (do_sample=False), max_new_tokens=100.
  - Prompt: "Dịch chữ trong ảnh sang tiếng Việt." — passed as a plain
    string, exactly as the official Qwen3.5 GenAI example passes
    "Describe this image." (the pipeline applies the chat template).

One protocol caveat, recorded honestly: VLMPipeline.generate() returns
decoded TEXT, not token ids, so the generated-token count is obtained by
re-tokenizing the output text with the model's tokenizer
(tokenizer.encode(output, add_special_tokens=False)). That can differ
by a token from the true generated-id count if the tokenizer merges
across the output boundary; it is the only token count this API offers.

Usage (run from the repo root):
    python bench/openvino_combo.py [--model-dir ./ov_qwen3.5_0.8b_int4] [--output ./results-openvino-combo.json]

Writes ./results-openvino-combo.json (schema per the hunt spec).
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

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."
RUNTIME = "openvino-combo"
MAX_NEW_TOKENS = 100
NUM_ASSISTANT_TOKENS = 2  # value used in the official Qwen3.5 MTP example
IMAGES = [
    "assets/img1_tmpt_ch155.png",
    "assets/img2_tmpt_ch136.png",
    "assets/img3_romcom.png",
]


def peak_rss_mb() -> float:
    """Process high-water mark, Linux: ru_maxrss is in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_pipeline(model_dir: Path, mtp_present: bool, n_threads: int):
    """Construct VLMPipeline. Returns (pipe, notes_list).

    MTP present -> draft model + prefix caching OFF (required by GenAI).
    MTP absent  -> no draft + prefix caching ON (the fallback stack).
    Thread properties are attempted first; on rejection retried without.
    """
    notes = []
    scheduler_config = ov_genai.SchedulerConfig()
    draft = None
    if mtp_present:
        scheduler_config.enable_prefix_caching = False
        draft = ov_genai.draft_model(str(model_dir), "CPU")
        notes.append(
            "MTP: openvino_mtp_model.xml found; draft_model loaded from the "
            "same INT4 dir, num_assistant_tokens=2, prefix caching DISABLED "
            "(required: GenAI does not support prefix caching in MTP mode)."
        )
    else:
        scheduler_config.enable_prefix_caching = True
        notes.append(
            "MTP: openvino_mtp_model.xml NOT found in the INT4 export, so NO "
            "speculative decoding is active (not faked); prefix caching "
            "ENABLED instead as the fallback stack."
        )

    kwargs = {"draft_model": draft} if draft is not None else {}
    kwargs["scheduler_config"] = scheduler_config
    thread_kwargs = {"INFERENCE_NUM_THREADS": n_threads, "NUM_STREAMS": 1}
    try:
        pipe = ov_genai.VLMPipeline(
            str(model_dir), "CPU", **kwargs, **thread_kwargs
        )
        notes.append(
            f"Thread tuning: INFERENCE_NUM_THREADS={n_threads}, NUM_STREAMS=1 "
            "passed as VLMPipeline device-property kwargs."
        )
    except Exception as exc:  # noqa: BLE001 - a tunable piece, drop + record
        print(f"[{RUNTIME}] thread-kwarg construction failed ({exc!r}); "
              "retrying without thread properties.", flush=True)
        pipe = ov_genai.VLMPipeline(str(model_dir), "CPU", **kwargs)
        notes.append(
            "Thread tuning DROPPED: VLMPipeline rejected "
            f"INFERENCE_NUM_THREADS/NUM_STREAMS kwargs ({exc!r}); pipeline "
            "built with OpenVINO CPU defaults instead."
        )
    return pipe, notes


def generate_once(pipe, gen_config, image: Image.Image):
    """One greedy generation via VLMPipeline. Returns output text."""
    image_tensor = ov.Tensor(np.ascontiguousarray(np.array(image)[None]))
    # generate() form copied from the official Qwen3.5-9B GenAI example:
    #   pipe.generate("Describe this image.", image=image_tensor,
    #                 generation_config=generation_config)
    result = pipe.generate(
        PROMPT, image=image_tensor, generation_config=gen_config
    )
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0]).strip()
    return str(result).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench OpenVINO combo (INT4+MTP)")
    ap.add_argument("--model-dir", default="./ov_qwen3.5_0.8b_int4")
    ap.add_argument("--output", default="./results-openvino-combo.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        print("Export it first, see bench/openvino_job_steps.md", file=sys.stderr)
        return 2

    n_threads = os.cpu_count() or 4
    mtp_present = (model_dir / "openvino_mtp_model.xml").is_file()
    print(f"[{RUNTIME}] MTP file present: {mtp_present}", flush=True)

    t0 = time.perf_counter()
    # Tokenizer only (GenAI returns text, not ids): used to count tokens by
    # re-tokenizing the output. Loaded from the same exported dir.
    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer
    pipe, stack_notes = build_pipeline(model_dir, mtp_present, n_threads)

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = MAX_NEW_TOKENS
    gen_config.do_sample = False
    gen_config.num_return_sequences = 1
    if mtp_present:
        gen_config.num_assistant_tokens = NUM_ASSISTANT_TOKENS
    load_s = time.perf_counter() - t0
    print(f"[{RUNTIME}] loaded {model_dir} in {load_s:.2f}s", flush=True)

    results_images = []
    for rel in IMAGES:
        path = Path(rel)
        image = Image.open(path).convert("RGB")

        # 1 untimed warmup generation.
        generate_once(pipe, gen_config, image)

        # 3 timed generations; timing covers pipe.generate() only.
        runs_s = []
        first_output = None
        first_tokens = 0
        for i in range(3):
            t1 = time.perf_counter()
            text = generate_once(pipe, gen_config, image)
            runs_s.append(time.perf_counter() - t1)
            if i == 0:
                first_output = text
                first_tokens = len(
                    tokenizer.encode(text, add_special_tokens=False)
                )
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

    notes = (
        "openvino-genai VLMPipeline on the INT4 export "
        "(./ov_qwen3.5_0.8b_int4). Greedy (do_sample=False), "
        "max_new_tokens=100, prompt passed as a plain string (pipeline "
        "applies the chat template, per the official Qwen3.5 GenAI example). "
        "Timing covers pipe.generate() only; 1 warmup + 3 timed runs per "
        "image, median reported. "
        + " ".join(stack_notes)
        + " MTP and prefix caching are never enabled together (GenAI "
        "requires prefix caching off in MTP mode). tokens = "
        "len(tokenizer.encode(output, add_special_tokens=False)) — "
        "re-tokenization of the returned text, because VLMPipeline returns "
        "text, not token ids; may differ by a token from the true generated "
        "count. peak_rss_mb via resource.getrusage(RUSAGE_SELF).ru_maxrss / "
        "1024 (Linux KiB -> MB), process-wide high-water mark at end of that "
        "image's runs. Qwen3.5-0.8B runs in non-thinking mode by default (per "
        "Qwen/Qwen3.5-0.8B model card)."
    )
    payload = {
        "runtime": RUNTIME,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": float(load_s),
        "images": results_images,
        "notes": notes,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[{RUNTIME}] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
