#!/usr/bin/env python3
"""Bench driver: llama.cpp (llama-server) — r14c merged, F16, zero-quant,
100 random images from the r14a/b/c legs.

Runtime "llama-r14c-fp16-100": ALL the combo speedups, ZERO quantization.
Weights are the r14c adapter merged into the base model, converted to F16
GGUF (no llama-quantize step at all) and the KV cache is F16 too, so
nothing in the run is quantized anywhere. Flags are otherwise the exact
round-3 combo: -fa, batch/ubatch 1024, --cache-prompt --cache-reuse 256.

Question this answers (user order 2026-11-22): how long does the fast
path take on 100 real training images with the r14c model? Unlike the
3-image hunt protocol (1 warmup + 3 timed per image, medians), this run
does ONE generation per image on 100 FRESH images — the real workload,
where prompt-cache reuse cannot help because every image is new. The
headline number is total_timed_s for the 100 images.

Images: assets/r14c-100/manifest.json — 100 crops sampled uniformly at
random (seed 20261122) from the deduped union of the r14a/b/c leg pairs
(5,964 unique pairs), TH + old + fresh crops. Same prompt, greedy,
max_new_tokens=100, enable_thinking=false.

Writes ./results-llama-r14c-fp16-100.json in the cwd. Stdlib only.
"""

from __future__ import annotations

import base64
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

RUNTIME = "llama-r14c-fp16-100"
MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."
MAX_NEW_TOKENS = 100

ROOT = Path.cwd()
SERVER_BIN = Path(os.environ.get("LLAMA_SERVER_BIN", ROOT / "llama.cpp" / "build" / "bin" / "llama-server"))
MODEL = ROOT / "qwen3.5-0.8b-r14c-f16.gguf"
MMPROJ = ROOT / "mmproj-qwen3.5-0.8b-r14c-f16.gguf"
MANIFEST = ROOT / "assets" / "r14c-100" / "manifest.json"
PORT = 8081
HOST = "127.0.0.1"
NPROC = os.cpu_count() or 4
READY_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 600.0


# ---------------------------------------------------------------- RSS sampler
# Best effort, no psutil: poll /proc/<pid>/status VmRSS from a thread while
# generations are in flight.

class RssSampler:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read_kb(self) -> int:
        try:
            with open(f"/proc/{self.pid}/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return 0
        return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            kb = self._read_kb()
            if kb > self.peak_kb:
                self.peak_kb = kb
            self._stop.wait(0.05)

    def __enter__(self) -> "RssSampler":
        self.peak_kb = self._read_kb()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return self.peak_kb / 1024.0


# ---------------------------------------------------------------- HTTP helpers

def http_json(method: str, url: str, payload: dict | None = None,
              timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} -> {e}") from e


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> float:
    """Poll GET /health until the server reports ok. Returns seconds waited."""
    t0 = time.perf_counter()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {proc.returncode}")
        try:
            resp = http_json("GET", f"{base_url}/health", timeout=5.0)
            if resp.get("status") == "ok":
                return time.perf_counter() - t0
        except RuntimeError:
            pass  # not up yet; keep polling until timeout / process death
        if time.perf_counter() - t0 > timeout:
            raise RuntimeError(f"llama-server not ready after {timeout:.0f}s")
        time.sleep(0.5)


def tokenize(base_url: str, text: str) -> int:
    """Fallback token counter via POST /tokenize (llama-server endpoint)."""
    resp = http_json("POST", f"{base_url}/tokenize",
                     {"content": text}, timeout=60.0)
    return len(resp.get("tokens", []))


def generate(base_url: str, image_b64: str) -> tuple[str, int, dict]:
    """One chat completion. Returns (output_text, completion_tokens, timings)."""
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
        # Qwen3.5 thinks by default under llama.cpp: without this the 100
        # tokens go to reasoning_content and content comes back empty
        # (round-2 diagnosis). Disabling thinking makes the answer land in
        # content, like the transformers/OpenVINO paths.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = http_json("POST", f"{base_url}/v1/chat/completions", payload,
                     timeout=REQUEST_TIMEOUT_S)
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected chat-completions response: {resp!r:.500}") from e
    if content is None:
        raise RuntimeError(f"server returned null content: {resp!r:.500}")
    content = str(content)
    if not content.strip():
        print(f"DEBUG: empty content, full response: {resp!r:.1500}", file=sys.stderr)
    usage = resp.get("usage") or {}
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int):
        tokens = tokenize(base_url, content)
    timings = resp.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    return content, tokens, timings


# ---------------------------------------------------------------- bench

def build_server_cmd(port: int, server_bin: Path, model: Path, mmproj: Path,
                     flash_attn: bool = True, cache_prompt: bool = True) -> list[str]:
    cmd = [
        str(server_bin),
        "-m", str(model),
        "--mmproj", str(mmproj),
        "-c", "2048",
        "-t", str(NPROC),
        "--threads-batch", str(NPROC),
        "--host", HOST,
        "--port", str(port),
        "--jinja",
        "-np", "1",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--batch-size", "1024",
        "--ubatch-size", "1024",
    ]
    if flash_attn:
        cmd += ["-fa", "on"]
    if cache_prompt:
        cmd += ["--cache-prompt", "--cache-reuse", "256"]
    return cmd


def start_server(cmd: list[str], log_path: Path):
    """Start llama-server, wait for readiness. Returns (proc, load_s, log_f)."""
    log_f = open(log_path, "w", encoding="utf-8")
    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        wait_ready(f"http://{HOST}:{cmd[cmd.index('--port') + 1]}", proc,
                   READY_TIMEOUT_S)
    except RuntimeError as e:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.flush()
        log_f.close()
        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace")
                       .splitlines(keepends=True)[-50:])
        raise RuntimeError(f"{e}\n--- server log tail ---\n{tail}") from e
    return proc, time.perf_counter() - t_start, log_f


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-bin", default=str(SERVER_BIN))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--mmproj", default=str(MMPROJ))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--output", default=f"results-{RUNTIME}.json")
    args = ap.parse_args(argv)

    server_bin, model, mmproj = Path(args.server_bin), Path(args.model), Path(args.mmproj)
    for p, what in ((server_bin, "llama-server binary"),
                    (model, "model GGUF"),
                    (mmproj, "mmproj GGUF")):
        if not p.is_file():
            print(f"FATAL: missing {what}: {p}",
                  file=sys.stderr)
            return 1
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        image_names = manifest["images"]
    except (OSError, ValueError, KeyError) as e:
        print(f"FATAL: cannot read manifest {MANIFEST}: {e}", file=sys.stderr)
        return 1
    image_paths = []
    for name in image_names:
        p = ROOT / "assets" / "r14c-100" / name
        if not p.is_file():
            print(f"FATAL: missing test image: {p}", file=sys.stderr)
            return 1
        image_paths.append(p)
    print(f"[{RUNTIME}] {len(image_paths)} images from manifest "
          f"(seed {manifest.get('seed')}, source: {manifest.get('source')})")

    base_url = f"http://{HOST}:{args.port}"
    log_path = ROOT / f"llama-server-{RUNTIME}.log"

    # Fallback ladder: full combo -> no flash-attn -> no flash-attn, no cache.
    attempts = [
        ("full combo", True, True),
        ("without -fa", False, True),
        ("without -fa and without --cache-prompt/--cache-reuse", False, False),
    ]
    proc = log_f = None
    load_s = None
    dropped: list[str] = []
    cmd: list[str] = []
    last_error = ""
    for label, flash_attn, cache_prompt in attempts:
        cmd = build_server_cmd(args.port, server_bin, model, mmproj,
                               flash_attn, cache_prompt)
        print(f"[{RUNTIME}] attempt '{label}': {' '.join(cmd)}")
        try:
            proc, load_s, log_f = start_server(cmd, log_path)
        except RuntimeError as e:
            last_error = str(e)
            print(f"[{RUNTIME}] attempt '{label}' failed: {e}", file=sys.stderr)
            dropped.append(label)
            continue
        print(f"[{RUNTIME}] ready in {load_s:.1f}s (load_s) with '{label}'")
        break
    if proc is None or load_s is None:
        print(f"FATAL: all llama-server attempts failed. Last error:\n{last_error}",
              file=sys.stderr)
        return 1

    try:
        results_images = []
        with RssSampler(proc.pid) as sampler:
            # One untimed warmup on the first image (server warm, not counted).
            first_b64 = base64.b64encode(image_paths[0].read_bytes()).decode("ascii")
            w_text, _, _ = generate(base_url, first_b64)
            if not w_text.strip():
                print(f"FATAL: warmup generation returned empty output",
                      file=sys.stderr)
                return 1
            print(f"[{RUNTIME}] warmup ok ({len(w_text)} chars)")
            t_wall0 = time.perf_counter()
            for n, img_path in enumerate(image_paths, 1):
                image_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                t0 = time.perf_counter()
                text, tokens, _ = generate(base_url, image_b64)
                dt = time.perf_counter() - t0
                if not text.strip():
                    print(f"FATAL: image {n}/{len(image_paths)} {img_path.name} "
                          f"returned empty output", file=sys.stderr)
                    return 1
                results_images.append({
                    "name": img_path.name,
                    "seconds": dt,
                    "tokens": tokens,
                    "tok_per_s": tokens / dt if dt > 0 else 0.0,
                    "output": text.strip(),
                })
                if n % 10 == 0 or n == len(image_paths):
                    done_s = sum(r["seconds"] for r in results_images)
                    print(f"[{RUNTIME}] {n}/{len(image_paths)}: "
                          f"{dt:.2f}s this, {done_s:.1f}s total so far "
                          f"(mean {done_s / n:.2f}s/image)")
            total_timed_s = time.perf_counter() - t_wall0
        peak_rss_mb = sampler.peak_mb
        for img in results_images:
            img["peak_rss_mb"] = peak_rss_mb
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()

    secs = [r["seconds"] for r in results_images]
    total_sum_s = sum(secs)
    notes = (
        f"llama-server (llama.cpp master, cmake Release build) on a free "
        f"GitHub Actions ubuntu-latest runner (4 vCPU, ~16GB RAM), CPU-only. "
        f"Model: r14c LoRA adapter merged into huihui-ai/Huihui-Qwen3.5-0.8B-"
        f"abliterated, converted to GGUF, kept at F16 (no quantize step; "
        f"mmproj also F16); KV cache F16. Combo server flags: "
        f"{' '.join(cmd[1:])}. "
        f"Flag rationale: -t/--threads-batch=nproc={NPROC} (all vCPUs); "
        f"-fa on (flash attention on full-attention layers, less KV "
        f"traffic); --cache-type-k/v f16 (explicitly NO KV quantization); "
        f"--batch-size/--ubatch-size 1024 (whole image+text prompt prefills "
        f"in ~1 chunk); --cache-prompt --cache-reuse 256 (kept for flag "
        f"parity with the noquant run; with 100 fresh images it cannot "
        f"reuse anything — every prompt is new). "
        f"Fallback attempts that failed before the successful one (dropped "
        f"flags): {'; '.join(dropped) if dropped else 'none — full combo started cleanly'}. "
        f"Protocol difference vs the 3-image hunt runs: ONE generation per "
        f"image (plus one untimed warmup on the first image), no repeats — "
        f"the real workload. Headline number is total_timed_s (wall clock "
        f"for the 100 timed generations, load_s excluded). Request: POST "
        f"/v1/chat/completions, image first then text prompt, temperature "
        f"0, max_tokens 100, stream false, chat_template_kwargs "
        f"enable_thinking=false. Token count from usage.completion_tokens "
        f"(fallback: POST /tokenize on the output). peak_rss_mb: peak "
        f"llama-server VmRSS sampled every 50 ms from /proc/<pid>/status "
        f"during the whole bench. load_s includes server start + model/mmproj "
        f"load until GET /health returns status ok."
    )
    result = {
        "runtime": RUNTIME,
        "model": "r14c-merged-f16 (adapter sft-r14c-out/pilot-out/epoch-1 on huihui-ai/Huihui-Qwen3.5-0.8B-abliterated)",
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": load_s,
        "n_images": len(results_images),
        "total_timed_s": total_timed_s,
        "total_sum_s": total_sum_s,
        "mean_s_per_image": total_sum_s / len(secs) if secs else 0.0,
        "median_s_per_image": float(statistics.median(secs)) if secs else 0.0,
        "min_s_per_image": min(secs) if secs else 0.0,
        "max_s_per_image": max(secs) if secs else 0.0,
        "peak_rss_mb": peak_rss_mb,
        "images": results_images,
        "notes": notes,
    }
    out_path = ROOT / args.output
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{RUNTIME}] wrote {out_path}")
    print(f"[{RUNTIME}] TOTAL for {len(results_images)} images: "
          f"{total_timed_s:.1f}s wall ({total_timed_s / len(results_images):.2f}s/image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
