#!/usr/bin/env python3
"""Bench driver: llama.cpp (llama-server) — base run, runtime "llama-cpp-q4km".

Protocol (identical for every runtime in this hunt):
  * load/start the model once, record load_s (includes server start + model load)
  * per image: 1 untimed warmup generation, then 3 timed generations; report median
  * greedy decoding, max_new_tokens=100, prompt "Dich chu trong anh sang tieng Viet."
    (see PROMPT below — the exact Vietnamese string)

Assumes (prepared by bench/llamacpp_setup.md, run from the repo root):
  * ./llama.cpp/build/bin/llama-server   (llama.cpp master, cmake build)
  * ./qwen3.5-0.8b-q4_k_m.gguf          (text model, Q4_K_M)
  * ./mmproj-qwen3.5-0.8b-f16.gguf      (vision projector, F16)
  * ./assets/img1_tmpt_ch155.png, ./assets/img2_tmpt_ch136.png,
    ./assets/img3_romcom.png

Writes ./results-llama-cpp-q4km.json in the cwd. Stdlib only (urllib, no
requests/psutil dependency). Fails loudly (nonzero exit) on any error.

Server flags for this base run (deliberately plain — the combo driver is where
tuning lives):
  -m <model> --mmproj <mmproj> -c 2048 -t <nproc> --threads-batch <nproc>
  --host 127.0.0.1 --port 8080 --jinja -np 1
  (--jinja: use the model's own chat template; -np 1: a single slot so the
  whole 2048 ctx belongs to this bench and no slot switching can occur.)

Overrides (for local testing only; the GH Actions run uses the defaults):
  env LLAMA_SERVER_BIN, or CLI --server-bin / --model / --mmproj / --port.
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

RUNTIME = "llama-cpp-q4km"
MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."
MAX_NEW_TOKENS = 100

ROOT = Path.cwd()
SERVER_BIN = Path(os.environ.get("LLAMA_SERVER_BIN", ROOT / "llama.cpp" / "build" / "bin" / "llama-server"))
MODEL = ROOT / "qwen3.5-0.8b-q4_k_m.gguf"
MMPROJ = ROOT / "mmproj-qwen3.5-0.8b-f16.gguf"
IMAGES = [
    "assets/img1_tmpt_ch155.png",
    "assets/img2_tmpt_ch136.png",
    "assets/img3_romcom.png",
]
PORT = 8080
HOST = "127.0.0.1"
NPROC = os.cpu_count() or 4
READY_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 600.0


def die(msg: str, code: int = 1) -> "SystemExit":
    print(f"FATAL: {msg}", file=sys.stderr)
    return SystemExit(code)


# ---------------------------------------------------------------- RSS sampler
# Best effort, no psutil: poll /proc/<pid>/status VmRSS from a thread while
# generations are in flight. VmRSS is the process resident set (weights +
# KV cache + runtime buffers); it misses child processes (llama-server is
# single-process here) and page-cache, which is stated in notes.

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
                f"llama-server exited during startup with code {proc.returncode} "
                f"(see server log tail in stderr / log file)")
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
    """One chat completion. Returns (output_text, completion_tokens, timings).

    timings is llama-server's own timings object from the response, if the
    server returns one (keys like prompt_n/prompt_ms, predicted_n/predicted_ms).
    """
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

def build_server_cmd(port: int, server_bin: Path, model: Path, mmproj: Path) -> list[str]:
    return [
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
    ]


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
                    (model, "quantized model GGUF"),
                    (mmproj, "mmproj GGUF")):
        if not p.is_file():
            print(f"FATAL: missing {what}: {p} (run bench/llamacpp_setup.md first)",
                  file=sys.stderr)
            return 1
    image_paths = []
    for rel in IMAGES:
        p = ROOT / rel
        if not p.is_file():
            print(f"FATAL: missing test image: {p}", file=sys.stderr)
            return 1
        image_paths.append(p)

    base_url = f"http://{HOST}:{args.port}"
    cmd = build_server_cmd(args.port, server_bin, model, mmproj)
    log_path = ROOT / f"llama-server-{RUNTIME}.log"
    print(f"[{RUNTIME}] starting: {' '.join(cmd)}")
    print(f"[{RUNTIME}] server log: {log_path}")

    t_start = time.perf_counter()
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        try:
            wait_ready(base_url, proc, READY_TIMEOUT_S)
        except RuntimeError as e:
            log_f.flush()
            tail = "".join(log_path.read_text(encoding="utf-8", errors="replace")
                           .splitlines(keepends=True)[-50:])
            print(f"FATAL: server failed to become ready: {e}\n--- server log tail ---\n{tail}",
                  file=sys.stderr)
            return 1
        load_s = time.perf_counter() - t_start
        print(f"[{RUNTIME}] ready in {load_s:.1f}s (load_s)")

        server_timings: list[str] = []  # one line per timed run, for notes
        results_images = []
        with RssSampler(proc.pid) as sampler:
            for img_path in image_paths:
                image_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                # 1 untimed warmup generation (also warms the vision encoder).
                w_text, w_tok, _ = generate(base_url, image_b64)
                if not w_text.strip():
                    log_f.flush()
                    tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:])
                    print(f"FATAL: warmup generation for {img_path.name} returned empty output\n--- server log tail ---\n{tail}",
                          file=sys.stderr)
                    return 1
                runs_s: list[float] = []
                outputs: list[str] = []
                tokens_list: list[int] = []
                for i in range(3):
                    t0 = time.perf_counter()
                    text, tokens, timings = generate(base_url, image_b64)
                    dt = time.perf_counter() - t0
                    if not text.strip():
                        log_f.flush()
                        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:])
                        print(f"FATAL: timed run {i+1} for {img_path.name} returned empty output\n--- server log tail ---\n{tail}",
                              file=sys.stderr)
                        return 1
                    runs_s.append(dt)
                    outputs.append(text)
                    tokens_list.append(tokens)
                    if timings:
                        server_timings.append(
                            f"{img_path.name} run{i+1}: "
                            + ", ".join(f"{k}={v}" for k, v in sorted(timings.items())))
                    print(f"[{RUNTIME}] {img_path.name} run{i+1}: {dt:.2f}s, "
                          f"{tokens} tok ({tokens / dt:.1f} tok/s)")
                if len(set(tokens_list)) != 1:
                    print(f"WARNING: token counts differ across timed runs for "
                          f"{img_path.name}: {tokens_list} (greedy should be "
                          f"deterministic; using first run's count)", file=sys.stderr)
                median_s = float(statistics.median(runs_s))
                tokens = tokens_list[0]
                results_images.append({
                    "name": img_path.name,
                    "runs_s": runs_s,
                    "median_s": median_s,
                    "tokens": tokens,
                    "tok_per_s": tokens / median_s if median_s > 0 else 0.0,
                    "peak_rss_mb": sampler.peak_mb,
                    "output": outputs[0].strip(),
                })
        peak_rss_mb = sampler.peak_mb
        # peak_rss_mb is a whole-run peak; report the same value per image
        # (the schema has no run-level RSS field). It is the peak over the
        # full bench, sampled every 50 ms from /proc/<pid>/status VmRSS.
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

    notes = (
        f"llama-server (llama.cpp master, cmake Release build) on a free "
        f"GitHub Actions ubuntu-latest runner (4 vCPU, ~16GB RAM), CPU-only. "
        f"Model {MODEL_ID} converted to GGUF and quantized to Q4_K_M "
        f"(mmproj kept F16); see bench/llamacpp_setup.md. Server flags: "
        f"{' '.join(cmd[1:])}. Threads: -t/--threads-batch = nproc = {NPROC}. "
        f"Request: POST /v1/chat/completions, image first then text prompt, "
        f"temperature 0, max_tokens 100, stream false. Token count from "
        f"usage.completion_tokens (fallback: POST /tokenize on the output). "
        f"peak_rss_mb: peak llama-server VmRSS sampled every 50 ms from "
        f"/proc/<pid>/status during the whole bench (same whole-run peak "
        f"reported for each image; no psutil used). load_s includes server "
        f"start + model/mmproj load until GET /health returns status ok. "
        f"Server's own timings per timed run: "
        + ("; ".join(server_timings) if server_timings else "not returned by server")
        + "."
    )
    result = {
        "runtime": RUNTIME,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "greedy": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_s": load_s,
        "images": results_images,
        "notes": notes,
    }
    out_path = ROOT / args.output
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{RUNTIME}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
