# llama.cpp lane — setup for a fresh `ubuntu-latest` GitHub Actions job

Run every command from the **repo root** unless a step says otherwise.
No secrets, no tokens, no paid services: the model repo is public and the
`huggingface_hub` download below is anonymous.

What this produces (filenames must match — the bench drivers assume them):
| File | What it is |
| --- | --- |
| `llama.cpp/build/bin/llama-server` | llama.cpp server binary |
| `llama.cpp/build/bin/llama-quantize` | llama.cpp quantizer |
| `qwen3.5-0.8b-f16.gguf` | text model, F16 (intermediate) |
| `qwen3.5-0.8b-q4_k_m.gguf` | text model, **Q4_K_M** (what the drivers load) |
| `mmproj-qwen3.5-0.8b-f16.gguf` | vision projector, F16 (what the drivers load) |

Verified 2026-11-21 against llama.cpp `master` (commit `1fdda89a`, README
quickstart uses `ggml-org/Qwen3.5-0.8B-GGUF`, i.e. the same `qwen35` arch):
Qwen3.5 is a supported architecture (`qwen35`/`qwen35moe` in
`src/llama-arch.*`), vision works via a separate mmproj file produced by
`convert_hf_to_gguf.py --mmproj`, and the server flags used by the drivers
(`-fa`/`--flash-attn`, `--cache-type-k/v`, `--batch-size`, `--ubatch-size`,
`--cache-prompt`, `--cache-reuse`, `--jinja`, `-np`) are all current
`llama-server` options. llama.cpp moves fast — if a flag is rejected at
runtime, the combo driver retries without it and records the drop in
`notes`; the base driver fails loudly instead.

## 0. Runner prerequisites (already on `ubuntu-latest`)

`git`, `cmake` (>= 3.27-ish), a C++ compiler, `python3` (>= 3.10). Check:

```bash
git --version && cmake --version | head -1 && python3 --version && nproc
```

## 1. Clone llama.cpp master and build (CPU-only, Release)

Per `llama.cpp/docs/build.md` (CPU build); plain `cmake -B build` builds for
the hardware it runs on, which is the same runner that later runs the bench,
so no extra `-DGGML_*` flags are needed. No GPU flags: the runner has no GPU.

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j "$(nproc)"
ls -lh build/bin/llama-server build/bin/llama-quantize
./build/bin/llama-server --version
cd ..
```

Expected: both binaries exist under `llama.cpp/build/bin/` and
`--version` prints a version string. Build takes on the order of 10 minutes
on a 4-vCPU runner; give the job a generous timeout.

## 2. Python dependencies for download + conversion

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python3 -m pip install huggingface_hub torch --index-url https://download.pytorch.org/whl/cpu transformers
```

Why each: `requirements-convert_hf_to_gguf.txt` is llama.cpp's own
conversion requirements; `huggingface_hub` downloads the checkpoint;
`torch` (CPU wheel) + `transformers` are what `convert_hf_to_gguf.py`
needs to read the Qwen3.5 checkpoint and tokenizer. No HF token anywhere.

## 3. Download the base model (public, no token)

`huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` is the abliterated version of
upstream `Qwen/Qwen3.5-0.8B` — a **full multimodal** checkpoint (language
model + ViT vision encoder + projector), ~0.85B parameters. Download the
whole repo directory; the converter needs `config.json`, the safetensors
shards, and the tokenizer files together.

```bash
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
                  local_dir="hf-model")
EOF
ls hf-model && du -sh hf-model
```

Expected: `hf-model/` contains `config.json`, `*.safetensors`,
`tokenizer.json` / `tokenizer_config.json`, `preprocessor_config.json`.
(`config.json` should list an architecture along the lines of
`Qwen3_5ForConditionalGeneration` — that is the `qwen35` arch llama.cpp
converts.)

## 4. Convert to GGUF — text model (F16)

Run from inside `llama.cpp/`, exactly as in llama.cpp's own conversion
examples (`--outtype f16` = full-precision intermediate; quantization is a
separate step below, never convert straight to a quantized type):

```bash
cd llama.cpp
python3 convert_hf_to_gguf.py ../hf-model \
  --outfile ../qwen3.5-0.8b-f16.gguf \
  --outtype f16
cd ..
ls -lh qwen3.5-0.8b-f16.gguf
```

Expected: `qwen3.5-0.8b-f16.gguf` appears at the repo root (order of ~1–2 GB).
**If this step fails with an unknown-architecture error, stop**: it means
the checked-out llama.cpp predates `qwen35` support — re-clone master (step 1
pins nothing, so this should not happen) and do not proceed to a hand-rolled
conversion. (Cross-check: upstream publishes the same conversion as
`ggml-org/Qwen3.5-0.8B-GGUF`; if conversion itself is ever blocked, that repo
is the known-good reference for what the converter should produce — but the
bench must still use the locally converted file unless the hunt coordinator
says otherwise.)

## 5. Convert to GGUF — mmproj (vision projector, F16)

Same converter, same checkpoint, with the `--mmproj` flag. This is the step
that produces the vision encoder + projector file; **without it the server
cannot take images**. Do not quantize the mmproj — it stays F16, as the
drivers assume.

```bash
cd llama.cpp
python3 convert_hf_to_gguf.py ../hf-model \
  --outfile ../mmproj-qwen3.5-0.8b-f16.gguf \
  --outtype f16 \
  --mmproj
cd ..
ls -lh mmproj-qwen3.5-0.8b-f16.gguf
```

Expected: `mmproj-qwen3.5-0.8b-f16.gguf` at the repo root (a few hundred MB).
Sanity check that it is actually a projector file, not a second copy of the
text model: it must be clearly smaller than `qwen3.5-0.8b-f16.gguf`, and
running the server without `--mmproj` must later fail on image input while
running with it succeeds (the drivers exercise this on every run).

## 6. Quantize the text model to Q4_K_M

Only the text model is quantized; the mmproj stays F16 (step 5).

```bash
./llama.cpp/build/bin/llama-quantize \
  qwen3.5-0.8b-f16.gguf \
  qwen3.5-0.8b-q4_k_m.gguf \
  Q4_K_M
ls -lh qwen3.5-0.8b-q4_k_m.gguf
```

Expected: `qwen3.5-0.8b-q4_k_m.gguf` at the repo root (~0.5–0.6 GB).
Keep `qwen3.5-0.8b-f16.gguf` around (it is the re-quantize source); delete
it only if runner disk gets tight.

## 7. Verify the full file set, then run the bench

```bash
ls -lh llama.cpp/build/bin/llama-server \
  qwen3.5-0.8b-q4_k_m.gguf \
  mmproj-qwen3.5-0.8b-f16.gguf \
  assets/img1_tmpt_ch155.png assets/img2_tmpt_ch136.png assets/img3_romcom.png

python3 bench/llamacpp.py          # base run  -> results-llama-cpp-q4km.json
python3 bench/llamacpp_combo.py    # combo run -> results-llama-cpp-combo.json
```

Each driver starts its own server (base on port 8080, combo on 8081),
records `load_s`, runs 1 warmup + 3 timed generations per image, stops the
server, and writes its `results-*.json` in the repo root. Either driver
exiting nonzero = the run failed; the server log
(`llama-server-llama-cpp-q4km.log` / `llama-server-llama-cpp-combo.log`)
holds the last lines of server output for diagnosis — and the workflow
uploads it as an artifact. On an empty generation the driver now also
prints the full chat-completions response plus the server log tail to
stderr before failing (added after run 1, where both drivers died on an
empty warmup with no diagnostic output).

**Thinking mode (round-2 diagnosis):** Qwen3.5 thinks by default under
llama.cpp — the 100 generated tokens went to `reasoning_content` and
`content` came back empty, even though the server's own timings showed
100 tokens generated (42.9 tok/s base). Both drivers now send
`"chat_template_kwargs": {"enable_thinking": false}` in the request, so
the answer lands in `content` like the transformers/OpenVINO paths.

## One-command recap (copy-paste, repo root)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j "$(nproc)" && cd ..
python3 -m pip install --upgrade pip
python3 -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python3 -m pip install huggingface_hub torch --index-url https://download.pytorch.org/whl/cpu transformers
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='huihui-ai/Huihui-Qwen3.5-0.8B-abliterated', local_dir='hf-model')"
cd llama.cpp
python3 convert_hf_to_gguf.py ../hf-model --outfile ../qwen3.5-0.8b-f16.gguf --outtype f16
python3 convert_hf_to_gguf.py ../hf-model --outfile ../mmproj-qwen3.5-0.8b-f16.gguf --outtype f16 --mmproj
cd ..
./llama.cpp/build/bin/llama-quantize qwen3.5-0.8b-f16.gguf qwen3.5-0.8b-q4_k_m.gguf Q4_K_M
python3 bench/llamacpp.py
python3 bench/llamacpp_combo.py
```
