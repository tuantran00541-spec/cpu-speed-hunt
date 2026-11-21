# OpenVINO job steps — fresh `ubuntu-latest` GitHub Actions job

Everything below was verified against current docs on 2026-11-21
(see "Sources" at the bottom). No secrets, no keys, no paid services.
Model is public on Hugging Face; the job only downloads it.

Model: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` (~1.6 GB download).
Runner: `ubuntu-latest` (4 vCPU, ~16 GB RAM), Python 3.12.
Scripts run from the repo root; model dirs and `results-*.json` land in
the repo root (cwd).

## 0. Job skeleton (YAML sketch — adapt to the real workflow file)

```yaml
jobs:
  openvino-bench:
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      # ... install (1), exports (2), checks (3), bench runs (4) below ...
      - uses: actions/upload-artifact@v4
        with:
          name: openvino-results
          path: results-*.json
```

## 1. Install (pinned — do not float these)

```bash
python -m pip install --upgrade pip

# CPU-only torch first (wheel index from the official OpenVINO Qwen3.5
# model-card install block), so pip never pulls a CUDA torch.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Pinned stack, exact versions from the Optimum-Intel v2.2.0 release
# announcement ("Get started" install block):
#   optimum-intel==2.2.0, openvino==2026.4, openvino-genai==2026.4,
#   openvino-tokenizers==2026.4, nncf==3.4
pip install \
  "optimum-intel==2.2.0" \
  "openvino==2026.4" \
  "openvino-genai==2026.4" \
  "openvino-tokenizers==2026.4" \
  "nncf==3.4" \
  "transformers==5.2.0" \
  Pillow huggingface_hub
```

Why `transformers==5.2.0` (not latest): optimum-intel 2.2.0's release
notes cap the Qwen3.5 architecture at transformers 5.2.0
("Qwen3_5 | 5.2.0" in the max-version table), and the official
OpenVINO/Qwen3.5-9B-int8-ov card installs exactly `transformers==5.2.0`.
A newer transformers will make the Qwen3.5 export fail its version gate.

Sanity check:

```bash
pip list 2>/dev/null | grep -i -E "optimum-intel|openvino|nncf|transformers|^torch |pillow"
python -c "import optimum.intel, openvino, openvino_genai, nncf, transformers; print('imports ok')"
```

## 2. Export the base model to OpenVINO IR (INT8 and INT4)

`--task image-text-to-text` is **required**: the optimum-intel exporter
only supports `image-text-to-text` for `qwen3_5` (stated in the
`ValueError` quoted in optimum-intel PR #1634). `--trust-remote-code`
is included because the Morteza89 Qwen3.5 INT4 export command uses it;
harmless for this standard checkpoint.

```bash
# INT8 — optimum-cli default is INT8_ASYM per-channel (group_size -1),
# matching the official OpenVINO/Qwen3.5-9B-int8-ov quantization params.
optimum-cli export openvino \
  --model huihui-ai/Huihui-Qwen3.5-0.8B-abliterated \
  --task image-text-to-text \
  --trust-remote-code \
  --weight-format int8 \
  ./ov_qwen3.5_0.8b_int8

# INT4 — group-size -1 (per-channel). --group-size 128 was the original
# recipe but NNCF rejects it on this checkpoint (InvalidGroupSizeError:
# rotary_emb MatMul channel size 1); the CLI has no
# group-size-fallback-mode flag, so per-channel is the fix (run 1).
optimum-cli export openvino \
  --model huihui-ai/Huihui-Qwen3.5-0.8B-abliterated \
  --task image-text-to-text \
  --trust-remote-code \
  --weight-format int4 \
  --group-size -1 \
  --ratio 1.0 \
  ./ov_qwen3.5_0.8b_int4
```

Export-time expectation (ESTIMATE, not measured): each export downloads
~1.6 GB, converts to IR and runs NNCF weight compression on 4 vCPU —
expect roughly **5–20 minutes per export**. Do not put a tight
`timeout-minutes` on the export steps; the 180-minute job timeout above
covers install + both exports + all three bench runs with headroom.
Disk: keep several GB free; each IR dir is on the order of ~1 GB
(INT8) / ~0.5 GB (INT4) plus tokenizer/vision files. `/tmp` staging is
fine at this model size (the known `/tmp`-tmpfs blow-up only bites
27B-scale exports).

## 3. Verify the exports before benching

```bash
ls ./ov_qwen3.5_0.8b_int8 ./ov_qwen3.5_0.8b_int4
# Each dir must contain openvino_model.xml/.bin, openvino_tokenizer/
# detokenizer files, processor/tokenizer configs, and the vision parts.

# MTP head check — decides what the combo script can do:
ls ./ov_qwen3.5_0.8b_int4/openvino_mtp_model.xml \
  && echo "MTP present: combo will use speculative decoding" \
  || echo "MTP ABSENT: combo will fall back to prefix-caching (by design, see bench/openvino_combo.py)"
df -h .   # confirm disk headroom
```

## 4. Run the bench scripts (repo root, in order)

```bash
python bench/openvino_int8.py    # -> ./results-openvino-int8.json
python bench/openvino_int4.py    # -> ./results-openvino-int4.json
python bench/openvino_combo.py   # -> ./results-openvino-combo.json
```

Each script is standalone (no shared bench module), follows the same
protocol (1 warmup + 3 timed generations per image, median; greedy;
max_new_tokens=100), and prints per-image median / tok/s / peak RSS.
Each run takes the model dir from `./ov_qwen3.5_0.8b_int8` /
`./ov_qwen3.5_0.8b_int4` in cwd. Optional overrides (defaults are the
spec paths): `--model-dir <dir> --output <file>`.

Then upload `results-*.json` as a workflow artifact (see skeleton).

## 5. What is deliberately NOT in this job

- **No ONNX Runtime lane.** `bench/onnxruntime.py` was not written:
  the optimum-onnx modeling reference lists no
  `ORTModelForVisualCausalLM` / `ORTModelForImageTextToText`, and the
  closest classes (`ORTModelForVision2Seq`, `ORTModelForPix2Struct`)
  are encoder–decoder / single-architecture classes that do not support
  Qwen3.5 VL image input. Full verdict:
  `bench/onnxruntime_DROPPED.md`. Do not add an ONNX export step
  without re-verifying the optimum-onnx docs first.
- **No LoRA, no other model.** All three scripts bench the base
  checkpoint only.
- **No `openvino` nightly / `--pre` wheels.** The 2026.4 release pins
  above are the supported set for optimum-intel 2.2.0; nightly is only
  needed for unreleased architectures (not Qwen3.5).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Export fails with a transformers version error for `qwen3_5` | `transformers` drifted past 5.2.0 — re-pin `transformers==5.2.0` and reinstall. |
| Export fails: "only supports the tasks image-text-to-text for qwen3_5" | `--task image-text-to-text` was omitted — it is mandatory. |
| `openvino_mtp_model.xml` missing after INT4 export | Combo still runs (prefix-caching fallback); check the export log for an MTP warning. The abliterated checkpoint may have dropped the MTP head — record it, do not hand-craft a draft model. |
| Combo: `VLMPipeline` rejects `INFERENCE_NUM_THREADS` | Script retries without thread kwargs automatically and records it in `notes`. |
| OOM / disk-full during export | Check `df -h`; the two IR dirs plus the ~1.6 GB HF cache need several GB. Do not shrink the model — report it. |
| `results-*.json` missing after a run | The script exits non-zero if its model dir is absent; read the step log — the export step above it is the usual culprit. |

## Sources (checked 2026-11-21)

- Optimum-Intel v2.2.0 & OpenVINO GenAI 2026.4.0 announcement (pins +
  MTP/DFlash for Qwen3.5): <https://huggingface.co/blog/echarlaix/optimum-intel-v22>
- optimum-intel 2.2.0 release notes (transformers max-version table,
  recommended versions): <https://github.com/huggingface/optimum-intel/releases>
- OpenVINO/Qwen3.5-9B-int8-ov model card (OVModelForVisualCausalLM
  snippet, GenAI+MTP snippet, install pins, INT8 params):
  <https://huggingface.co/OpenVINO/Qwen3.5-9B-int8-ov/blob/main/README.md>
- Optimum-Intel supported-models list (Qwen3.5 incl. MTP head):
  <https://huggingface.co/docs/optimum-intel/openvino/models.md>
- OpenVINO GenAI MTP guide (versions with MTP export support):
  <https://medium.com/openvino-toolkit/accelerating-llm-inference-with-multi-token-prediction-mtp-and-openvino-407fa299e57f>
- OpenVINO GenAI VLMPipeline API (constructor kwargs = device
  properties; `generate(prompt, images=..., ...)`):
  <https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.VLMPipeline.html>
- optimum-onnx modeling reference (ORT class list — no visual VLM
  class): <https://huggingface.co/docs/optimum-onnx/en/onnxruntime/package_reference/modeling>
