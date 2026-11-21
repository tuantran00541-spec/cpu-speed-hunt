# PyTorch-lane job steps (GitHub Actions, ubuntu-latest)

How to run the four `bench/pytorch_*.py` scripts on a fresh
`ubuntu-latest` runner (4 vCPU, ~16 GB RAM, Python 3.12). The model
downloads from Hugging Face inside the job — no token, no secrets, no
paid services. Each script writes `./results-<runtime>.json` in the
working directory (repo root).

## Version pins — why these exact versions

| Package | Pin | Reason |
|---|---|---|
| `torch` | `==2.8.0` (CPU wheel) | IPEX 2.8.0 is the newest IPEX release (checked 2026-11-21) and IPEX must match the torch major.minor version exactly. torch 2.15 exists, but no IPEX 2.15 exists. One torch version for all four scripts keeps the comparison fair. |
| `torchvision` | CPU wheel (matching torch) | **Required — run 1 failed without it.** transformers 5.x `Qwen3VLVideoProcessor` requires torchvision at `AutoProcessor.from_pretrained` time (`ImportError` otherwise). Install from the CPU index right after torch. |
| `intel-extension-for-pytorch` | `==2.8.0` | Matches torch 2.8.0. |
| `transformers` | `>=5.2.0,<6` | **Hard requirement.** This model is `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`). transformers 4.57.1 does NOT recognize `qwen3_5` (`ValueError: ... Transformers does not recognize this architecture`); native support needs `transformers>=5.2.0`. Do not "fix" this by pinning 4.57. |
| `accelerate`, `Pillow` | unpinned | `device_map="cpu"` needs accelerate; images are loaded with Pillow. |

Install order matters: install the CPU `torch` FIRST, then
`transformers`. If transformers is installed first (from PyPI), pip can
pull the default CUDA torch build and replace the CPU wheel.

## Example workflow (one job per script; run them as separate jobs)

```yaml
name: cpu-speed-pytorch
on:
  workflow_dispatch:

jobs:
  # Duplicate this job 4x, changing only `script` (and the artifact name).
  # script: pytorch_eager | pytorch_compile | pytorch_ipex | pytorch_combo
  bench:
    runs-on: ubuntu-latest
    timeout-minutes: 180        # compile/combo need the headroom; see below
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Show CPU (record what the runner actually is)
        run: |
          grep -m1 'model name' /proc/cpuinfo
          lscpu | grep -E 'Model name|CPU\(s\)|Flags' | head -5
          g++ --version | head -1
          python3 --version

      - name: Install PyTorch CPU 2.8.0
        run: |
          python -m pip install --upgrade pip
          python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
          python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu

      - name: Install transformers + helpers
        run: |
          python -m pip install "transformers>=5.2.0,<6" accelerate Pillow

      # --- ONLY for pytorch_ipex and pytorch_combo (skip for eager/compile) ---
      - name: Install IPEX 2.8.0 (matching torch 2.8.0)
        run: |
          python -m pip install "intel-extension-for-pytorch==2.8.0" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        # Fallback if the Intel index has no wheel for this runner:
        #   python -m pip install "intel-extension-for-pytorch==2.8.0"
        # (2.8.0 is also published on PyPI.)

      - name: Verify installed versions
        run: |
          python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('threads', torch.get_num_threads())"
          python -c "import intel_extension_for_pytorch as ipex; print('ipex', ipex.__version__)"   # ipex/combo jobs only

      - name: Run bench (from repo root)
        run: python bench/pytorch_eager.py
        # pytorch_compile.py / pytorch_ipex.py / pytorch_combo.py respectively

      - name: Upload result
        uses: actions/upload-artifact@v4
        with:
          name: results-pytorch-eager
          path: results-pytorch-eager.json
          # results-pytorch-compile.json / results-pytorch-ipex.json /
          # results-pytorch-combo.json respectively
```

## Per-script commands (if running by hand instead of the workflow)

All from the repo root, after the install steps above:

```bash
python bench/pytorch_eager.py    # -> results-pytorch-eager.json
python bench/pytorch_compile.py  # -> results-pytorch-compile.json
python bench/pytorch_ipex.py     # -> results-pytorch-ipex.json
python bench/pytorch_combo.py    # -> results-pytorch-combo.json
```

IPEX is required only for `pytorch_ipex.py` and `pytorch_combo.py`.
`pytorch_ipex.py` FAILS LOUDLY (nonzero exit, no eager fallback) if IPEX
is missing, mismatched, or `ipex.optimize()` raises — that is the spec.
`pytorch_combo.py` instead drops the failed piece, records why in
`notes`, and keeps the fastest working subset.

## Expected runtimes / timeouts

- Model download inside the job: ~1.6 GB (single safetensors file).
- `pytorch_eager.py`: expect tens of minutes (0.8B VLM, 4 CPU cores,
  3 images × (1 warmup + 3 timed) × up to 100 tokens). `timeout-minutes: 120`
  is ample.
- `pytorch_compile.py`: Inductor compilation happens on the first
  generate call and can take many minutes on CPU on top of the eager
  time. `timeout-minutes: 180`. `compile_s` in the JSON is first-call
  wall time (compile + one generation), not pure compile time.
- `pytorch_ipex.py`: IPEX optimize adds ~1–2 min (it deep-copies the
  model). `timeout-minutes: 120`.
- `pytorch_combo.py`: worst case it loads the model twice (once for the
  INT8 attempt, once fresh for the IPEX+compile path) plus a compile.
  `timeout-minutes: 180`.

## Known open issues / things to watch in the first run

1. **IPEX on a non-Intel runner.** `ubuntu-latest` runners are often
   AMD EPYC. IPEX runs on any AVX2 CPU, but its AMX/VNNI fast paths are
   Intel-only, so expect smaller (possibly ~zero) IPEX gains on AMD.
   Record the CPU from the `lscpu` step alongside the numbers.
2. **torch.compile vs the model's `linear_attention` layers.** The
   Qwen3.5 text config is mostly `linear_attention` layers
   (Gated-DeltaNet-style). Inductor may graph-break around them, which
   caps compile gains; `suppress_errors` is intentionally left `False`
   so a hard Dynamo failure raises instead of silently going eager.
   If the compile job is much slower than eager with no tok/s gain,
   that is the likely mechanism.
3. **transformers 5.x + torch 2.8.0 coexistence.** Both pins were
   verified to exist (PyPI, 2026-11-21) but the pair was NOT executed
   end-to-end here; if pip's resolver complains, that is the first
   thing to report back — do not silently loosen the pins.
4. **Combo INT8 quality.** The combo's INT8 smoke test only checks
   "runs and output is non-empty". If `results-pytorch-combo.json`
   shows much higher tok/s but its `output` text is degraded vs
   `results-pytorch-eager.json`, INT8 bought speed with quality — the
   script records that it kept INT8, and the call is the reader's.
5. **Disk/RAM.** torch CPU wheel (~800 MB) + model (~1.6 GB) +
   Inductor cache fit easily in the runner's disk/RAM; IPEX's model
   deepcopy transiently holds ~2× weights (~3.2 GB) — fine on 16 GB.

## Sources checked (2026-11-21)

- `torch.compile` signature/modes: PyTorch docs,
  `https://docs.pytorch.org/docs/main/generated/torch.compile.html`
  (`mode="default"`, backend `"inductor"`, `max-autotune` NOT used).
- `ipex.optimize` signature: Intel IPEX API docs,
  `https://intel.github.io/intel-extension-for-pytorch/cpu/1.12.100+cpu/tutorials/api_doc.html`
  (`ipex.optimize(model, dtype=torch.bfloat16)`; `weights_prepack=False`
  required under TorchDynamo per the IPEX examples page).
- `ipex.llm.optimize` replaces `optimize_transformers` for LLMs, but its
  supported list (Llama, GPT-J, OPT, Falcon, ...) does not include
  Qwen3.5-VL — plain `ipex.optimize` is used instead.
- `AutoModelForImageTextToText` + `processor.apply_chat_template(...)`
  + `model.generate(..., max_new_tokens=...)` pattern: Hugging Face
  image-text-to-text docs and the Qwen3-VL README.
- Model repo verified public: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`
  (`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, single
  safetensors file, Apache-2.0).
- Latest versions via PyPI JSON (2026-11-21): torch 2.15.0, IPEX 2.8.0,
  transformers 5.23.0 — hence the torch==2.8.0 / IPEX==2.8.0 pairing.
- `transformers>=5.2.0` requirement for `qwen3_5`: multiple reports
  that 4.57.1 fails to recognize the architecture and that upgrading
  to `>=5.2.0` fixes it.
