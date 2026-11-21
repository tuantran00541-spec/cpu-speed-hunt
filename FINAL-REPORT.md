# CPU Speed Hunt — FINAL REPORT

**Model:** base `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` (no LoRA — runtime comparison only).
**Test set:** 3 images (TMPT ch155 crop, TMPT ch136 crop, romcom synthetic). Prompt: `Dịch chữ trong ảnh sang tiếng Việt.` Greedy, max_new_tokens=100, 1 warmup + 3 timed runs per image, median.
**Runs:** GitHub Actions, free `ubuntu-latest`, 4 vCPU each, repo `tuantran00541-spec/cpu-speed-hunt`, workflow `bench.yml`.
**Final round:** run **44895867277** (commit 0d89847), completed/failure, **8/9 green**. Prior rounds: R1 run 44893753958 (1/9), R2 run 44894837776 (5/9).

## Ranked table (round 3, medians)

| runtime | s/image (median) | tok/s | peak RAM | output correct? | notes |
|---|---|---|---|---|---|
| **llama-cpp-combo** | **0.58** (0.35 / 0.89 / 0.49) | 31.5 | 1.3 GB | ✅ all 3 | Q4_K_M + mmproj F16, `-fa`, q8_0 KV cache, batch/ubatch 1024, `--cache-prompt --cache-reuse 256`. Load 1.0s. Beats base q4km by ~9%. |
| **llama-cpp-q4km** | **0.64** (0.38 / 1.03 / 0.51) | 29.4 | 1.6 GB | ✅ all 3 | Same GGUF, plain flags. Load 1.0s. |
| pytorch-eager | 2.30 (1.28 / 2.99 / 2.64) | 9.1 | 2.5 GB | ✅ all 3 | torch **2.8.0**, transformers 5.23.0, bf16, SDPA. Load 17.8s. |
| openvino-int8 | 2.32 (1.20 / 2.48 / 3.27) | 9.8 | 3.9 GB | ✅ all 3 | optimum-intel 2.0, OVModelForVisualCausalLM INT8. Load 4.0s. Tie with eager within runner noise. |
| openvino-int4 | 4.61 (3.80 / 4.27 / 5.76) | 22.4 | 3.4 GB | ❌ garbage | Echoes the English source text / loops "THẞ". Per-channel INT4 (group-size -1) destroys quality. tok/s meaningless. |
| openvino-combo | 6.07 (5.02 / 5.07 / 8.11) | 16.9 | 2.9 GB | ❌ garbage | INT4 + MTP draft (`openvino_mtp_model.xml` present, draft loaded, prefix caching OFF). MTP works; INT4 weights sink it. |
| pytorch-compile | 9.21 (4.71 / 9.28 / 13.65) | 2.5 | 2.6 GB | ✅ all 3 | torch.compile mode=default (Inductor). Load 26.9s, compile 4.7s. 4× SLOWER than eager on torch 2.8. |
| pytorch-combo | 39.75 (21.43 / 49.36 / 48.47) | 0.5 | 5.9 GB | ✅ all 3 (slow) | IPEX (bf16, weights_prepack=False) + torch.compile + 4 threads. INT8 dropped in smoke (Float vs BFloat16). 17× slower than eager — stacking hurts. Load 51.3s, compile 23.1s. |
| pytorch-ipex | FAILED | — | — | ❌ | `RuntimeError` in Qwen3.5 linear-attention `causal_conv1d_fn`: `Expected 7-dimensional input for 7-dimensional weight [6144, 1, 1, 4, 8, 16, 2], but got 3-dimensional input [1, 6144, 113]`. IPEX's default weight prepacking repacks the conv1d weight into a layout transformers' conv path cannot consume. IPEX is incompatible with this model's linear-attention path. |

**Combo verdict:** llama-cpp-combo beats the best single method (llama-cpp-q4km, 0.64s) by **~9% (0.58s vs 0.64s)**. PyTorch combo does NOT beat its best single (eager 2.30s) — it is 17× worse. OpenVINO combo does NOT beat INT8 — it is garbage (INT4).

## Winner: llama.cpp — ~100× faster than the current ~60s/image transformers baseline

- **0.58s/image** vs ~60s today. 31 tok/s, 1.3 GB RAM, 1.0s load, correct Vietnamese on all 3 test images.
- Combo recipe (all inside llama-server, one runtime): Q4_K_M text GGUF + F16 mmproj, `-fa` (flash-attn), `--cache-type-k q8_0 --cache-type-v q8_0`, `--batch-size 1024 --ubatch-size 1024`, `--cache-prompt --cache-reuse 256`, `-t 4 --threads-batch 4`, `-c 2048`.
- Honest caveat: the combo's median benefits from prompt-cache reuse on timed runs 2–3 (same image sent 4×); a fresh-image-every-time workload would sit closer to base q4km's 0.64s. Still the fastest by 4×.

## Recommendations

**(a) User's Windows laptop:** llama.cpp llama-server with the combo flags above. Single static binary, no Python/torch stack, 1.3 GB RAM. Build via CMake Release (or a release binary), convert the base model to GGUF + quantize Q4_K_M + export mmproj F16, serve with the combo flags. His laptop has more cores than the 4-vCPU runner, so expect faster than 0.58s/image. No GPU needed.

**(b) This VM:** same llama.cpp setup — a static binary dodges the entire torch-version roulette below. If he insists on staying in Python/transformers: **pin `torch==2.8.0` + `torchvision==0.23.0`**, eager, ~2.3s/image.

## Biggest gotcha found: torch version matters more than the optimization

- Round 2 (torch **2.15.0**, dragged in by unpinned torchvision): eager **36.9s**/image, compile 2.03s.
- Round 3 (torch **2.8.0**, pinned): eager **2.30s**/image (16× faster), compile 9.21s (4.5× slower).
- Same script, same protocol, same transformers 5.23.0. Never let an unpinned torchvision upgrade torch. Pin 2.8.0.

## Dropped candidates (verified, not guessed)

- **ONNX Runtime:** no `ORTModelForVisualCausalLM` / `ORTModelForImageTextToText` in optimum-onnx (2,112 lines checked).
- **vLLM GGUF:** does not support `qwen35` (`Unknown gguf model_type: qwen35`, issues #36456/#38122). CPU HF-weights path unproven.
- **SGLang CPU:** needs Intel AMX, no Qwen3.5-VL listing — effectively out.
- **ncnn / MNN:** no usable Linux-CPU path for this model.

## Caveats

- Runner CPUs vary per job (Intel Xeon Platinum 8573C / Xeon 6975P-C; AMD EPYC 9V45 / 9V74 / 7763; all 4 vCPU) — cross-runtime comparisons carry ~±1.5× noise. The llama.cpp win (4× over next) is far outside noise; openvino-int8 vs pytorch-eager (2.32 vs 2.30) is a tie.
- **Base model only.** The fine-tuned quality lives in the r14 LoRA; llama.cpp runs the BASE model until the adapter is merged/converted. Quality on 3 images only — no broad Q4_K_M vs FP16 quality check.
- Speculative decoding not applicable to llama.cpp here: no smaller Qwen3.5 than 0.8B exists and the checkpoint ships no MTP draft head.
- IPEX targets Intel AMX; runners are mixed AMD/Intel — IPEX had little upside even before the incompatibility.

## OPEN items / obstacles

1. **Adapter merge (follow-up, NOT done):** merge the r14 LoRA into base weights and convert to GGUF Q4_K_M (or convert the PEFT adapter to GGUF LoRA format for llama.cpp `--lora`). Until this is done, the llama.cpp path runs base-model quality, not fine-tuned quality. Needs a user decision on which adapter (r14a/b/c, eval pending).
2. **Q4_K_M quality check:** compare llama.cpp Q4_K_M output vs FP16 eager on 20+ images before replacing the pipeline. Only 3 images checked so far.
3. **Fresh-image combo measurement:** re-run llama-cpp-combo with a different image per timed run to isolate the `--cache-reuse` contribution.
4. **INT4 group-size 128 retry:** round-1 NNCF export failed (`InvalidGroupSizeError`); per-channel INT4 is degenerate. Low priority — llama.cpp already wins.
5. **IPEX:** closed as incompatible (see table). Do not retry without a new IPEX/transformers version that fixes the linear-attention conv path.
6. **Windows-laptop native test:** verify the combo flags + speed on his actual machine (core count, AVX2/AVX-512) — GH runner numbers are a proxy.
7. **Round-2 vs round-3 torch-version flip** is measured but the mechanism was not isolated (no per-kernel profile). Pin 2.8.0 and move on.
8. **Q8_0 test CLOSED 2026-11-21** (run 44897505997): Q8_0 not worth it vs Q4_K_M for this 0.8B — not faster, no quality gain beyond run-to-run noise, +52% disk. Keep Q4_K_M. The 20+ image Q4_K_M-vs-FP16 check (item 2) is still open and is the right way to settle quantization quality.

## Q8_0 vs Q4_K_M (user-ordered follow-up, 2026-11-21 18:05)

User worried Q4_K_M is too lossy and asked for Q8_0. Same protocol, same combo flags, same 3 images. Run **44897505997** (commit 24ddae1), job `bench (llama-cpp-q8)` success; result JSON `results-q8/results-llama-cpp-q8.json`. The same run also re-ran Q4 combo (mean of medians 0.381s — faster than round 3's 0.577s; runner lottery, see caveat).

| | Q8_0 combo | Q4_K_M combo (round 3) | Q4_K_M combo (same run) |
|---|---|---|---|
| s/image (mean of per-image medians) | **0.663** (0.755 / 0.820 / 0.413) | 0.577 (0.350 / 0.890 / 0.491) | 0.381 (0.233 / 0.630 / 0.280) |
| tok/s (per image) | 14.6 / 37.8 / 33.9 | 31.4 / 32.6 / 30.6 | 47.2 / 49.2 / 46.4 |
| peak RAM | 1,366 MB | 1,304 MB | 1,304 MB |
| load_s | 1.01 | 1.01 | 1.01 |
| GGUF file size | 795 MB | 522 MB | 522 MB |

Outputs (verbatim): img1 identical on all three runs ("Anh ấy không tồn tại trong thế giới này."). img2: Q8 "…sức mạnh vô hạn của Tai Chi…" (31 tok) vs Q4 round-3 "…sức mạnh của Tai Chi…" (29 tok) vs Q4 same-run "…sức mạnh vô tận của Tai Chi…" (31 tok) — the "vô hạn/vô tận" nuance appears in same-run Q4 too, so it is run-to-run variation, not a Q8 gain. img3: all three correct; Q4 wording marginally more natural. All three runs still say "Tai Chi" (not "Thái Cực") and "bạn" (not "ngươi") — base-model register gaps, unchanged by quantization.

**Verdict: Q8_0 is NOT worth it for this 0.8B model.** Not faster (equal-to-slower within runner noise), no quality gain beyond run-to-run noise on 3 images, +52% disk (+273 MB) for +5% RAM. Keep Q4_K_M. Two caveats: (1) img1's Q8 median (0.755s) is noise-contaminated — its 3 runs spread 1.077/0.755/0.323s and server timings show decode ramping 11.5→17→39 tok/s (shared-runner contention on the EPYC 7763 host); steady-state Q8 img1 is ~0.32s. (2) Greedy outputs are NOT reproducible across runs: round-3 Q4 vs same-run Q4 differ on 2/3 images (llama.cpp master moved + different host CPU: EPYC 7763 vs 9V74) — future A/B comparisons must be same-run, same-machine.

## Sources

- All round-3 result JSONs: `~/workspace/cpu-speed-hunt/results-round3/` (8 files, one per green runtime).
- Bench scripts: `~/workspace/cpu-speed-hunt/bench/` (llamacpp.py, llamacpp_combo.py, openvino_*.py, pytorch_*.py).
- Research notes: `~/workspace/cpu-speed-hunt/research.md` (17 sources).
- Repo: `tuantran00541-spec/cpu-speed-hunt`, workflow `bench.yml`, run 44895867277 (round 3), 44894837776 (round 2), 44893753958 (round 1).
- IPEX failure log: job 136821265519 (round 3, pytorch-ipex).
