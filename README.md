# CPU speed hunt — Qwen3.5-0.8B vision

Find the fastest CPU setup for running `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`
(image in → Vietnamese out) by benchmarking every viable CPU runtime in
parallel on free GitHub Actions runners.

- Prompt: `Dịch chữ trong ảnh sang tiếng Việt.` — greedy, `max_new_tokens=100`
- Test set: 3 images in `assets/` (2 TMPT manga crops + 1 romcom test image)
- Protocol: load once, 1 warmup + 3 timed generations per image, median
- Runtimes: PyTorch eager / torch.compile / IPEX / PyTorch combo,
  llama.cpp GGUF Q4_K_M + mmproj / llama.cpp combo,
  OpenVINO INT8 / INT4 / OpenVINO combo
- `research.md` — verified candidate table (what is supported, with sources)
- `bench/` — one bench script per runtime + job-steps docs; dropped
  candidates documented in `bench/onnxruntime_DROPPED.md`
- Workflow: `.github/workflows/bench.yml` (manual dispatch, 9 matrix jobs)

Results land as `results-<runtime>.json` artifacts. Ranked table goes in
this README after the runs complete.
