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
2. **Q4_K_M quality check: DONE 2026-11-21** (run 44902453529; see FP16 section below) — FP16 wins 5–0 over Q4 on 21 images, Q4 never wins; both bad on murim terms (base-model gap, not quantization). Keep Q4 for speed; re-check after the r14 adapter is merged into GGUF.
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

## FP16 vs Q4_K_M quality check (user-ordered follow-up, 2026-11-21 18:25)

Settles OPEN item 2: compare Q4_K_M against FP16 eager on 20+ images before replacing the pipeline. 21 images: the 3 standard hunt images + 18 unseen TMPT hard-test crops (5 short, 4 long, 4 questions, 3 honorific, 2 medium). All 18 crops verified on disk via the TMPT manifest; EN + trusted VI in `results-fp16/reference.json` (local only, not committed).

Runs: **44902453529** (commit 11c1718), jobs `bench (pytorch-fp16)` and `bench (llama-fp16-q4km)` both completed/success; whole run 11/12 green (only known-bad pytorch-ipex failed). FP16 = PyTorch eager, torch 2.8.0 + torchvision 0.23.0 pinned, bf16. Q4 = llama.cpp combo, exact round-3 winning flags (Q4_K_M + mmproj F16, `-fa`, q8_0 KV, batch/ubatch 1024, `--cache-prompt --cache-reuse 256`). Same images, same prompt, greedy, max 100 tokens, warmup + 3 timed runs, first timed run's output scored. Result JSONs: `results-fp16/artifacts-fp16/results-pytorch-fp16.json`, `results-fp16/artifacts-q4km/results-llama-fp16-q4km.json`.

First attempt (run 44899224189) was cancelled: its `pytorch-fp16` job sat 47 min in the bench step (round 3's whole run took 10 min); install steps had all completed in <1 min, so it was stuck inside the bench script (likely a slow/contended runner or HF download hang — logs are unavailable while a job runs). Nothing else was lost: 11/12 jobs were already done, including the Q4 job. Re-dispatch (run 44902453529) finished normally.

| | FP16 eager | Q4_K_M combo |
|---|---|---|
| s/image (mean of per-image medians) | **2.57** (median 2.01, range 0.93–5.19) | **0.46** (median 0.29, range 0.12–2.18) |
| tok/s (mean) | 7.7 | 41.8 |
| load_s | 17.6 | 1.0 |
| peak RSS (max over images) | 2,582 MB | 2,013 MB |
| total tokens over 21 images | 421 | 420 |
| runner CPU | Intel Xeon 6973P-C | AMD EPYC 7763 |

Speed: Q4 is **5.6× faster** than FP16 on this set (0.46s vs 2.57s mean). Note the Q4 peak RSS here (2.0 GB) is higher than round 3's 1.3 GB — same flags, different host/measurement run; treat 1.3–2.0 GB as the range, not a regression.

### Verdict: Q4_K_M is NOT equal to FP16 on 21 images — FP16 wins 5–0, 16 ties

Scoring vs trusted VI: correct / paraphrase-right (meaning right, wording/register/term off) / wrong (mistranslated term, wrong register, dropped clause, hallucination, untranslated). **FP16: correct 1, paraphrase 12, wrong 8. Q4: correct 1, paraphrase 7, wrong 13.** Head-to-head (strict — better category wins): **FP16 5 wins, Q4 0 wins, 16 ties.** FP16 was also marginally closer on 6 of the ties (fp16_05, 06, 09, 12, 16, 18).

But read the absolute numbers too: only 1/21 fully correct on EITHER side. Both fail the murim terminology systematically — Taiji never becomes Thái Cực, sát khí/địch ý never land, PHONG MA KIẾM / THẤT SÁT TỬ / TAIJI WISDOM SWORD are botched by both, both use modern bạn/tôi where the trusted text uses ngươi/ta. That is the BASE model, not quantization: the terminology lives in the r14 LoRA / TMPT data, not in FP16 vs Q4. Quantization costs a little on hard lines; the base model costs a lot everywhere. The fine-tuned adapter merged into GGUF needs its own check before any pipeline switch — this test does not cover it.

The 5 FP16 wins: fp16_01 (Q4 drops 'lethal combat', hallucinates 'loại bỏ'), fp16_07 (Q4 loops to 100 tok — see caveat), fp16_10 (Q4 flips meaning: 'không cần thời gian'), fp16_13 (Q4 turns a proposal into a 'when' question), fp16_15 (Q4: 'LORD GON MA' → 'Chúa Giê-su'). Q4 never wins a single image.

### Per-image table

| image | EN | trusted VI | FP16 output | Q4 output | verdict |
|---|---|---|---|---|---|
| img1_tmpt_ch155.png [STD-SHORT] | HE DOESN'T EXIST IN THIS WORLD. | HẮN KHÔNG CÒN TỒN TẠI TRONG THẾ GIỚI NÀY NỮA. | Anh ấy không tồn tại trong thế giới này. (1.74s) | Anh ấy không tồn tại trong thế giới này. (0.27s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| img2_tmpt_ch136.png [STD-LONG] | THAT'S BECAUSE YOU CAN USE TAIJI'S LIMITLESS POWERS WHEN YOUR OPPONENT IS HOSTILE OR HAS A KILLING INTENT. | KHI ĐỐI ĐẦU VỚI SÁT KHÍ VÀ ĐỊCH Ý CỦA ĐỐI THỦ, THÁI CỰC CÓ THỂ PHÁT HUY SỨC MẠNH GẦN NHƯ VÔ HẠN. | Vì vậy, bạn có thể sử dụng sức mạnh vô hạn của Tai Chi khi đối thủ của bạn là thù địch hoặc có ý định tiêu diệt. (4.05s) | Đó là vì bạn có thể sử dụng sức mạnh vô tận của Tai Chi khi đối thủ của bạn là thù địch hoặc có ý định sát hại. (0.7s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| img3_romcom.png [STD-ROMCOM] | I'VE LIKED YOU SINCE THE DAY WE FIRST MET. | (no reference — judge by ear) | Tôi đã yêu thích bạn từ ngày chúng tôi lần đầu gặp. (3.01s) | Tôi đã yêu thích bạn từ ngày đầu tiên chúng tôi gặp nhau. (0.37s) | FP16 correct / Q4 correct — tie |
| fp16_01.png [SHORT ch184 idx10386] | OF COURSE, WE'LL HAVE TO PERMIT LETHAL COMBAT. | VÀ HÃY CHO PHÉP CHÚNG CHIẾN ĐẤU SINH TỬ. | Đúng rồi, chúng ta sẽ phải cho phép chiến tranh tàn khốc. (2.34s) | Theo tôi, chúng tôi sẽ cần phép loại bỏ. (0.28s) | FP16 paraphrase-right / Q4 wrong — FP16 wins |
| fp16_02.png [SHORT ch131 idx7095] | I JUST THINK IT'S A WASTE OF TIME... | TA NGHĨ ĐÓ CHỈ LÀ LÃNG PHÍ THỜI GIAN... | Tôi chỉ nghĩ nó là một lãng phí thời gian... (1.75s) | Tôi chỉ nghĩ nó là thời gian lãng phí... (0.26s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| fp16_03.png [SHORT ch103 idx5889] | I DON'T KNOW HOW DID YOU KNOW THAT, | DÙ KHÔNG BIẾT NGƯỜI BIẾT ĐƯỢC BẰNG CÁCH NÀO, | Tôi không biết cách bạn đã biết điều đó, (1.8s) | Tôi không biết cách bạn đã biết điều đó, (0.26s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| fp16_04.png [SHORT ch139 idx7726] | I CAN THINK OF ONLY ONE POSSIBLE EXPLANATION... | VẬY THÌ CHỈ CÓ THỂ LÀ... | Tôi có thể nghĩ đến chỉ một cách giải thích duy nhất... (2.05s) | Tôi có thể chỉ ra một cách giải thích duy nhất... (0.32s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| fp16_05.png [SHORT ch152 idx8539] | YOU WILL MEET ME AGAIN. | RẰNG CÁC NGƯỜI... SẼ LẠI GẶP TA LẦN NỮA. | Bạn sẽ gặp lại tôi lần nữa. (1.45s) | Bạn sẽ gặp lại tôi. (0.19s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| fp16_06.png [LONG ch154 idx8636] | IN THE PAST, IT WAS AN ABILITY THAT COULD ONLY BE UNLEASHED IF I HAD THE WIND-CLOUD ULTIMATE DEMONIC SWORD. | TRONG QUÁ KHỨ, CHỈ KHI SỞ HỮU PHONG MA KIẾM THÌ MỚI CÓ THỂ THI TRIỂN ĐƯỢC NÓ. | Trong quá khứ, nó là một khả năng mà chỉ có thể giải phóng nếu tôi có vũ khí cuối cùng của gió mây. (3.67s) | Trong quá khứ, nó là một khả năng mà chỉ có thể được giải phóng nếu tôi có chiếc kiếm thần linh cuối cùng. (0.63s) | FP16 wrong / Q4 wrong — tie |
| fp16_07.png [LONG ch158 idx8995] | INSIDE, THERE ARE LIKELY BEASTS WITH STRENGTH SEVERAL TIMES THAT OF A HUMAN, AFFECTED BY VARIOUS MYSTICAL LAWS, AS WELL AS THINGS LIKE JIANGSHI. | TRONG ĐÓ CÓ NHỮNG LOÀI THÚ DỊ DẠNG, SỨC MẠNH GẤP NHIỀU LẦN CON NGƯỜI, CẢ NHỮNG THỨ GIỐNG NHƯ CƯƠNG THI. | Trong đó, có lẽ có những con thú có sức mạnh nhiều lần so với con người, bị ảnh hưởng bởi nhiều quy luật mystic, như cũng như các vật thể như Jiangshi. (5.19s) | Ở trong, có lẽ có những thú vật với sức mạnh nhiều lần, những thú vật như vậy đã bị ảnh hưởng bởi nhiều quy luật mystic, cũng như những vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các vật như vậy như là các (2.18s) | FP16 paraphrase-right / Q4 wrong — FP16 wins |
| fp16_08.png [LONG ch93 idx4928] | THEY OFTEN GO INTO A FRENZY DURING BATTLE. ONCE THEY DESIGNATE A SPECIFIC TARGET AS AN ENEMY, THEY TEND TO FOCUS ONLY ON THAT TARGET TO THEIR OWN DETRIMENT. | ĐIỂM YẾU- DỄ RƠI VÀO TRẠNG THÁI CUỒNG LOẠN KHI BƯỚC VÀO TRẬN CHIẾN. SAU KHI XÁC ĐỊNH MỘT CÁ THỂ CỤ THỂ LÀ KẺ THÙ, CHÚNG CÓ XU HƯỚNG LAO VÀO TẤN CÔNG MỤC TIÊU MỘT CÁCH LIỀU LĨNH. | Đường thường bị kích động vào một trận chiến. Khi họ thiết lập một mục tiêu cụ thể như một đối thủ, họ thường tập trung chỉ vào mục tiêu đó để lợi ích của mình. (5.07s) | Họ thường bị kích động vào một thời điểm cụ thể trong trận chiến. Khi họ thiết lập một mục tiêu cụ thể như một đối thủ, họ thường tập trung duy nhất vào mục tiêu đó để phục vụ lợi ích của mình. (1.07s) | FP16 wrong / Q4 wrong — tie |
| fp16_09.png [LONG ch110 idx6238] | THIS POSITION IS EQUIVALENT TO THE POSITION OF THE SEVEN LIONS, AND THE COM- MANDERS OF THE SECONDARY GROUPS ARE EQUIVALENT TO THE POSITION OF A FIRST-RATE GUARD WARRIOR. | CHỨC VỤ TƯƠNG ĐƯƠNG VỚI THẤT SÁT TỬ. PHÓ LỆNH CHỦ TƯƠNG ĐƯƠNG VỚI HỘ VỆ CẤP 1. | Đây vị trí tương đương vị trí của bảy con rồng và các chỉ huy các nhóm phụ trợ tương đương vị trí của một chiến binh bảo vệ đầu tiên cấp 1. (4.72s) | vị trí này tương đương vị trí của bảy con rồng và vị trí của các cấp bậc quân sự thứ nhất. (0.56s) | FP16 wrong / Q4 wrong — tie |
| fp16_10.png [QUESTION ch84 idx4393] | DOES IT TAKE TIME FOR IT TO BE MEASURED? | CẦN CÓ THỜI GIAN ĐỂ ĐO LƯỜNG? | Đúng không, nó mất thời gian để đo lường nó? (2.01s) | Đó không cần thời gian để đo lường nó? (0.29s) | FP16 paraphrase-right / Q4 wrong — FP16 wins |
| fp16_11.png [QUESTION ch145 idx7966] | MU KI? | VÔ TÂM? | Mu ki? (0.93s) | Mu ki? (0.12s) | FP16 wrong / Q4 wrong — tie |
| fp16_12.png [QUESTION ch128 idx6796] | DOES HE HAVE SOME OTHER HIDDEN TRICK? | HẮN TA CÓ ÂM MƯU GÌ SAO? | Anh ấy có những mẹo bí ẩn khác không? (1.81s) | Anh ấy có những mẹo khác không? (0.22s) | FP16 paraphrase-right / Q4 paraphrase-right — tie |
| fp16_13.png [QUESTION ch87 idx4711] | THEN, SHALL WE FINISH IT? | KẾT THÚC THÔI. | Thì, chúng ta sẽ hoàn thành điều này chưa? (1.92s) | Khi nào thì chúng ta hoàn thành nó? (0.25s) | FP16 paraphrase-right / Q4 wrong — FP16 wins |
| fp16_14.png [HONORIFIC ch83 idx4228] | CAPTAIN! THAT'S IT! THAT SHOWED UP SOMETI— | ĐẠI CA! CHÍNH LÀ NÓ! MỖI LẦN... | Captain! That's it! That showed up sometime. (1.82s) | Captain! That's it! That showed up sometime. (0.28s) | FP16 wrong / Q4 wrong — tie |
| fp16_15.png [HONORIFIC ch160 idx9151] | WHY DOES LORD GON MA HAVE HIS OFFICE HERE IN CHEON-IL HALL? | TẠI SAO CÔN MA LẠI ĐẶT PHÒNG LÀM VIỆC TẠI THIÊN NHẤT QUÁN NÀY VẬY? | Tại sao Lord Gómà lại có phòng ở trong Cheon-il Hall? (2.34s) | Tại sao Chúa Giê-su lại có phòng ở đây trong Hall Cheon-il? (0.39s) | FP16 paraphrase-right / Q4 wrong — FP16 wins |
| fp16_16.png [HONORIFIC ch168 idx12049] | I AM A SUBORDINATE OF LORD GON MA, THE FOURTH DISCIPLE OF THE HEAVENLY DEMON. | TA LÀ THUỘC HẠ CỦA CÔN MA, VỊ ĐỆ TỬ THỨ TƯ CỦA THIÊN MA. | Tôi là một con trai của Chúa Giê-su, con thứ tư của Thiên Chúa. (2.58s) | Tôi là một phó giáo sư của Giáo sư Ma, Giáo sư thứ 4 của Thánh Thần. (0.48s) | FP16 wrong / Q4 wrong — tie |
| fp16_17.png [MEDIUM ch172 idx9295] | WHY THE HELL WOULD YOU USE A WHIRLWIND KICK LIKE THAT!!! | TẠI SAO NGƯỜI LẠI DÙNG TOÀN PHONG THỐI THEO CÁCH ĐÓ HẢ!!! | Tại sao trời đất sẽ dùng một cú kick như vậy!!! (1.86s) | Tại sao trời sẽ dùng một cú kick như vậy? (0.29s) | FP16 wrong / Q4 wrong — tie |
| fp16_18.png [MEDIUM ch136 idx7492] | YOU'RE RIGHT ABOUT THE TAIJI WISDOM SWORD BEING CONSIDERED BETTER. | TẤT NHIÊN, THÁI CỰC TUỆ KIỂM ĐƯỢC ĐÁNH GIÁ LÀ MẠNH HƠN. | Bạn đúng về tinh thần của kiếm thần Taiji. (1.93s) | Bạn đã đúng về tinh thần của kiếm thần Nhật Bản. (0.31s) | FP16 wrong / Q4 wrong — tie |

### Caveats

- **Greedy outputs are NOT reproducible across runs (confirmed again).** The cancelled run's Q4 job also finished: Q4 run1 vs run2 outputs are identical on only 10/21 images, differ on 11/21 (same script, same flags, different hosts/builds). Two consequences: (1) the fp16_07 Q4 loop (100 tok) did NOT reproduce in run1 (clean 41-tok output ending '...như: Jiangshi.') — it is a one-off instability, not a stable Q4 property; counting run1 instead, FP16 wins 4–0 rather than 5–0. (2) The 'Chúa Giê-su' error (fp16_15) DID reproduce in both Q4 runs while FP16 eager never produces it — stable on the llama.cpp path (quantization vs implementation not separated here). A/B quality claims from a single run overstate their precision; the 5–0 direction is consistent across both Q4 runs, the exact margin is not.
- **Different runner CPUs** (FP16 on Xeon 6973P-C, Q4 on EPYC 7763) — speed comparisons carry the usual ±1.5× host noise; the 5.6× gap is far outside it.
- **Base model only.** Same caveat as round 3: this says nothing about the fine-tuned r14 adapter's quality under quantization.
- **fp16_07 Q4 loop** is the scariest failure mode seen (degenerate repetition to the token cap). If the pipeline ever ships Q4, a repetition guard / max-token cap is mandatory.

**Bottom line: keep Q4_K_M for speed, but do not claim it is quality-equal to FP16 — on hard murim lines it is measurably worse (5–0, 0 wins), and the base model itself is the bigger quality problem on both sides. Re-check after the r14 adapter is merged into GGUF.**

## F16, zero-quant (user-ordered follow-up, 2026-11-21 19:52)

Question: how much of the combo speed survives with ZERO quantization? Run **44906163199** (commit 54064de), job `bench (llama-fp16-noquant)` completed/success. F16 GGUF straight out of `convert_hf_to_gguf.py` (no `llama-quantize` step at all) + F16 KV cache (`--cache-type-k/v f16`), otherwise the exact round-3 combo flags (`-fa`, batch/ubatch 1024, `--cache-prompt --cache-reuse 256`). 3 standard images, same prompt, greedy, max 100 tokens, warmup + 3 timed runs, median. Result JSON: `results-fp16-noquant/artifacts/results-llama-fp16-noquant.json` (local only).

| | F16 no-quant (llama.cpp) | Q4_K_M combo | FP16 eager |
|---|---|---|---|
| s/image (mean of per-image medians) | **0.91** (0.54 / 1.50 / 0.69) | 0.58 round 3 (0.46 on the 21-img set) | 2.57 (21-img) / 2.30 (3-img) |
| tok/s (mean) | 20.4 | 31.5 | 7.7 |
| load_s | 1.0 | 1.0 | 17.6 |
| peak RSS | 2,049 MB | 1,304 MB | 2,582 MB |
| runner CPU | AMD EPYC 7763 | EPYC 7763 (21-img set) | Xeon 6973P-C |

**Verdict: yes, the combo speedups work without quantization.** F16-no-quant is **2.5–2.8× faster than FP16 eager** (0.91s vs 2.30/2.57s) and only **~1.6× slower than the Q4 combo** (0.91s vs 0.58s round 3; ~2× vs the 0.46s 21-image mean). Split the gain: the engine + flags are worth ~2.5× over eager; the Q4 quantization itself adds another ~1.6–2× on top. The F16 GGUF is ~1.6 GB on disk (vs 522 MB for Q4_K_M) and uses ~2.0 GB RSS (vs 1.3 GB Q4, 2.6 GB eager).

Outputs are **verbatim-identical to FP16 eager on all 3 images** ("Anh ấy không tồn tại trong thế giới này." / "Vì vậy, bạn có thể sử dụng sức mạnh vô hạn của Tai Chi…" / "Tôi đã yêu thích bạn từ ngày chúng tôi lần đầu gặp.") — same weights, greedy, so this path loses nothing vs eager on these 3, unlike Q4 which lost 5–0 on the 21-image quality set. The per-image split shows where quant matters most: the 31-token img2 is 1.50s F16 vs 0.89s Q4 (decode-bound long output), while the 11-token img1 is 0.54s vs 0.35s.

Caveats: 3 images, one run — runner noise ±1.5× applies, though the 2.5× gap over eager is outside it and the host (EPYC 7763) matches the Q4 21-image run. The quality side still rests on the 21-image FP16 section above; this run only proves speed + output identity on 3 images. The r14 adapter still needs merging before any of this is the shipped pipeline (OPEN item 1); re-check quality after the merge.
