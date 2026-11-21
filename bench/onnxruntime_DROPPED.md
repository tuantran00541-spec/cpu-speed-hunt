# ONNX Runtime lane — DROPPED (no script written)

## Verdict

**No `bench/onnxruntime.py` was written, deliberately.** The task required
verifying from optimum-onnxruntime docs that an `ORTModelForVisualCausalLM`
(or equivalent) exists AND supports Qwen3.5 VL image input. That
verification failed.

## What was checked (2026-11-21)

1. **optimum-onnx modeling reference** —
   <https://huggingface.co/docs/optimum-onnx/en/onnxruntime/package_reference/modeling>
   (full page, 2,112 lines). The complete class list contains:
   `ORTModel`, `ORTModelForCausalLM`, `ORTModelForMaskedLM`,
   `ORTModelForSeq2SeqLM`, `ORTModelForSpeechSeq2Seq`,
   `ORTModelForVision2Seq`, `ORTModelForPix2Struct`,
   `ORTModelForCustomTasks`, `ORTModelForFeatureExtraction`, plus
   classification / QA / token-classification variants.
   **There is no `ORTModelForVisualCausalLM` and no
   `ORTModelForImageTextToText` anywhere on the page** (exact-name
   searches return zero hits).
2. **Closest vision-generative classes are not equivalents:**
   - `ORTModelForVision2Seq` is an encoder–decoder class (documented
     example: `nlpconnect/vit-gpt2-image-captioning`) — ViT encoder +
     GPT-2 decoder, not a decoder-only image-text-to-text VLM.
   - `ORTModelForPix2Struct` is Pix2Struct-specific.
   - Neither supports the Qwen3.5 architecture
     (`Qwen3_5ForConditionalGeneration`: hybrid Gated DeltaNet +
     attention, native multimodal, MTP head) or its image-text-to-text
     task.
3. **ONNX exporter side:** the optimum ONNX exporter documents
   text-generation exports (e.g. `optimum-cli export onnx --model
   Qwen/Qwen3-8B ... --task text-generation`) and no Qwen3.5 /
   image-text-to-text VLM export path was found in the optimum-onnx
   docs. A community pre-export (`onnx-community/Qwen3.5-2B-ONNX-OPT`)
   exists on the Hub, but that is not an optimum `ORTModelForXxx` class
   with verified image-input support, so it does not satisfy the
   verification bar — and the hunt's model is the 0.8B abliterated
   checkpoint, not that 2B export.
4. General web search for `"ORTModelForVisualCausalLM"` /
   `"ORTModelForImageTextToText"` returns zero hits (only the
   optimum-intel `OVModelForVisualCausalLM` exists under that name).

## Consequence

- Do **not** add an ONNX export step or an `onnxruntime` job to the
  workflow on the basis of an assumed `ORTModelForVisualCausalLM` —
  it does not exist in optimum-onnxruntime as documented.
- If the ONNX lane is ever revived, the honest routes would be:
  (a) a raw `onnxruntime.InferenceSession` hand-roll over separately
  exported vision-encoder + LLM ONNX graphs (a new build task, not a
  drop-in `ORTModelForXxx` bench, and outside this hunt's protocol);
  or (b) re-checking the optimum-onnx docs for a newly added VLM
  class and only then writing the script.
- The OpenVINO lanes (INT8 / INT4 / combo) are unaffected.
