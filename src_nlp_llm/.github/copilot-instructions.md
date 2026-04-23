# Copilot Instructions for src_nlp_llm

- Primary task: maintain and extend the PWL approximation pipeline for GPT-2 and LLaMA 2 7B.
- Key entry points:
  - `Makefile` for build/test/demo flows.
  - `t0_make_pwl.py` to generate PWL JSON from activation distributions.
  - `t2_gpt2_run.py` to evaluate GPT-2 with torch / PWL softmax / poly/PWL GELU.
  - `t3_llama_demo.py` to evaluate LLaMA 2 7B with fixed-point PWL softmax/SiLU.
- Important modules:
  - `m0_udanf.py`: generic PWL loader + `PWLSoftmax` / `PWLGelu` / `DebugSoftmax`.
  - `m1_poly_act.py`: polynomial GELU wrapper loaded from JSON.
  - `m3_udanf_fixed.py`: fixed-point quantized PWL implementations used by LLaMA demo.

## Data flow and conventions
- `t0_make_pwl.py` hooks model internals and captures:
  - `softmax_input` distributions for softmax PWL exp.
  - `gelu_input` / SiLU input distributions for activation PWL.
- Output files are written under `dst_pwl/<num_samples>/` with names like:
  - `pwl_exp_sm_<model>_<N>seg.json`
  - `pwl_gelu_act_<model>_<N>seg.json`
- `t2_gpt2_run.py` expects these JSON files and loads them via `PWLSoftmax` / `PWLGelu`.
- `t3_llama_demo.py` expects fixed-point wrappers from `m3_udanf_fixed.py` and uses a global softmax patch plus `act_fn` replacement.
- Dataset is WikiText-2 (`wikitext-2-raw-v1`) and deliberately avoids `attention_mask` due to dataset pipeline issues.

## Recommended commands
- Generate PWL data:
  - `make run MODEL=gpt2 LOSS=dwmse SEGMENTS=8 NUM_SAMPLES=256`
  - `make run MODEL=llama2-7b LOSS=dwmse SEGMENTS=8 NUM_SAMPLES=256`
- Run end-to-end demos:
  - `make demo_gpt2`
  - `make demo_llama`
- Test / compare modes directly:
  - `make test_gpt2_baseline`
  - `make test_gpt2_pwl8`
  - `make test_llama_baseline`
  - `make test_llama_pwl8`
- Clean generated outputs:
  - `make clean`

## Project-specific patterns
- Avoid changing model load patterns in LLaMA: `AutoModelForCausalLM.from_pretrained(..., device_map="auto")`.
- For GPT-2 injection, patch `GPT2Attention._attn` and `GPT2MLP` `act` logic inside `inject_gpt2_pwl_softmax` / `inject_act_into_gpt2_mlp`.
- For LLaMA, patch global softmax in `transformers.models.llama.modeling_llama.nn.functional.softmax` and replace `module.act_fn` for `nn.SiLU`.
- Follow existing JSON naming and directory structure exactly; evaluation scripts rely on `dst_pwl/<num_samples>/...`.
- `config.py` contains Hugging Face token and path constants, but the demo scripts rely on explicit CLI arguments and `Cache_dir` values.

## What to preserve for future edits
- The separation between generation (`t0_make_pwl.py`) and evaluation (`t2_gpt2_run.py`, `t3_llama_demo.py`).
- The PWL JSON format used by `m0_udanf.py` / `m1_poly_act.py` / `m3_udanf_fixed.py`.
- The `wikitext-2-raw-v1` data preprocessing strategy that groups tokenized text into fixed 1024-length blocks.

> If any part of the data path, JSON naming, or patching logic is unclear, ask for clarification before changing `t0_make_pwl.py`, `t2_gpt2_run.py`, or `t3_llama_demo.py`.
