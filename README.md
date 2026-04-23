# DAPA: Distribution-Aware Piecewise Activation Functions

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conference](https://img.shields.io/badge/Accepted-DAC_2026-success.svg)](#)

This repository contains the official implementation of the paper: **"DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training"** (Accepted at DAC 2026).

## 💡 The Core Insight: Why DAPA?

When deploying Large Language Models (LLMs) and Vision Transformers (ViTs) on resource-constrained edge hardware (e.g., NPUs, FPGAs), replacing complex non-linear functions (like `exp` and `GELU`) with Piecewise Linear (PWL) approximation is a standard practice to save DSPs and power.

However, traditional Uniform PWL optimized for Mean Squared Error (MSE) often leads to **catastrophic attention collapse** during autoregressive generation. Because MSE ignores the highly skewed pre-activation data distribution caused by the Causal Mask, it introduces asymmetric noise. Over multiple generation steps, this noise accumulates linearly `O(N)`, trapping the model in "repetition loops" or generating gibberish.

**DAPA solves this by shifting the paradigm from Numerical Approximation to Distribution Matching:**
1. **Distribution-Weighted MSE (DWMSE):** DAPA allocates dense hardware segments (knots) exclusively to the high-probability active regions of the input distribution, ignoring the "dead zones".
2. **Symmetric Zero-Mean Noise:** By matching the mathematical expectation of the noise to the true distribution, DAPA converts the catastrophic `O(N)` linear error drift into a random walk. 

The result? **A 16-segment Fixed-Point (Fix16) hardware logic that reduces GELU/Softmax DSP utilization by 16x~48x, while perfectly maintaining the logical reasoning and text generation capabilities of a full-precision baseline.**

---

## 📂 Repository Structure

The codebase is systematically organized into independent evaluation environments for Vision and NLP tasks. We adopt a strict naming convention: `m_*` scripts contain core algorithmic modules (hardware simulation, bit-width search), and `t_*` scripts contain test pipelines and model evaluation logic.

```text
DAPA_Activation/
├── LICENSE                     # Apache 2.0 License
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── figure/                     # Directory for generated plots and visualizations
│   └── plot_pwl_vs_orig_gelu_act_vit-tiny_16seg.png
│
├── src_img_cls/                # 👁️ Vision Transformers (ViT, DeiT, Swin) Evaluation
│   ├── Makefile                # Automation for all vision experiments (EXP1 to EXP6)
│   ├── config.py               # Global configurations for image classification tasks
│   │
│   ├── m0_udanf.py             # Floating-point DAPA (PWL) PyTorch modules
│   ├── m1_poly_act.py          # Baseline Polynomial approximation modules
│   ├── m2_find_fixed_bit.py    # DWMSE-guided Fixed-Point Bit-Width Search (Paper Algorithm 1)
│   ├── m3_udanf_fixed.py       # Fixed-point DAPA hardware simulation modules
│   │
│   ├── t0_make_pwl.py          # Generates PWL JSON parameters based on data distribution
│   ├── t1_vit_run.py           # Evaluates Vision models using floating-point DAPA
│   ├── t2_make_poly.py         # Generates polynomial approximation parameters
│   └── t3_vit_run_fixed.py     # Evaluates Vision models using Fixed-point simulated DAPA
│
└── src_nlp_llm/                # 💬 NLP & Large Language Models (GPT-2, LLaMA-2) Evaluation
    ├── Makefile                # Automation for GPT-2 evaluation and LLaMA-2 text generation
    ├── config.py               # Global configurations for NLP tasks
    │
    ├── m0_udanf.py             # Floating-point DAPA (PWL) PyTorch modules
    ├── m1_poly_act.py          # Baseline Polynomial approximation modules
    ├── m2_dwmse_cal.py         # Utilities to calculate Distribution-Weighted MSE (DWMSE)
    ├── m3_find_fixed_bit.py    # Fixed-Point Bit-Width Search for LLM activations
    ├── m3_udanf_fixed.py       # Fixed-point DAPA hardware simulation modules
    │
    ├── t0_make_pwl.py          # Generates PWL parameters using LLM pre-activation distributions
    ├── t0_make_poly.py         # Generates polynomial approximation parameters for NLP
    ├── t2_gpt2_run.py          # Evaluates GPT-2 perplexity (Float)
    ├── t2b_gpt2_run_fixed.py   # Evaluates GPT-2 perplexity (Fixed-point)
    ├── t3_llama_demo.py        # Evaluates LLaMA-2 7B WikiText perplexity (Float & Fixed)
    └── t4_llama_generate.py    # Core script for LLaMA-2 Autoregressive Text Generation Demo
```
## 🛠️ Installation & Setup

### 1. Python Environment

Install dependencies with: ```pip install -r requirements.txt```

### 2. Hugging Face Login
Pre-trained models and the ImageNet-1K evaluation set are downloaded via Hugging Face.

* Step 1: Login to your Hugging Face account: ```huggingface-cli login```
* Step 2: Request access to ImageNet-1K on Hugging Face Dataset: ILSVRC/imagenet-1k
* Step 3: (Optional) Configure a shared cache to avoid redundant downloads ```export HF_HOME=/path/to/hf_cache```

## 🚀 Real Text Generation Demo (LLaMA-2 7B)

To observe the "Attention Collapse" phenomenon and how DAPA prevents it, navigate to the NLP directory and run the interactive demo:

```
cd src_nlp_llm
make demo_llama_gen
```
You can also test your own sentences by passing arguments:

```
make demo_llama_gen PROMPT="The capital of France is Paris, and the capital of Japan is" MAX_TOKENS=60
```

### Example 1: Factual Recall

Input: ```The capital of France is Paris, and the capital of Japan is```

* [FP16 | Softmax: torch | Act: torch] (Baseline)

    *** The capital of France is Paris, and the capital of Japan is *** Tokyo. The capital of France is Paris,

* [FP16 | Softmax: pwl-8 | Act: pwl-8] (Uniform PWL)

    *** The capital of France is Paris, and the capital of Japan is *** Tokyo. nobody knows the capital of the capital of ❌ (Attention Collapse)

* [FP16 | Softmax: pwl-16 | Act: pwl-16] (DAPA)

    *** The capital of France is Paris, and the capital of Japan is *** Tokyo. The capital of the United States is Washington ✅ (Stable Logic)


### Example 2: Long-Context Storytelling
Input: ```Once upon a time in a magical forest, there lived a tiny brave fox named Foxy. Foxy```

* [FP16 | Softmax: torch | Act: torch] (Baseline)

    *** Once upon a time in a magical forest, there lived a tiny brave fox named Foxy. Foxy *** was a very brave fox, but he was also very lonely. He wanted to find a friend, but he didn’t know where to look.
One day, Foxy was walking through the forest when he

* [FP16 | Softmax: pwl-8 | Act: pwl-8] (Uniform PWL)

    *** Once upon a time in a magical forest, there lived a tiny brave fox named Foxy.Foxy. *** He was a very brave and kind fox. He was a very good friend to all the other animals in the forest. He was a very good friend to all the other animals in the forest. He was ❌ (Repetition Loop)

* [FP16 | Softmax: pwl-16 | Act: pwl-16] (DAPA)

    ***Once upon a time in a magical forest, there lived a tiny brave fox named Foxy. Foxy*** was a brave little fox. He was the smallest of the foxes in the forest. He was also the bravest. He was the only fox in the forest who was not afraid of the big bad ✅ (Perfect Syntax & Diversity)

## 📊 Reproducing Paper Experiments
You can reproduce the data for the tables and figures in the DAC 2026 paper using the provided Makefiles.

### Part 1: Vision Transformers (src_img_cls/)

Navigate to cd ```src_img_cls/```. Run make all for the full pipeline, or run specific targets:

#### Figure 1 (MSE vs DWMSE Performance Correlation):

```Bash
make test_exp1
```
Evaluates how different error metrics correlate with actual ViT Top-1 accuracy drops.

#### Figure 4 (Impact of Number of Samples):

```Bash
make test_exp2
```
Proves that DAPA can reliably model the distribution using as few as 4-16 calibration images.

#### Table 2 (Image Classification Architecture vs Performance):

```Bash
make test_exp3
```

Runs the comprehensive 5-step evaluation across ViT, DeiT, and Swin variants, comparing Baseline, Softmax-only, GELU-only, and fully Fixed-Point Quantized (Q9.7/Q6.8) DAPA configurations.

### Part 2: NLP & LLMs (src_nlp_llm/)
Navigate to cd ```src_nlp_llm/```.

#### Table 2 & Figure 2 (GPT-2 Perplexity Evaluation):

```Bash
make demo_gpt2
```
Generates PWL files and evaluates the perplexity (PPL) on the WikiText-2 dataset for the GPT-2 baseline, 8-seg, and 16-seg DAPA models.

## 📝 License & Citation
This project is open-sourced under the Apache License 2.0. You are free to use, modify, and distribute this software for both academic and commercial purposes, provided that proper attribution is given and the patent terms are respected.

If you find this repository (or the DAPA framework) useful in your research, hardware design, or commercial products, please kindly cite our DAC 2026 paper:

```
@article{xiang2026dapa,
  title={DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training},
  author={Xiang, Maoyang and Wang, Bo},
  journal={arXiv preprint arXiv:2603.19338},
  year={2026}
}
```