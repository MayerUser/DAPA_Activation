# DAPA: Distribution-Aware Piecewise Activation Functions

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conference](https://img.shields.io/badge/Accepted-DAC_2026-success.svg)](#)

This repository contains the official implementation of the DAC 2026 paper:

**DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training**

The current public release focuses on the experiments used for the accepted paper: Vision Transformer image classification and GPT-2 language-model perplexity evaluation.

## Core Idea

Piecewise linear (PWL) approximation is commonly used to replace expensive nonlinear functions such as `exp`, `softmax`, and `GELU` on edge hardware. A uniform PWL fit optimized only for numerical MSE can waste segments in rarely used regions and introduce biased approximation noise in the high-probability activation range.

DAPA instead fits the approximation under the empirical input distribution observed in Transformer workloads.

Key components:

- **Distribution-weighted fitting**: DAPA uses Distribution-Weighted MSE (DWMSE) so the fitted segments prioritize the regions that actually dominate runtime activations.
- **Activation-aware PWL generation**: The calibration scripts collect pre-softmax and pre-activation distributions from the target model, then generate JSON coefficients for each approximation target.
- **Fixed-point hardware simulation**: The fixed-point modules evaluate quantized PWL coefficients and bit-width choices for hardware-oriented deployment.

## Repository Structure

```text
DAPA_Activation/
├── LICENSE
├── README.md
├── requirements.txt
│
├── src_img_cls/
│   ├── Makefile
│   ├── aux0_exp3.py
│   ├── config.py
│   ├── exp3_results.csv
│   ├── imagenet_cache.py
│   ├── m0_udanf.py
│   ├── m1_poly_act.py
│   ├── m2_find_fixed_bit.py
│   ├── m3_udanf_fixed.py
│   ├── t0_make_pwl.py
│   ├── t1_vit_run.py
│   ├── t2_make_poly.py
│   └── t3_vit_run_fixed.py
│
└── src_nlp_llm/
    ├── Makefile
    ├── m0_udanf.py
    ├── m1_poly_act.py
    ├── m2_dwmse_cal.py
    ├── t0_make_poly.py
    ├── t0_make_pwl.py
    └── t2_gpt2_run.py
```

The `m_*` files implement reusable modules and algorithms. The `t_*` files are experiment and evaluation entry points.

## Installation

A Python 3.11 environment is recommended.

```bash
conda create -n dapa python=3.11
conda activate dapa
pip install -r requirements.txt
```

The provided `requirements.txt` keeps only the core dependencies needed for the public vision and GPT-2 PPL demos. Install the PyTorch build that matches your CUDA or CPU environment if the default package resolver does not pick the desired wheel.

## Hugging Face Cache

The Makefiles use a shared Hugging Face cache by default:

```make
CACHE_DIR ?= $(HOME)/MyWorkspace/HfHome
```

You can override it for any run:

```bash
make demo_gpt2 CACHE_DIR=/path/to/hf_cache
make test_exp1 CACHE_DIR=/path/to/hf_cache
```

Required Hugging Face assets:

- GPT-2 model and WikiText-2 for NLP experiments.
- ImageNet-1K validation data for image-classification experiments.
- Vision model checkpoints such as ViT, DeiT, and Swin.

For ImageNet-1K, request access to `ILSVRC/imagenet-1k` on Hugging Face and authenticate once:

```bash
huggingface-cli login
```

`src_img_cls/imagenet_cache.py` first tries to reuse locally prepared ImageNet Arrow shards from the configured cache. If they are not present, it falls back to Hugging Face dataset loading.

## Makefile Usage

Both experiment directories default to help output:

```bash
cd src_img_cls
make

cd ../src_nlp_llm
make
```

The Python executable can be overridden without editing the Makefiles:

```bash
make demo_gpt2 PYTHON=/path/to/python
```

## Reproducing Vision Experiments

```bash
cd src_img_cls
```

Generate DAPA PWL coefficients:

```bash
make pwl_img
```

Generate polynomial baselines:

```bash
make poly
```

Run the main image-classification experiments:

```bash
make test_exp1
make test_exp2
make test_exp3
make test_exp5
make test_exp6
```

Run fixed-point bit-width analysis:

```bash
make all-bits
```

Run the full paper pipeline:

```bash
make paper-all
```

## Reproducing GPT-2 NLP Experiments

```bash
cd src_nlp_llm
```

Run the complete GPT-2 perplexity demo:

```bash
make demo_gpt2
```

Or run individual GPT-2 targets:

```bash
make run MODEL=gpt2 SEGMENTS=8
make run MODEL=gpt2 SEGMENTS=16
make test_gpt2_baseline
make test_gpt2_pwl8
make test_gpt2_pwl16
```

## Quick Smoke Tests

These smaller commands are useful for checking the environment before launching full experiments.

Image classification:

```bash
cd src_img_cls
python t1_vit_run.py --model_name vit-tiny --num_samples 1 --cache_dir /path/to/hf_cache
python t0_make_pwl.py --model_name vit-tiny --segment_number 4 --num_samples 1 --cache_dir /path/to/hf_cache
```

NLP:

```bash
cd src_nlp_llm
make run MODEL=gpt2 SEGMENTS=8 NUM_SAMPLES=64 CACHE_DIR=/path/to/hf_cache
```

## License and Citation

This project is released under the Apache License 2.0.

If you use this repository or the DAPA framework in your research, please cite:

```bibtex
@inproceedings{xiang2026dapa,
  title     = {DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training},
  author    = {Xiang, Maoyang and Wang, Bo},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference},
  series    = {DAC '26},
  year      = {2026},
  location  = {Long Beach, CA, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3770743.3804274},
  isbn      = {979-8-4007-2254-7}
}
```
