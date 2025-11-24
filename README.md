# DAPA Project (Anonymous Code for Review)

> Anonymous code repository for double-blind conference review.  
> The label "DAPA" is used only as an internal project name and does not reveal the final paper title.

# Environment Setup

## Python environment

Install dependencies with:

    pip install -r requirements.txt

The code has been tested with recent Python 3.x versions and a single GPU.  
A GPU is strongly recommended, but small demos can run on CPU (slow).

## Hugging Face login

Pre-trained models and the ImageNet-1K evaluation set are downloaded via Hugging Face.

1. Login:

    huggingface-cli login

2. Request access to ImageNet-1K on Hugging Face:

    Dataset: ILSVRC/imagenet-1k

After access is granted, the code can stream the validation split automatically.

Optionally configure a shared cache:

    export HF_HOME=/path/to/hf_cache


# Quick Demo (Recommended for Reviewers)

To avoid long runtimes and large memory usage, this repository provides a small demo that runs on a subset of ImageNet-1K and a few ViT variants.

From the repository root:

    make

This launches two demos under "src_cls/".

2.1. Demo 0 – ViT variants with 16-segment DAPA

For each ViT variant (vit-tiny, vit-small, vit-base) and 16 segments, the demo:

1) Generates a DAPA (piecewise linear) approximation of the activation and softmax from 256 samples.
2) Evaluates the corresponding ViT model using this 16-segment DAPA configuration.

This provides a quick view of how a single, moderately fine-grained DAPA setting behaves across several ViT backbones.

2.2. Demo 1 – ViT-Tiny: DAPA vs Polynomial GELU

For ViT-Tiny, the demo compares:

- DAPA GELU with segments: 4, 6, 8, 10, 12, 14, 16
- Polynomial GELU with orders: 4, 5, 6, 7, 8

Internally, the demo:

1) Evaluates ViT-Tiny with several DAPA segment counts.
2) Evaluates ViT-Tiny with several polynomial GELU approximations.
3) Aggregates metrics into a single JSON file for analysis.

For each configuration, the following quantities are computed:

- MSE between exact GELU and the approximation on [-4, 4] (uniform in x).
- DWMSE (distribution-weighted MSE) with respect to the empirical activation PDF.
- Top-1 accuracy and accuracy drop (ΔAccuracy) relative to a ViT-Tiny FP32 baseline.
- Pearson correlation (r, p-value) between:
    - ΔAccuracy vs MSE
    - ΔAccuracy vs DWMSE

All aggregated metrics are saved to:

    src_cls/dst_aux/demo1_metrics.json


# Where to Find Results

All outputs live under "src_cls/":

- Logs: src_cls/dst_log/
  Test logs, including reported Top-1 accuracy.

- Plots: src_cls/dst_plot/
  Figures comparing original functions vs DAPA / polynomial approximations, for example:
    plot_pwl_vs_orig_gelu_act_vit-tiny_16seg.png

- PWL (DAPA) configs: src_cls/dst_pwl/
  JSON files describing the piecewise segments and coefficients.

- Polynomial configs: src_cls/dst_poly/
  JSON files describing polynomial GELU approximations and tails.

- PDFs (distributions): src_cls/dst_pdf/
  Empirical PDFs of activations / softmax inputs used for DWMSE.

- Aggregated metrics: src_cls/dst_aux/
  Metrics and correlations produced by the demo analysis.

Example PWL JSON (DAPA config):

    {
      "intervals": [
        ["-inf", "-9.930583"],
        ["-9.930583", "-6.573429"]
      ],
      "params": [
        { "p1": 0.0, "p0": 0.0 },
        { "p1": -1.611651355024917e-14, "p0": -1.4029910592064354e-13 }
      ]
    }

Each interval [x_start, x_end] uses a linear function:

    σ̂(x) = p1 * x + p0

Higher-order variants may also include coefficients p2, p3 for quadratic/cubic segments.


# Source Tree Overview

High-level structure:

    ├── figure/                  # Figures used in this README (not needed for code)
    ├── LICENSE                  # License file
    ├── Makefile                 # Top-level demo targets (see Section 2)
    ├── README.md                # This file
    ├── requirements.txt         # Python dependencies
    └── src_cls/                 # DAPA for image classification Transformers
        ├── config.py            # Global config; SAMPLE_NUM controls evaluation size
        ├── m0_udanf.py          # DAPA-based activation implementation
        ├── m1_poly_act.py       # Polynomial activation implementation
        ├── Makefile             # Optional internal flow control
        ├── t2_make_poly.py      # Polynomial GELU approximation generation
        └── ...                  # Additional helper scripts for DAPA/PDF generation and ViT evaluation

Full ImageNet-1K evaluation:

To evaluate on the full ImageNet-1K validation set rather than a 256-sample subset, adjust SAMPLE_NUM in "src_cls/config.py" (for example, set it to 50000). This will significantly increase runtime and resource usage.


# Reproducibility Notes

- All demos use public Hugging Face models and datasets.
- Randomness is limited (no training, only evaluation), so results should be broadly stable across runs.
- GPU/driver differences may cause minor numerical variations but should not affect the overall trends relating MSE / DWMSE to accuracy.

This repository is provided solely for anonymous artifact / code review.  
Please do not attempt to de-anonymize based on model or dataset choices.


# Note

For fast testing and easy reproduction on a single GPU, the demo in this repository deliberately uses:

- Only a small subset of ImageNet-1K (for example, 256 images by default), and
- Empirical PDFs estimated from randomly sampled activations to compute DWMSE.

Because of this, the numerical results you obtain here may differ slightly from the numbers reported in the final paper, due to different random seeds, sample subsets, or hardware. These differences affect exact values but do not change the qualitative conclusion:

    Distribution-Weighted MSE (DWMSE) is a more reliable error metric than plain MSE for predicting how activation approximations impact end-to-end accuracy.

As an illustration, one representative run for Demo 1 (combining piecewise-linear and polynomial cases) produced:

    Correlation (All (PWL + Poly)):
      DeltaTop1 vs MSE:   r = -0.1884, p = 5.5763e-01
      DeltaTop1 vs DWMSE: r = -0.9840, p = 8.0333e-09

In this run, the correlation between ΔAccuracy and DWMSE is very strong and highly significant, whereas the correlation between ΔAccuracy and MSE is weak and not statistically significant. This behavior is representative of the trend observed in our full experiments, even though individual numbers may vary slightly across runs.
