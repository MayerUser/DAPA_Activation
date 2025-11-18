# DAPA Project (Anonymous Submission)

> Anonymous code repository for a double-blind conference review.

The project name **DAPA** is used only as an internal label for this repo and does not necessarily match the final paper title.

---

## 1. Overview

To eva the image task performance, move to src_cls, run the following Makefile

```
This Makefile manages the full experiment pipeline for ViT PWL approximation and testing.

Usage: make <target>

--- 1. FULL PIPELINE & EXPERIMENT TESTS ---
  all             : Runs the complete experiment pipeline: clean, generate PWL/Poly files, and run EXP1, EXP3, and EXP5 tests.
  pwl_img         : Generates Piecewise Linear (PWL) approximation files for Image models (DWMSE loss).
  pwl_img_mse     : Generates PWL approximation files for Image models (MSE loss, for EXP6).
  poly            : Generates Polynomial approximation files for GELU.
  test_exp1       : Runs EXP1: Performance tests comparing DWMSE PWL and Polynomial GELU approximations.
  test_exp3       : Runs EXP3: Architecture classification tests (FP32 and Fixed-Point) across various models.
  test_exp5       : Runs EXP5: Ablation study comparing single vs. combined PWL approximation for Softmax/GELU.
  test_exp6       : Runs EXP6: Tests comparing DWMSE-based vs. MSE-based PWL files.
  demo            : Runs a quick demo test (vit-small with PWL-16 for Softmax/Activation).
  ref             : Runs a reference test (vit-small using PyTorch FP32 baseline). @echo 
--- 3. UTILITY ---
  clean           : Removes all generated files and directories (dst_pdf, dst_log, dst_pwl, etc.).
  help            : Displays this help message.
```