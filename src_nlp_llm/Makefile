# =========================
# Makefile (updated - GPT-2 PPL & LLaMA 2 Text Generation)
# =========================

PYTHON        := python3
SHELL         := /bin/bash

# --- Scripts ---
SCRIPT_PWL    := t0_make_pwl.py
SCRIPT_POLY   := t0_make_poly.py
SCRIPT_T2     := t2_gpt2_run.py
SCRIPT_T3     := t3_llama_demo.py
SCRIPT_T4     := t4_llama_generate.py
CONFIG        := config.py
SCRIPT_M3     := m3_find_fixed_bit.py

# --- Dirs ---
LOG_DIR       := dst_log
TEST_LOG_DIR  := dst_log_test
DATE_DIR      := dst_date
PDF_DIR       := dst_pdf
PWL_DIR       := dst_pwl
PWL_MSE_DIR   := dst_pwl_mse

# --- Time / Defaults ---
DATE          := $(shell date +%Y-%m-%d)
NUM_SAMPLES ?= 256
BATCH_SIZE  ?= 32
BLOCK_SIZE  ?= 1024
CACHE_DIR   ?= ./huggingface_cache
PRECISION   ?= fp16   # fp32 | fp16

# --- Params for generators ---
MODEL        ?=
SEGMENTS     ?=
LOSS         ?= dwmse

# --- Generated file flags ---
FLAG_FILE     = $(DATE_DIR)/$(DATE)-$(MODEL)-$(SEGMENTS)-$(LOSS).success
LOG_FILE      = $(LOG_DIR)/$(DATE)-$(MODEL)-$(SEGMENTS)-$(LOSS).log

# --- Phony ---
.PHONY: all help clean \
        run \
        test_gpt2_baseline test_gpt2_pwl8 test_gpt2_pwl16 demo_gpt2 \
        pwl_llama_8 pwl_llama_16 \
        gen_llama_baseline gen_llama_pwl8 gen_llama_pwl16 demo_llama_gen

all: help

# =========================
# PWL generation (t0_make_pwl.py)
# =========================
run: check_params $(FLAG_FILE)
	@echo "--- Target $(FLAG_FILE) is up to date. Skipping. ---"

$(FLAG_FILE): $(SCRIPT_PWL)
	@echo "--- Building target: $(FLAG_FILE) ---"
	@mkdir -p $(LOG_DIR) $(DATE_DIR)
	set -o pipefail; $(PYTHON) $(SCRIPT_PWL) \
		--model_name        $(MODEL) \
		--loss              $(LOSS) \
		--segment_number    $(SEGMENTS) \
		--num_samples       $(NUM_SAMPLES) \
		--cache_dir $(CACHE_DIR) 2>&1 | tee -- "$(LOG_FILE)"
	@touch "$(FLAG_FILE)"

check_params:
ifeq ($(MODEL),)
	$(error MODEL is not set. e.g., make run MODEL=gpt2 SEGMENTS=8)
endif
ifeq ($(SEGMENTS),)
	$(error SEGMENTS is not set. e.g., make run MODEL=gpt2 SEGMENTS=8)
endif

# =========================
# GPT-2 End-to-End Demo (PPL)
# =========================
test_gpt2_baseline:
	$(PYTHON) $(SCRIPT_T2) --precision fp32 --softmax torch --act torch

test_gpt2_pwl8:
	$(PYTHON) $(SCRIPT_T2) --precision fp32 --softmax pwl-8 --act pwl-8

test_gpt2_pwl16:
	$(PYTHON) $(SCRIPT_T2) --precision fp32 --softmax pwl-16 --act pwl-16

demo_gpt2: 
	@$(MAKE) run MODEL=gpt2 LOSS=dwmse SEGMENTS=8
	@$(MAKE) run MODEL=gpt2 LOSS=dwmse SEGMENTS=16
	@$(MAKE) test_gpt2_baseline
	@$(MAKE) test_gpt2_pwl8
	@$(MAKE) test_gpt2_pwl16
	@echo ">>> GPT-2 End-to-End Demo Completed Successfully!"

# =========================
# LLaMA 2 7B - Phase 1: Generate PWL Files
# =========================
pwl_llama_8:
	$(MAKE) run MODEL=llama2-7b LOSS=dwmse SEGMENTS=8 NUM_SAMPLES=256 PRECISION=fp16

pwl_llama_16:
	$(MAKE) run MODEL=llama2-7b LOSS=dwmse SEGMENTS=16 NUM_SAMPLES=256 PRECISION=fp16

# =========================
# LLaMA 2 7B - Phase 2: Real Text Generation Demo
# =========================
PROMPT ?= As an IC designer, I think the most challenging part of designing hardware for Transformer models is
MAX_TOKENS ?= 60

gen_llama_baseline:
	@echo "\n===================================================================="
	@echo ">>> Generating Text with Baseline (FP16 Original)"
	@echo "===================================================================="
	$(PYTHON) $(SCRIPT_T4) --precision fp16 --softmax torch --act torch --prompt "$(PROMPT)" --max_new_tokens $(MAX_TOKENS) --model_name meta-llama/Llama-2-7b-hf

gen_llama_pwl8:
	@echo "\n===================================================================="
	@echo ">>> Generating Text with DWMSE 8-Seg"
	@echo "===================================================================="
	$(PYTHON) $(SCRIPT_T4) --precision fp16 --softmax torch --act pwl-8 --prompt "$(PROMPT)" --max_new_tokens $(MAX_TOKENS) --model_name meta-llama/Llama-2-7b-hf

gen_llama_pwl16:
	@echo "\n===================================================================="
	@echo ">>> Generating Text with DWMSE 16-Seg"
	@echo "===================================================================="
	$(PYTHON) $(SCRIPT_T4) --precision fp16 --softmax torch --act pwl-16 --prompt "$(PROMPT)" --max_new_tokens $(MAX_TOKENS) --model_name meta-llama/Llama-2-7b-hf

demo_llama_gen: pwl_llama_8 pwl_llama_16 gen_llama_baseline gen_llama_pwl8 gen_llama_pwl16
	@echo "\n>>> LLaMA 2 Real Text Generation Comparison Completed!"

# =========================
# Utilities
# =========================
clean:
	@echo "--- Deep Cleaning Workspace ---"
	@rm -rf "$(LOG_DIR)" "$(TEST_LOG_DIR)" "$(DATE_DIR)" "$(PDF_DIR)" "$(PWL_DIR)" "$(PWL_MSE_DIR)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "--- Workspace is clean. ---"

help:
	@echo "======== Demos ========"
	@echo " make demo_gpt2             : Run complete GPT-2 pipeline (PPL)"
	@echo " make demo_llama_gen        : Run LLaMA 2 text generation comparison"
	@echo "   -> Use: make demo_llama_gen PROMPT=\"Your text here\" MAX_TOKENS=50"
	@echo "======== Utilities ========"
	@echo " make clean                 : Remove all generated JSONs and logs"