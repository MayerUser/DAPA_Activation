# =========================
# Makefile (GPT-2 PPL)
# =========================

PYTHON        ?= python3
SHELL         := /bin/bash

# --- Scripts ---
SCRIPT_PWL    := t0_make_pwl.py
SCRIPT_T2     := t2_gpt2_run.py

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
CACHE_DIR   ?= $(HOME)/MyWorkspace/HfHome
PRECISION   ?= fp16   # fp32 | fp16

# --- Params for generators ---
MODEL        ?=
SEGMENTS     ?=
LOSS         ?= dwmse

# --- Generated file flags ---
FLAG_FILE     = $(DATE_DIR)/$(DATE)-$(MODEL)-$(SEGMENTS)-$(LOSS)-n$(NUM_SAMPLES).success
LOG_FILE      = $(LOG_DIR)/$(DATE)-$(MODEL)-$(SEGMENTS)-$(LOSS)-n$(NUM_SAMPLES).log

# --- Phony ---
.PHONY: all help clean \
        run \
        test_gpt2_baseline test_gpt2_pwl8 test_gpt2_pwl16 demo_gpt2

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
ifneq ($(MODEL),gpt2)
	$(error NLP Makefile only supports MODEL=gpt2 for GPT-2 PPL evaluation)
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
	@echo "======== Utilities ========"
	@echo " make clean                 : Remove all generated JSONs and logs"
