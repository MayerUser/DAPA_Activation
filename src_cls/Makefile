# --- Makefile for ViT PWL Approximation, Testing, and Analysis ---

# --- Global Configuration ---

# Use python3 by default. Can be changed to just "python" if needed.
PYTHON = python3

# CACHE_DIR: Optional. Set a shared Hugging Face cache directory.
# This will be passed to all python scripts that support it.
# Example: CACHE_DIR = /path/to/my/hf/cache
CACHE_DIR =

# --- Handle Optional Cache Dir ---
# This creates a CACHE_DIR_ARG variable that is either empty
# or "--cache_dir /path/to/cache"
CACHE_DIR_ARG =
ifneq ($(CACHE_DIR),)
	CACHE_DIR_ARG = --cache_dir $(CACHE_DIR)
endif

# --- Experiment Pipeline Configuration (EXP1, EXP3, EXP5) ---

# EXP1 : MSE vs Performance and DWMSE vs Performance
EXP1_MODELS = vit-tiny vit-small vit-base
EXP1_SEGMENTS = 4 6 8 10 12 14 16
EXP1_SAMPLES = 256
EXP1_POLY_ORDER = 4 5 6 7 8

# EXP2 : No of Samples vs ACC
EXP2_MODELS =  vit-small
EXP2_SEGMENTS = 4 6 8 16
EXP2_SAMPLES = 1 2 4 8 16 32 64

# EXP3 : Architecture (Image Classfication) vs. Performance Test Config 
EXP3_MODELS = vit-tiny vit-small vit-base deit-tiny deit-small deit-base swin-small swin-base
EXP3_SEGMENTS = 16 
EXP3_SAMPLES = 256

# EXP5 : Ablation Experiment
EXP5_MODELS = vit-tiny vit-small vit-base
EXP5_SEGMENTS = 4 8 12 16
EXP5_SAMPLES = 256

# EXP6 MSE
EXP6_MODELS = vit-tiny vit-small vit-base deit-tiny deit-small deit-base swin-small swin-base
EXP6_SEGMENTS = 8 16
EXP6_SAMPLES = 256

# --- Aggregated variables for generation ---
ALL_MODELS = $(sort $(EXP3_MODELS))
ALL_SEGMENTS = $(sort $(EXP1_SEGMENTS) $(EXP3_SEGMENTS) $(EXP5_SEGMENTS))
ALL_SAMPLES = $(sort $(EXP1_SAMPLES) $(EXP3_SAMPLES) $(EXP5_SAMPLES))


# --- Bit-Width Analysis Configuration (for m2_find_fixed_bit.py) ---

# THETA: The DWMSE error threshold multiplier.
# This can be overridden from the command line: make all-bits ANALYSIS_THETA=1.5
ANALYSIS_THETA = 1.5

# SAMPLES: The number of samples used to generate the PWL files
# (must match dst_pwl/ directory)
ANALYSIS_SAMPLES = 256

# SEGMENTS: The number of segments in the PWL files
# (must match filename)
ANALYSIS_SEGMENTS = 16

# Default model for a single 'find-bits' run
ANALYSIS_MODEL_DEFAULT = vit-tiny


# --- Phony Targets (Targets that are not files) ---
.PHONY: all pwl_img pwl_nlp poly test_exp1 test_exp3 finetune_exp4 test_exp4 test_exp5 clean demo ref
.PHONY: all-bits find-bits bits-%


# --- Main Experiment Targets ---

# Default target: Run the entire pipeline from generation to testing.
all: clean pwl_img poly test_exp1 test_exp3 test_exp5

define run_pwl_generation
	@mkdir -p dst_log
	$(foreach sample,$(1), \
		$(foreach model,$(2), \
			$(foreach seg,$(3), \
				if [ ! -f dst_log/pwl_$(model)_$(seg)_$(sample).log ]; then \
					echo ">> Generating IMG PWL for $(model)-$(seg)seg-$(sample)samples..."; \
					$(PYTHON) t0_make_pwl.py --model_name $(model) --segment_number $(seg) --num_samples $(sample) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/pwl_$(model)_$(seg)_$(sample).log; \
				else \
					echo ">> Skipping IMG PWL generation for $(model)-$(seg)seg-$(sample)samples (log exists)."; \
				fi; \
			) \
		) \
	)
endef

define run_pwl_generation_mse
	@mkdir -p dst_log
	$(foreach sample,$(1), \
		$(foreach model,$(2), \
			$(foreach seg,$(3), \
				if [ ! -f dst_log/pwl_$(model)_$(seg)_$(sample)_mse.log ]; then \
					echo ">> Generating IMG MSE PWL for $(model)-$(seg)seg-$(sample)samples..."; \
					$(PYTHON) t0_make_pwl.py --loss mse --model_name $(model) --segment_number $(seg) --num_samples $(sample) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/pwl_$(model)_$(seg)_$(sample)_mse.log; \
				else \
					echo ">> Skipping IMG PWL generation for $(model)-$(seg)seg-$(sample)samples_mse (log exists)."; \
				fi; \
			) \
		) \
	)
endef

# Generate all necessary PWL approximation files for Image models.
pwl_img:
	@echo "--- Generating Piecewise (PWL) approximation files for Image Models... ---"
	@mkdir -p dst_log dst_pdf dst_plot dst_pwl
	$(call run_pwl_generation,$(EXP1_SAMPLES),$(EXP1_MODELS),$(EXP1_SEGMENTS))
	@echo "--- Image PWL file generation complete. ---"

# $(call run_pwl_generation,$(EXP1_SAMPLES),$(EXP1_MODELS),$(EXP1_SEGMENTS))
# $(call run_pwl_generation,$(EXP2_SAMPLES),$(EXP2_MODELS),$(EXP2_SEGMENTS))
# $(call run_pwl_generation,$(EXP3_SAMPLES),$(EXP3_MODELS),$(EXP3_SEGMENTS))
# $(call run_pwl_generation,$(EXP5_SAMPLES),$(EXP5_MODELS),$(EXP5_SEGMENTS))
# $(call run_pwl_generation,$(EXP6_SAMPLES),$(EXP6_MODELS),$(EXP6_SEGMENTS))

pwl_img_mse:
	@echo "--- Generating Piecewise (PWL) approximation files for Image Models... ---"
	@mkdir -p dst_log dst_pdf dst_plot dst_pwl
	$(call run_pwl_generation_mse,$(EXP6_SAMPLES),$(EXP6_MODELS),$(EXP6_SEGMENTS))
	@echo "--- Image PWL file generation complete. ---"


# Generate all necessary Polynomial approximation files for GELU.
poly:
	@echo "--- Generating Polynomial (POLY) approximation files for GELU... ---"
	@mkdir -p dst_log dst_poly dst_plot
	@$(foreach order,$(EXP1_POLY_ORDER), \
		if [ ! -f dst_log/poly_gelu_$(order).log ]; then \
			echo ">> Generating POLY files for GELU with order $(order)..."; \
			$(PYTHON) t2_make_poly.py --order $(order) 2>&1 | tee dst_log/poly_gelu_$(order).log; \
		else \
			echo ">> Skipping POLY generation for GELU order $(order) (log exists)."; \
		fi; \
	)
	@echo "--- POLY file generation complete. ---"

# Run EXP1 tests for PWL and Polynomial GELU.
test_exp1:
	@echo ""
	@echo "--- Running EXP1: Performance Tests ---"
	@mkdir -p dst_log
	# --- Part A: Test PWL implementations ---
	@echo ""
	@echo ">> EXP1 - Part A: Testing PWL implementations"
	@$(foreach model,$(EXP1_MODELS), \
		$(foreach seg,$(EXP1_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp1_pwl_$(model)_$(seg)_$(EXP1_SAMPLES).log ]; then \
				echo ">> Testing $(model) with PWL-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax pwl-$(seg) --act pwl-$(seg) --num_samples $(EXP1_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp1_pwl_$(model)_$(seg)_$(EXP1_SAMPLES).log; \
			else \
				echo ">> Skipping EXP1 PWL test for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)
	# --- Part B: Test Polynomial GELU implementations ---
	@echo ""
	@echo ">> EXP1 - Part B: Testing Polynomial GELU Performance"
	@$(foreach model,$(EXP1_MODELS), \
		$(foreach order,$(EXP1_POLY_ORDER), \
			echo ""; \
			if [ ! -f dst_log/test_exp1_poly_$(model)_$(order)_$(EXP1_SAMPLES).log ]; then \
				echo ">> Testing $(model) with poly-$(order) GELU..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --act poly-$(order) --num_samples $(EXP1_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp1_poly_$(model)_$(order)_$(EXP1_SAMPLES).log; \
			else \
				echo ">> Skipping EXP1 POLY test for $(model) with poly-$(order) (log exists)."; \
			fi; \
		) \
	)
	@echo ""
	@echo "--- EXP1 tests complete. ---"


test_exp2:
	@echo ""
	@echo "--- Running EXP2: Image Classification Test---"
	@mkdir -p dst_log
	# Test baseline (all torch) for each model
	@$(foreach seg,$(EXP2_SEGMENTS), \
		$(foreach sample,$(EXP2_SAMPLES), \
		echo ""; \
		if [ ! -f dst_log/test_exp2_$(EXP2_MODELS)_$(seg)_$(sample).log ]; then \
			echo ">> Testing $(EXP2_MODELS)_$(seg)_$(sample) with torch ..."; \
			$(PYTHON) t1_vit_run.py --model_name $(EXP2_MODELS) --num_samples $(sample)  --act pwl-$(seg) 2>&1 | tee dst_log/test_exp2_$(EXP2_MODELS)_$(seg)_$(sample).log; \
		else \
			echo ">> Skipping EXP3 torch test for $(EXP2_MODELS)_$(seg)_$(sample) (log exists)."; \
		fi; \
		) \
	)

# --- REVISED EXP3 ---
# Run EXP3: Image Classification Architecture vs. Performance tests.
test_exp3:
	@echo ""
	@echo "--- Running EXP3: Image Classification Architecture Tests (5 Steps) ---"
	@mkdir -p dst_log
	@$(foreach seg,$(EXP3_SEGMENTS), \
		$(foreach model,$(EXP3_MODELS), \
			echo ""; \
			echo "--- Processing EXP3 for $(model) with $(seg) segments ---"; \
			if [ ! -f dst_log/test_exp3_1_torch_$(model).log ]; then \
				echo ">> EXP3 [1/5] Testing $(model) with torch (fp32)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --sqrt torch --softmax torch --act torch --precision fp32 $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp3_1_torch_$(model).log; \
			else \
				echo ">> Skipping EXP3 [1/5] torch test for $(model) (log exists)."; \
			fi; \
			if [ ! -f dst_log/test_exp3_2_sm-pwl_$(model)_$(seg).log ]; then \
				echo ">> EXP3 [2/5] Testing $(model) with Softmax=pwl-$(seg), Act=torch..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax pwl-$(seg) --act torch --num_samples $(EXP3_SAMPLES) --precision fp32 $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp3_2_sm-pwl_$(model)_$(seg).log; \
			else \
				echo ">> Skipping EXP3 [2/5] sm-pwl test for $(model) (log exists)."; \
			fi; \
			if [ ! -f dst_log/test_exp3_3_act-pwl_$(model)_$(seg).log ]; then \
				echo ">> EXP3 [3/5] Testing $(model) with Softmax=torch, Act=pwl-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax torch --act pwl-$(seg) --num_samples $(EXP3_SAMPLES) --precision fp32 $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp3_3_act-pwl_$(model)_$(seg).log; \
			else \
				echo ">> Skipping EXP3 [3/5] act-pwl test for $(model) (log exists)."; \
			fi; \
			if [ ! -f dst_log/test_exp3_4_both-pwl-float_$(model)_$(seg).log ]; then \
				echo ">> EXP3 [4/5] Testing $(model) with Softmax=pwl-$(seg), Act=pwl-$(seg) (FLOAT)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax pwl-$(seg) --act pwl-$(seg) --num_samples $(EXP3_SAMPLES) --precision fp32 $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp3_4_both-pwl-float_$(model)_$(seg).log; \
			else \
				echo ">> Skipping EXP3 [4/5] both-pwl-float test for $(model) (log exists)."; \
			fi; \
			if [ ! -f dst_log/test_exp3_5_both-pwl-fixed_$(model)_$(seg).log ]; then \
				echo ">> EXP3 [5/5] Testing $(model) with Softmax=pwl-$(seg), Act=pwl-$(seg) (FIXED-POINT)..."; \
				ACT_Q=$$(grep $(model) bit_config_results.csv | grep "Activation (GELU/SiLU)" | awk -F, '{gsub(/ /, "", $$4); gsub(/ /, "", $$5); print $$4"."$$5}'); \
				SM_Q=$$(grep $(model) bit_config_results.csv | grep "Softmax (EXP)" | awk -F, '{gsub(/ /, "", $$4); gsub(/ /, "", $$5); print $$4"."$$5}'); \
				if [ -z "$$ACT_Q" ] || [ -z "$$SM_Q" ]; then \
					echo ">> ERROR: Could not find bit-width info for $(model) in bit_config_results.csv. Run 'make all-bits' first."; \
					echo "Required bits not found" > dst_log/test_exp3_5_both-pwl-fixed_$(model)_$(seg).log; \
				else \
					echo ">> Found bits: --act_q $$ACT_Q --sm_q $$SM_Q"; \
					$(PYTHON) t3_vit_run_fixed.py --model_name $(model) --softmax pwl-$(seg) --act pwl-$(seg) --num_samples $(EXP3_SAMPLES) --act_q $$ACT_Q --sm_q $$SM_Q $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp3_5_both-pwl-fixed_$(model)_$(seg).log; \
				fi; \
			else \
				echo ">> Skipping EXP3 [5/5] both-pwl-fixed test for $(model) (log exists)."; \
			fi; \
		) \
	)
	@echo ""
	@echo "--- EXP3 tests complete. ---"
# --- END REVISED EXP3 ---


# Run EXP5: Ablation Experiments
test_exp5:
	@echo ""
	@echo "--- Running EXP5: Ablation Experiments ---"
	@mkdir -p dst_log

	# --- Option 1: Softmax=torch, Act=PWL ---
	@echo ""
	@echo ">> EXP5 - Option 1: Softmax=torch, Act=PWL"
	@$(foreach model,$(EXP5_MODELS), \
		$(foreach seg,$(EXP5_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp5_opt1_$(model)_$(seg)_$(EXP5_SAMPLES).log ]; then \
				echo ">> Testing $(model) with softmax=torch, act=pwl-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax torch --act pwl-$(seg) --num_samples $(EXP5_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp5_opt1_$(model)_$(seg)_$(EXP5_SAMPLES).log; \
			else \
				echo ">> Skipping EXP5-Opt1 for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)
	# --- Option 2: Softmax=PWL, Act=torch ---
	@echo ""
	@echo ">> EXP5 - Option 2: Softmax=PWL, Act=torch"
	@$(foreach model,$(EXP5_MODELS), \
		$(foreach seg,$(EXP5_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp5_opt2_$(model)_$(seg)_$(EXP5_SAMPLES).log ]; then \
				echo ">> Testing $(model) with softmax=pwl-$(seg), act=torch..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax pwl-$(seg) --act torch --num_samples $(EXP5_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp5_opt2_$(model)_$(seg)_$(EXP5_SAMPLES).log; \
			else \
				echo ">> Skipping EXP5-Opt2 for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)

	# --- Option 3: Softmax=PWL, Act=PWL ---
	@echo ""
	@echo ">> EXP5 - Option 3: Softmax=PWL, Act=PWL"
	@$(foreach model,$(EXP5_MODELS), \
		$(foreach seg,$(EXP5_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp5_opt3_$(model)_$(seg)_$(EXP5_SAMPLES).log ]; then \
				echo ">> Testing $(model) with softmax=pwl-$(seg), act=pwl-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax pwl-$(seg) --act pwl-$(seg) --num_samples $(EXP5_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp5_opt3_$(model)_$(seg)_$(EXP5_SAMPLES).log; \
			else \
				echo ">> Skipping EXP5-Opt3 for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)

	@echo ""
	@echo "--- EXP5 tests complete. ---"

test_exp6:
	@echo ""
	@echo "--- Running EXP6: MSE ---"
	@mkdir -p dst_log
	@echo ""
	@echo ">> EXP6 - Option 1: Act=PWL Base on DWMSE"
	@$(foreach model,$(EXP6_MODELS), \
		$(foreach seg,$(EXP6_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp6_dwmse_$(model)_$(seg)_$(EXP6_SAMPLES).log ]; then \
				echo ">> Testing $(model) with softmax=torch, act=pwl-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --model_name $(model) --softmax torch --act pwl-$(seg) --num_samples $(EXP6_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp6_dwmse_$(model)_$(seg)_$(EXP6_SAMPLES).log; \
			else \
				echo ">> Skipping EXP6-DWMSE for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)
	@echo ">> EXP6 - Option 1: Act=PWL Base on MSE"
	@$(foreach model,$(EXP6_MODELS), \
		$(foreach seg,$(EXP6_SEGMENTS), \
			echo ""; \
			if [ ! -f dst_log/test_exp6_mse_$(model)_$(seg)_$(EXP6_SAMPLES).log ]; then \
				echo ">> Testing $(model) with softmax=torch, act=pwl-$(seg)..."; \
				$(PYTHON) t1_vit_run.py --loss mse --model_name $(model) --softmax torch --act pwl-$(seg) --num_samples $(EXP6_SAMPLES) $(CACHE_DIR_ARG) 2>&1 | tee dst_log/test_exp6_mse_$(model)_$(seg)_$(EXP6_SAMPLES).log; \
			else \
				echo ">> Skipping EXP6-DWMSE for $(model)-$(seg)seg (log exists)."; \
			fi; \
		) \
	)


demo:
	@echo "--- RUNNING VIT DAPU TEST ---"
	@mkdir -p dst_log
	@$(PYTHON) t1_vit_run.py --model_name vit-small --softmax pwl-16 --act pwl-16 --num_samples 256 $(CACHE_DIR_ARG) 2>&1 | tee dst_log/demo.log

ref:
	@echo "--- RUNNING VIT TEST ---"
	@mkdir -p dst_log
	@$(PYTHON) t1_vit_run.py --model_name vit-small $(CACHE_DIR_ARG) 2>&1 | tee dst_log/ref.log


# --- Bit-Width Analysis Targets (from m2_find_fixed_bit.py) ---

# Runs the analysis for *all* models defined in EXP3_MODELS.
# Usage: make all-bits
# Usage with different theta: make all-bits ANALYSIS_THETA=1.5
all-bits: $(addprefix bits-,$(EXP3_MODELS))
	@echo "--- All model bit-width analyses complete. ---"

# Runs the analysis for *only* the ANALYSIS_MODEL_DEFAULT.
# Good for a quick single test.
# Usage: make find-bits
find-bits:
	@echo "--- Running bit-width analysis for single model: $(ANALYSIS_MODEL_DEFAULT) ---"
	$(PYTHON) m2_find_fixed_bit.py \
		--model_name $(ANALYSIS_MODEL_DEFAULT) \
		--num_samples $(ANALYSIS_SAMPLES) \
		--segment_number $(ANALYSIS_SEGMENTS) \
		--theta $(ANALYSIS_THETA) \
		$(CACHE_DIR_ARG)

# --- Pattern Rule for All Models ---
# This is the rule that 'all-bits' uses.
# You can also run it for one specific model.
# Example: make bits-vit-tiny
bits-%:
	@echo "--- Running bit-width analysis for model: $* ---"
	$(PYTHON) m2_find_fixed_bit.py \
		--model_name $* \
		--num_samples $(ANALYSIS_SAMPLES) \
		--segment_number $(ANALYSIS_SEGMENTS) \
		--theta $(ANALYSIS_THETA) \
		$(CACHE_DIR_ARG)


# --- Utility Targets ---

# Clean up all generated directories and files.
clean:
	@echo "--- Cleaning up generated directories... ---"
	@rm -rf dst_pdf dst_plot dst_pwl dst_poly dst_log __pycache__ dst_chk
	@rm -f bit_config_results.csv
	@echo "--- Cleanup complete. ---"


# --- Utility Targets ---

# Help target: Displays available commands and a brief description.
help:
	@echo "--- Makefile Help & Targets ---"
	@echo "This Makefile manages the full experiment pipeline for ViT PWL approximation and testing."
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "--- 1. FULL PIPELINE & EXPERIMENT TESTS ---"
	@echo "  all             : Runs the complete experiment pipeline: clean, generate PWL/Poly files, and run EXP1, EXP3, and EXP5 tests."
	@echo "  pwl_img         : Generates Piecewise Linear (PWL) approximation files for Image models (DWMSE loss)."
	@echo "  pwl_img_mse     : Generates PWL approximation files for Image models (MSE loss, for EXP6)."
	@echo "  poly            : Generates Polynomial approximation files for GELU."
	@echo "  test_exp1       : Runs EXP1: Performance tests comparing DWMSE PWL and Polynomial GELU approximations."
	@echo "  test_exp3       : Runs EXP3: Architecture classification tests (FP32 and Fixed-Point) across various models."
	@echo "  test_exp5       : Runs EXP5: Ablation study comparing single vs. combined PWL approximation for Softmax/GELU."
	@echo "  test_exp6       : Runs EXP6: Tests comparing DWMSE-based vs. MSE-based PWL files."
	@echo "  demo            : Runs a quick demo test (vit-small with PWL-16 for Softmax/Activation)."
	@echo "  ref             : Runs a reference test (vit-small using PyTorch FP32 baseline)."	@echo ""
	@echo "--- 3. UTILITY ---"
	@echo "  clean           : Removes all generated files and directories (dst_pdf, dst_log, dst_pwl, etc.)."
	@echo "  help            : Displays this help message."

# You must also add 'help' to your .PHONY list to ensure it works correctly.

.PHONY: help pwl_img_mse test_exp2 test_exp6