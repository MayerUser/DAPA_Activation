# m2_find_fixed_bit.py

import argparse
import json
import copy
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import csv
import os

# --- New: CSV Output Configuration ---
CSV_FILE = "bit_config_results.csv"

# --- Original Activation Functions (for comparison) ---

def original_gelu(x):
    """Numpy implementation of the 'new' GELU approximation (used by ViT/DeiT)."""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def original_silu(x):
    """Numpy implementation of SiLU (Swish, used by BERT)."""
    return x / (1 + np.exp(-x))

# --- Helper Functions ---

# --- REMOVED: run_model_to_capture_data ---

def evaluate_pwl_json(x, pwl_data):
    """
    Evaluates the PWL function defined in the JSON data for a numpy array x.
    """
    x = np.asarray(x)
    y = np.zeros_like(x)
    
    boundaries = [float(interval[1]) for interval in pwl_data["intervals"][:-1]]
    params_list = pwl_data["params"]
    
    p0 = np.array([p.get('p0', 0) for p in params_list])
    p1 = np.array([p.get('p1', 0) for p in params_list])
    p2 = np.array([p.get('p2', 0) for p in params_list])
    p3 = np.array([p.get('p3', 0) for p in params_list])

    indices = np.searchsorted(boundaries, x)
    
    p0_x = p0[indices]
    p1_x = p1[indices]
    p2_x = p2[indices]
    p3_x = p3[indices]
    
    y = p3_x * (x**3) + p2_x * (x**2) + p1_x * x + p0_x
    return y

def quantize_numpy(value, integer_bits, fraction_bits):
    """
    Performs symmetric Q-format quantization (2's complement) on NumPy arrays.
    """
    scale = 2.0**fraction_bits
    total_bits = integer_bits + fraction_bits
    
    q_max = (2.0**(total_bits - 1) - 1)
    q_min = -(2.0**(total_bits - 1))
    
    scaled_val = np.round(value * scale)
    clipped_val = np.clip(scaled_val, q_min, q_max)
    return clipped_val / scale

def create_fix16_coeffs(pwl_data):
    """
    Quantizes PWL coefficients to Fix16, mirroring m3_udanf_fixed.py logic.
    """
    all_coeffs = []
    for p in pwl_data["params"]:
        all_coeffs.extend([p.get("p3", 0.0), p.get("p2", 0.0), p.get("p1", 0.0), p.get("p0", 0.0)])
    
    max_abs_coeff = np.max(np.abs(all_coeffs))
    TOTAL_BITS_COEFF = 16
    i_bits_coeff = int(np.ceil(np.log2(max_abs_coeff + 1e-9))) + 1
    
    if i_bits_coeff > TOTAL_BITS_COEFF:
        i_bits_coeff = TOTAL_BITS_COEFF
        f_bits_coeff = 0
    else:
        f_bits_coeff = TOTAL_BITS_COEFF - i_bits_coeff
    
    quant_pwl_data = copy.deepcopy(pwl_data)
    for p in quant_pwl_data["params"]:
        p['p3'] = quantize_numpy(p.get('p3', 0.0), i_bits_coeff, f_bits_coeff)
        p['p2'] = quantize_numpy(p.get('p2', 0.0), i_bits_coeff, f_bits_coeff)
        p['p1'] = quantize_numpy(p.get('p1', 0.0), i_bits_coeff, f_bits_coeff)
        p['p0'] = quantize_numpy(p.get('p0', 0.0), i_bits_coeff, f_bits_coeff)
        
    return quant_pwl_data, i_bits_coeff, f_bits_coeff


def calculate_dwmse(
    hist, bin_edges, original_func,
    eval_pwl_data, 
    quant_params_data=None
):
    """
    Calculates DWMSE.
    - eval_pwl_data: The PWL model to use (can be float or pre-quantized).
    - quant_params_data: (i_bits, f_bits) to apply symmetrically to x and y.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    y_true = original_func(bin_centers)
    
    if quant_params_data:
        # --- Apply symmetric data quantization ---
        i_bits_data, f_bits_data = quant_params_data
        
        x_quant = quantize_numpy(bin_centers, i_bits_data, f_bits_data)
        y_approx_pre_quant = evaluate_pwl_json(x_quant, eval_pwl_data)
        y_approx = quantize_numpy(y_approx_pre_quant, i_bits_data, f_bits_data)
        
    else:
        # --- Baseline (float data) ---
        y_approx = evaluate_pwl_json(bin_centers, eval_pwl_data)
        
    squared_error = (y_true - y_approx)**2
    weighted_squared_error = squared_error * hist
    dwmse = np.sum(weighted_squared_error * bin_width)
    
    return dwmse

def write_csv_header_if_not_exists():
    """Creates the CSV file and writes the header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "function", "status", 
                "i_bits_data", "f_bits_data", "total_bits_data",
                "i_bits_coeff", "f_bits_coeff",
                "final_dwmse", "theta(multiplier)", "real_threshold",
                "baseline_float_dwmse", "baseline_coeff_dwmse"
            ])

def write_csv_row(row_data):
    """Appends a single row of data to the CSV file."""
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            row_data.get("model_name", ""),
            row_data.get("function", ""),
            row_data.get("status", ""),
            row_data.get("i_bits_data", ""),
            row_data.get("f_bits_data", ""),
            row_data.get("total_bits_data", ""),
            row_data.get("i_bits_coeff", ""),
            row_data.get("f_bits_coeff", ""),
            row_data.get("final_dwmse", ""),
            row_data.get("theta(multiplier)", ""),
            row_data.get("real_threshold", ""),
            row_data.get("baseline_float_dwmse", ""),
            row_data.get("baseline_coeff_dwmse", "")
        ])

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Calculate DWMSE and find fixed-point bits for PWL functions.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model name (e.g., gpt2) to get the correct distribution.")
    parser.add_argument("--num_samples", type=int, default=256,
                        help="Number of samples (e.g., 256) to get the correct distribution.")
    parser.add_argument("--segment_number", type=int, default=16,
                        help="Number of PWL segments (e.g., 16).")
    parser.add_argument("--theta", type=float, default=1.05,
                        help="DWMSE *multiplier* (e.g., 1.05 = allow 5%% more error than coeff-quant baseline).")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Path to a shared Hugging Face cache directory.")
    args = parser.parse_args()

    # --- New: Create CSV file/header ---
    write_csv_header_if_not_exists()

    # --- REMOVED: Step 1 (run_model_to_capture_data) ---

    tasks = [
        {
            "name": "Activation (GELU/SiLU)",
            "data_key": "gelu_input",
            "json_key": "gelu_act",
            "original_func": original_silu if "bert" in args.model_name else original_gelu
        },
        {
            "name": "Softmax (EXP)",
            "data_key": "softmax_input",
            "json_key": "exp_sm",
            "original_func": np.exp
        }
    ]

    for task in tasks:
        print(f"\n========================================================")
        print(f"--- Processing Function: {task['name']} ---")
        print(f"========================================================")

        csv_row = {
            "model_name": args.model_name,
            "function": task['name'],
            "theta(multiplier)": args.theta
        }

        # --- MODIFIED: Step 2: Load PDF and JSON from disk ---
        json_path_str = f"dst_pwl/{args.num_samples}/pwl_{task['json_key']}_{args.model_name}_{args.segment_number}seg.json"
        pdf_path_str = f"dst_pdf/256/pdf_{task['json_key']}_{args.model_name}_{args.num_samples}samples.npz"
        
        json_path = Path(json_path_str)
        pdf_path = Path(pdf_path_str)
        
        if not json_path.is_file():
            print(f"Error: JSON file not found at {json_path}")
            print(f"       -> Run 'make pwl_img' first.")
            csv_row["status"] = "ERROR_NO_JSON"
            write_csv_row(csv_row)
            continue
            
        if not pdf_path.is_file():
            print(f"Error: PDF data file not found at {pdf_path}")
            print(f"       -> Run 'make pwl_img' first.")
            csv_row["status"] = "ERROR_NO_PDF"
            write_csv_row(csv_row)
            continue

        print(f"Loading PWL data from: {json_path}")
        with open(json_path, 'r') as f:
            pwl_data_float = json.load(f)

        print(f"Loading PDF data from: {pdf_path}")
        pdf_data = np.load(pdf_path)
        hist = pdf_data['hist']
        bin_edges = pdf_data['bin_edges']
        
        # Get the true min/max from the saved data
        max_abs_min = np.abs(pdf_data['global_min'])
        max_abs_max = np.abs(pdf_data['global_max'])
        max_x = np.max([max_abs_min, max_abs_max])
        
        original_func = task['original_func']
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        print("Distribution loaded and PDF generated.")
        # --- END MODIFIED STEP 2 ---


        # --- Step 3: Calculate Baselines ---
        print("\n--- Step 3: Calculating Baselines ---")
        
        baseline_float_dwmse = calculate_dwmse(
            hist, bin_edges, original_func,
            eval_pwl_data=pwl_data_float
        )
        csv_row["baseline_float_dwmse"] = baseline_float_dwmse
        print(f"Baseline DWMSE (Pure Float): {baseline_float_dwmse: .6e}")
        
        pwl_data_fix16_coeff, i_c, f_c = create_fix16_coeffs(pwl_data_float)
        csv_row["i_bits_coeff"], csv_row["f_bits_coeff"] = i_c, f_c
        
        baseline_coeff_dwmse = calculate_dwmse(
            hist, bin_edges, original_func,
            eval_pwl_data=pwl_data_fix16_coeff
        )
        csv_row["baseline_coeff_dwmse"] = baseline_coeff_dwmse
        print(f"Baseline DWMSE (Coeff-Quant): {baseline_coeff_dwmse: .6e}")
        
        real_threshold = args.theta * baseline_coeff_dwmse
        csv_row["real_threshold"] = real_threshold
        print(f"Real Threshold ({args.theta} * Coeff-Quant): {real_threshold: .6e}")

        # --- Step 4: Find Symmetric Fixed-Point Bits (16-bit total) ---
        print("\n--- Step 4: Finding Symmetric 16-Bit Fixed-Point Allocation ---")
        
        # del all_data (No longer needed, memory is safe)
        
        y_true_range = original_func(bin_centers)
        y_approx_range = evaluate_pwl_json(bin_centers, pwl_data_fix16_coeff)
        max_y = np.max(np.abs(np.concatenate([y_true_range, y_approx_range])))
        
        max_data = np.max([max_x, max_y])
        i_bits_data = int(np.ceil(np.log2(max_data + 1e-9))) + 1 if max_data > 0 else 1
        
        print(f"Max input (X) abs value:  {max_x:.4f}")
        print(f"Max output (Y) abs value: {max_y:.4f}")
        print(f"-> Combined Integer Bits for Data (X/Y): {i_bits_data}")
        
        csv_row["i_bits_data"] = i_bits_data

        MAX_TOTAL_BITS = 16
        max_f_bits = MAX_TOTAL_BITS - i_bits_data
        
        if max_f_bits < 0:
            print(f"\n❌ ERROR: Data range (I-Bits: {i_bits_data}) is too large for 16-bit total.")
            print("Cannot find a valid bit configuration.")
            csv_row["status"] = "ERROR_RANGE_TOO_LARGE"
            csv_row["f_bits_data"] = "N/A"
            csv_row["total_bits_data"] = "N/A"
            csv_row["final_dwmse"] = "N/A"
            write_csv_row(csv_row)
            continue

        print(f"\nSearching for Fraction Bits (Max F-bits = {max_f_bits}, Total Bits <= 16)...")
        found = False
        final_quant_dwmse = -1.0
        
        for f_bits in range(0, max_f_bits + 1):
            quant_params_data = (i_bits_data, f_bits)
            
            quant_dwmse = calculate_dwmse(
                hist, bin_edges, original_func,
                eval_pwl_data=pwl_data_fix16_coeff,
                quant_params_data=quant_params_data
            )
            final_quant_dwmse = quant_dwmse 
            
            print(f"  - Testing Q{i_bits_data}.{f_bits} (Total: {i_bits_data+f_bits} bits): DWMSE = {quant_dwmse: .6e}")
            
            if quant_dwmse < real_threshold:
                print(f"\n✅ Success! Threshold reached for {task['name']}.")
                csv_row["status"] = "SUCCESS"
                csv_row["f_bits_data"] = f_bits
                csv_row["total_bits_data"] = i_bits_data + f_bits
                csv_row["final_dwmse"] = quant_dwmse
                found = True
                break
        
        if not found:
            print(f"\n❌ Search finished. Threshold not met.")
            print(f"   Using best-effort Fix16 config: Q{i_bits_data}.{max_f_bits}")
            csv_row["status"] = "FAIL_USING_FIX16"
            csv_row["f_bits_data"] = max_f_bits
            csv_row["total_bits_data"] = i_bits_data + max_f_bits
            csv_row["final_dwmse"] = final_quant_dwmse
        
        write_csv_row(csv_row)

if __name__ == "__main__":
    main()