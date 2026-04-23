import os
import re
import csv
import sys
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
LOG_DIR = "dst_log"
OUTPUT_CSV = "exp3_results.csv"
# --- End Configuration ---

def parse_accuracy_from_log(log_content):
    """
    Parses the log file content to find the accuracy and precision.
    """
    # Regex to find: "Top-1 Accuracy on 50000 samples (FP32): 78.42%"
    acc_pattern = re.compile(r"Top-1 Accuracy on \d+ samples \((.+)\): (\d+\.\d+)%")
    match = acc_pattern.search(log_content)
    
    if match:
        precision = match.group(1)
        accuracy = float(match.group(2))
        return "OK", precision, accuracy
    
    # Check for known errors from the Makefile
    if "ERROR: Could not find bit-width info" in log_content or "Required bits not found" in log_content:
        return "ERROR_MISSING_BITS", "N/A", 0.0
        
    if "Traceback" in log_content:
        return "ERROR_TRACEBACK", "N/A", 0.0
        
    return "ERROR_PARSE_FAILED", "N/A", 0.0

def main():
    log_dir_path = Path(LOG_DIR)
    if not log_dir_path.is_dir():
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        print("Please run 'make test_exp3' first.")
        sys.exit(1)

    # Regex to parse filenames like:
    # test_exp3_1_torch_vit-tiny.log
    # test_exp3_5_both-pwl-fixed_swin-base_16.log
    filename_pattern = re.compile(
        r"test_exp3_(\d)_([a-zA-Z0-9-]+)_([a-zA-Z0-9-]+)(?:_(\d+))?.*\.log"
    )

    all_results = []
    print(f"Scanning {LOG_DIR} for exp3 log files...")

    for filename in os.listdir(log_dir_path):
        match = filename_pattern.match(filename)
        if not match:
            continue

        step, config, model, segments = match.groups()
        step = int(step)
        segments = int(segments) if segments else "N/A"
        
        log_file_path = log_dir_path / filename
        
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
            
            status, precision, accuracy = parse_accuracy_from_log(content)
            
            result = {
                "model": model,
                "step": step,
                "config_name": config,
                "segments": segments,
                "precision": precision,
                "accuracy": accuracy,
                "status": status,
                "log_file": filename
            }
            all_results.append(result)

        except FileNotFoundError:
            print(f"Warning: Could not find file {filename} during read, skipping.")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            all_results.append({
                "model": model, "step": step, "config_name": config,
                "segments": segments, "precision": "N/A", "accuracy": 0.0,
                "status": "ERROR_READING_FILE", "log_file": filename
            })

    if not all_results:
        print("No 'test_exp3' log files found. Nothing to compile.")
        return

    # Sort results for a clean CSV
    all_results.sort(key=lambda r: (r['model'], r['step']))

    # Write to CSV
    try:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            fieldnames = [
                "model", "step", "config_name", "segments", 
                "precision", "accuracy", "status", "log_file"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        
        print(f"\n✅ Successfully compiled {len(all_results)} results into {OUTPUT_CSV}")

    except PermissionError:
        print(f"\nError: Could not write to {OUTPUT_CSV}. Is the file open in another program?")
    except Exception as e:
        print(f"\nAn error occurred while writing CSV: {e}")

if __name__ == "__main__":
    main()