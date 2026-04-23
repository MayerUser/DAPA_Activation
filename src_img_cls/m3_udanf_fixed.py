# m3_udanf_fixed.py

import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# --- PyTorch-based Quantization Function ---

def quantize_torch(tensor: torch.Tensor, i_bits: int, f_bits: int) -> torch.Tensor:
    """
    Performs symmetric Q-format quantization on a PyTorch tensor.
    """
    scale = 2.0**f_bits
    total_bits = i_bits + f_bits
    
    # Calculate min/max values for signed representation
    q_max = (2.0**(total_bits - 1) - 1)
    q_min = -(2.0**(total_bits - 1))
    
    scaled_val = torch.round(tensor * scale)
    clipped_val = torch.clamp(scaled_val, q_min, q_max)
    return clipped_val / scale

# --- Base Class for Fixed-Point PWL ---

class PWLFunctionFixed(nn.Module):
    """
    A generic, device-aware PyTorch module for evaluating piecewise polynomial functions
    that simulates fixed-point (Q-number) quantization on:
    1. Coefficients (to Fix16) on initialization.
    2. Data (x, y) on every forward pass.
    """
    def __init__(self, json_path: str, i_bits_x: int, f_bits_x: int, i_bits_y: int, f_bits_y: int):
        super().__init__()
        
        if not Path(json_path).is_file():
            raise FileNotFoundError(f"JSON file not found at: {json_path}")

        with open(json_path, 'r') as f:
            pwl_data = json.load(f)
        
        print(f"  - Creating FIXED-POINT (Data+Coeff) module from {Path(json_path).name}")
        print(f"  - Data Path: (Q{i_bits_x}.{f_bits_x}) -> (Q{i_bits_y}.{f_bits_y})")

        # Store data quantization parameters
        self.i_bits_x = i_bits_x
        self.f_bits_x = f_bits_x
        self.i_bits_y = i_bits_y
        self.f_bits_y = f_bits_y

        # --- New Coefficient Quantization Logic ---
        
        # 1. Extract all float coefficients to find range
        all_coeffs = []
        params_float = pwl_data["params"]
        num_segments = len(params_float)
        
        p3_float = torch.zeros(num_segments)
        p2_float = torch.zeros(num_segments)
        p1_float = torch.zeros(num_segments)
        p0_float = torch.zeros(num_segments)

        for i, p in enumerate(params_float):
            p3_f, p2_f, p1_f, p0_f = p.get("p3", 0.0), p.get("p2", 0.0), p.get("p1", 0.0), p.get("p0", 0.0)
            p3_float[i], p2_float[i], p1_float[i], p0_float[i] = p3_f, p2_f, p1_f, p0_f
            all_coeffs.extend([p3_f, p2_f, p1_f, p0_f])

        # 2. Find max abs value
        max_abs_coeff = np.max(np.abs(all_coeffs))
        
        # 3. Determine "fix16" bit widths for coefficients
        TOTAL_BITS_COEFF = 16
        # Calculate integer bits needed (magnitude + sign)
        i_bits_coeff = int(np.ceil(np.log2(max_abs_coeff + 1e-9))) + 1 # +1 for sign

        if i_bits_coeff > TOTAL_BITS_COEFF:
            print(f"  - WARNING: Max coeff {max_abs_coeff:.4f} requires {i_bits_coeff} integer bits, but total is only {TOTAL_BITS_COEFF}. Clamping.")
            i_bits_coeff = TOTAL_BITS_COEFF
            f_bits_coeff = 0
        else:
            f_bits_coeff = TOTAL_BITS_COEFF - i_bits_coeff
        
        print(f"  - Coefficient Path: Quantizing to Q{i_bits_coeff}.{f_bits_coeff} (Fix{TOTAL_BITS_COEFF})")
        print(f"  - (Max coeff abs value: {max_abs_coeff:.4f})")

        # 4. Quantize and store coefficients
        p3_quant = quantize_torch(p3_float, i_bits_coeff, f_bits_coeff)
        p2_quant = quantize_torch(p2_float, i_bits_coeff, f_bits_coeff)
        p1_quant = quantize_torch(p1_float, i_bits_coeff, f_bits_coeff)
        p0_quant = quantize_torch(p0_float, i_bits_coeff, f_bits_coeff)
        
        # --- End New Logic ---

        # Convert string intervals to float boundaries
        boundaries = []
        for interval in pwl_data["intervals"][:-1]: # Exclude the last (+inf) boundary
            boundary_val = float(interval[1])
            boundaries.append(boundary_val)
        
        # Register *quantized* tensors as buffers
        self.register_buffer('boundaries', torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer('p3', p3_quant) # Storing the quantized version
        self.register_buffer('p2', p2_quant)
        self.register_buffer('p1', p1_quant)
        self.register_buffer('p0', p0_quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize input (x)
        x_quant = quantize_torch(x, self.i_bits_x, self.f_bits_x)

        # 2. Evaluate PWL (with float coefficients) using quantized input
        indices = torch.bucketize(x_quant, self.boundaries)

        # Get the pre-quantized coefficients
        p3 = self.p3[indices]
        p2 = self.p2[indices]
        p1 = self.p1[indices]
        p0 = self.p0[indices]
        
        # This is now a full fixed-point evaluation
        y_approx_pre_quant = p3 * (x_quant**3) + p2 * (x_quant**2) + p1 * x_quant + p0
        
        # 3. Quantize output (y)
        y_quant = quantize_torch(y_approx_pre_quant, self.i_bits_y, self.f_bits_y)
        
        return y_quant

# --- Specific Fixed-Point Implementations ---

class PWLSqrtFixed(PWLFunctionFixed):
    def __init__(self, json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y):
        super().__init__(json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y)

class PWLExpFixed(PWLFunctionFixed):
    def __init__(self, json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y):
        super().__init__(json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y)

class PWLGeluFixed(PWLFunctionFixed):
    def __init__(self, json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y):
        super().__init__(json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y)

# --- Fixed-Point Softmax Implementation ---

class PWLSoftmaxFixed(nn.Module):
    """
    A custom softmax module that uses a fixed-point piecewise polynomial 
    approximation of exp.
    """
    def __init__(self, json_path: str, i_bits_x: int, f_bits_x: int, i_bits_y: int, f_bits_y: int, debug: bool = False):
        super().__init__()
        # The input to exp() is the shifted_x (we use X bits)
        # The output of exp() is the exp_x (we use Y bits)
        # The coefficients inside pwl_exp are quantized to Fix16 on init
        self.pwl_exp = PWLExpFixed(json_path, i_bits_x, f_bits_x, i_bits_y, f_bits_y)
        self.debug = debug
        self.debug_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the log-sum-exp trick for numerical stability (in float)
        max_vals = torch.max(x, dim=-1, keepdim=True)[0]
        shifted_x = x - max_vals

        # Approximate e^x using the fixed-point PWL function
        # The "double quantization" (data) and coefficient quantization
        # all happen inside this call.
        exp_x = self.pwl_exp(shifted_x)
        
        # Apply ReLU to ensure non-negativity
        exp_x = torch.relu(exp_x)

        # --- Debugging block (optional) ---
        if self.debug and self.debug_counter < 3:
            torch_exp = torch.exp(shifted_x)
            mae = torch.mean(torch.abs(exp_x - torch_exp)).item()
            print("\n--- FIXED-POINT Softmax Debug ---")
            print(f"Input range (shifted): [{shifted_x.min():.4f}, {shifted_x.max():.4f}]")
            print(f"PWL-Fixed exp() output range:   [{exp_x.min():.4f}, {exp_x.max():.4f}]")
            print(f"Mean Absolute Error vs torch.exp: {mae:2f}")
            print("---------------------------------\n")
            self.debug_counter += 1
        
        # Manually compute the softmax probabilities
        return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)

class DebugSoftmax(nn.Module):
    """
    A dummy softmax module (unchanged from m0).
    """
    def __init__(self):
        super().__init__()
        print("  - [DEBUG] Creating DebugSoftmax module (uses standard torch.exp with stability trick).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_vals = torch.max(x, dim=-1, keepdim=True)[0]
        shifted_x = x - max_vals
        exp_x = torch.exp(shifted_x)
        return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)