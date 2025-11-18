# # m0_udanf.py

# import json
# import torch
# import torch.nn as nn
# from pathlib import Path

# class PWLFunction(nn.Module):
#     """
#     A generic, device-aware PyTorch module for evaluating piecewise polynomial functions.
#     It can handle polynomials up to the 3rd degree (cubic).
#     """
#     def __init__(self, json_path: str):
#         super().__init__()
        
#         if not Path(json_path).is_file():
#             raise FileNotFoundError(f"JSON file not found at: {json_path}")

#         with open(json_path, 'r') as f:
#             pwl_data = json.load(f)
        
#         print(f"  - Creating polynomial approximation module from {Path(json_path).name}")

#         # Convert string intervals to float boundaries, handling 'inf'
#         boundaries = []
#         for interval in pwl_data["intervals"][:-1]: # Exclude the last (+inf) boundary
#             # The right side of the interval is the boundary point
#             boundary_val = float(interval[1])
#             boundaries.append(boundary_val)
        
#         # Extract polynomial coefficients for each segment
#         params = pwl_data["params"]
#         num_segments = len(params)
        
#         # Initialize coefficient tensors with zeros
#         p3 = torch.zeros(num_segments) # ax^3
#         p2 = torch.zeros(num_segments) # bx^2
#         p1 = torch.zeros(num_segments) # cx
#         p0 = torch.zeros(num_segments) # d

#         for i, p in enumerate(params):
#             p3[i] = p.get("p3", 0.0)
#             p2[i] = p.get("p2", 0.0)
#             p1[i] = p.get("p1", 0.0)
#             p0[i] = p.get("p0", 0.0)

#         # Register tensors as buffers to ensure they are moved to the correct device
#         # along with the model (e.g., when calling model.to('cuda'))
#         self.register_buffer('boundaries', torch.tensor(boundaries, dtype=torch.float32))
#         self.register_buffer('p3', p3.float())
#         self.register_buffer('p2', p2.float())
#         self.register_buffer('p1', p1.float())
#         self.register_buffer('p0', p0.float())

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Determine which segment each element of the input tensor falls into
#         # bucketize finds the index of the boundary just to the right of each input value
#         indices = torch.bucketize(x, self.boundaries)

#         # Gather the correct polynomial coefficients for each element
#         p3 = self.p3[indices]
#         p2 = self.p2[indices]
#         p1 = self.p1[indices]
#         p0 = self.p0[indices]
        
#         # Evaluate the polynomial: p3*x^3 + p2*x^2 + p1*x + p0
#         # This handles linear, quadratic, and cubic cases automatically, as unused
#         # coefficients (e.g., p3 for a quadratic) will be zero.
#         y = p3 * (x**3) + p2 * (x**2) + p1 * x + p0
        
#         return y

# # Specific implementations for each function type
# class PWLSqrt(PWLFunction):
#     def __init__(self, json_path):
#         super().__init__(json_path)

# class PWLExp(PWLFunction):
#     def __init__(self, json_path):
#         super().__init__(json_path)

# class PWLGelu(PWLFunction):
#     def __init__(self, json_path):
#         super().__init__(json_path)

# # --- Softmax Implementations ---

# class PWLSoftmax(nn.Module):
#     """
#     A custom softmax module that uses a piecewise polynomial approximation of exp.
#     Includes a ReLU to ensure the output is non-negative.
#     """
#     def __init__(self, json_path: str, debug: bool = False):
#         super().__init__()
#         self.pwl_exp = PWLExp(json_path)
#         self.debug = debug
#         self.debug_counter = 0

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Approximate e^x using the piecewise polynomial function
#         exp_x = self.pwl_exp(x)
        
#         # Apply ReLU to ensure non-negativity, which is a key property of exp(x)
#         exp_x = torch.relu(exp_x)

#         # --- Debugging block ---
#         if self.debug and self.debug_counter < 3: # Print for the first 3 batches
#             torch_exp = torch.exp(x)
#             mae = torch.mean(torch.abs(exp_x - torch_exp)).item()
#             print("\n--- Softmax Debug ---")
#             print(f"Input range: [{x.min():.4f}, {x.max():.4f}] | Input mean: {x.mean():.4f}")
#             print(f"PWL exp() output range:   [{exp_x.min():.4f}, {exp_x.max():.4f}]")
#             print(f"Torch exp() output range: [{torch_exp.min():.4f}, {torch_exp.max():.4f}]")
#             print(f"Mean Absolute Error between exp versions: {mae:2f}")
#             print("---------------------\n")
#             self.debug_counter += 1
        
#         # Manually compute the softmax probabilities
#         # Add a small epsilon for numerical stability, preventing division by zero
#         return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)

# class DebugSoftmax(nn.Module):
#     """
#     A dummy softmax module that uses the standard torch.exp but follows the same
#     replacement structure. Used to verify that the model modification logic works.
#     """
#     def __init__(self):
#         super().__init__()
#         print("  - [DEBUG] Creating DebugSoftmax module (uses standard torch.exp).")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Standard softmax implementation
#         exp_x = torch.exp(x)
#         return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-6)


# m0_udanf.py

import json
import torch
import torch.nn as nn
from pathlib import Path

class PWLFunction(nn.Module):
    """
    A generic, device-aware PyTorch module for evaluating piecewise polynomial functions.
    It can handle polynomials up to the 3rd degree (cubic).
    """
    def __init__(self, json_path: str):
        super().__init__()
        
        if not Path(json_path).is_file():
            raise FileNotFoundError(f"JSON file not found at: {json_path}")

        with open(json_path, 'r') as f:
            pwl_data = json.load(f)
        
        print(f"  - Creating polynomial approximation module from {Path(json_path).name}")

        # Convert string intervals to float boundaries, handling 'inf'
        boundaries = []
        for interval in pwl_data["intervals"][:-1]: # Exclude the last (+inf) boundary
            # The right side of the interval is the boundary point
            boundary_val = float(interval[1])
            boundaries.append(boundary_val)
        
        # Extract polynomial coefficients for each segment
        params = pwl_data["params"]
        num_segments = len(params)
        
        # Initialize coefficient tensors with zeros
        p3 = torch.zeros(num_segments) # for x^3
        p2 = torch.zeros(num_segments) # for x^2
        p1 = torch.zeros(num_segments) # for x
        p0 = torch.zeros(num_segments) # for constant

        for i, p in enumerate(params):
            p3[i] = p.get("p3", 0.0)
            p2[i] = p.get("p2", 0.0)
            p1[i] = p.get("p1", 0.0)
            p0[i] = p.get("p0", 0.0)

        # Register tensors as buffers to ensure they are moved to the correct device
        self.register_buffer('boundaries', torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer('p3', p3.float())
        self.register_buffer('p2', p2.float())
        self.register_buffer('p1', p1.float())
        self.register_buffer('p0', p0.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine which segment each element of the input tensor falls into
        indices = torch.bucketize(x, self.boundaries)

        # Gather the correct polynomial coefficients for each element
        p3 = self.p3[indices]
        p2 = self.p2[indices]
        p1 = self.p1[indices]
        p0 = self.p0[indices]
        
        # Evaluate the polynomial: p3*x^3 + p2*x^2 + p1*x + p0
        y = p3 * (x**3) + p2 * (x**2) + p1 * x + p0
        
        return y

# Specific implementations for each function type
class PWLSqrt(PWLFunction):
    def __init__(self, json_path):
        super().__init__(json_path)

class PWLExp(PWLFunction):
    def __init__(self, json_path):
        super().__init__(json_path)

class PWLGelu(PWLFunction):
    def __init__(self, json_path):
        super().__init__(json_path)

# --- Softmax Implementations ---

class PWLSoftmax(nn.Module):
    """
    A custom softmax module that uses a piecewise polynomial approximation of exp.
    Includes a ReLU and the log-sum-exp trick for stability.
    """
    def __init__(self, json_path: str, debug: bool = False):
        super().__init__()
        self.pwl_exp = PWLExp(json_path)
        self.debug = debug
        self.debug_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- NEW: Apply the log-sum-exp trick for numerical stability ---
        max_vals = torch.max(x, dim=-1, keepdim=True)[0]
        shifted_x = x - max_vals

        # Approximate e^x using the piecewise polynomial function on the shifted inputs
        exp_x = self.pwl_exp(shifted_x)
        
        # Apply ReLU to ensure non-negativity
        exp_x = torch.relu(exp_x)

        # --- Debugging block ---
        if self.debug and self.debug_counter < 3: # Print for the first 3 batches
            torch_exp = torch.exp(shifted_x)
            mae = torch.mean(torch.abs(exp_x - torch_exp)).item()
            print("\n--- Softmax Debug ---")
            print(f"Input range (shifted): [{shifted_x.min():.4f}, {shifted_x.max():.4f}] | Input mean: {shifted_x.mean():.4f}")
            print(f"PWL exp() output range:   [{exp_x.min():.4f}, {exp_x.max():.4f}]")
            print(f"Torch exp() output range: [{torch_exp.min():.4f}, {torch_exp.max():.4f}]")
            print(f"Mean Absolute Error between exp versions: {mae:2f}")
            print("---------------------\n")
            self.debug_counter += 1
        
        # Manually compute the softmax probabilities
        return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)

class DebugSoftmax(nn.Module):
    """
    A dummy softmax module that uses the standard torch.exp but with the
    stabilization trick. Used to verify that the model modification logic works.
    """
    def __init__(self):
        super().__init__()
        print("  - [DEBUG] Creating DebugSoftmax module (uses standard torch.exp with stability trick).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard softmax implementation with stability trick
        max_vals = torch.max(x, dim=-1, keepdim=True)[0]
        shifted_x = x - max_vals
        exp_x = torch.exp(shifted_x)
        return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)


