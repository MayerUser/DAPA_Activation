# t2_make_poly.py

import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# Suppress RankWarning from polyfit
# warnings.filterwarnings("ignore", category=np.RankWarning)


# --- GELU Function Definition ---
def gelu_exact(x):
    """
    Exact GELU activation function using the error function (erf), which is
    what PyTorch and TensorFlow use. More stable than the tanh approximation.
    """
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

# --- Plotting Function ---
def plot_poly_vs_original(poly_coeffs, order, save_path: Path):
    """
    Plots the polynomial approximation against the original GELU function,
    including the linear tails for x < -4 and x > 4.
    """
    plt.figure(figsize=(12, 7))

    # 1. Plot the original GELU function over a wide range
    x_orig = np.linspace(-6, 6, 400)
    y_orig = gelu_exact(x_orig)
    plt.plot(x_orig, y_orig, label='Original GELU Function', color='blue', linewidth=3, zorder=5)

    # 2. Plot the polynomial approximation in the core range [-4, 4]
    poly_func = np.poly1d(poly_coeffs)
    x_poly = np.linspace(-4, 4, 200)
    y_poly = poly_func(x_poly)
    plt.plot(x_poly, y_poly, label=f'Order-{order} Polynomial Approx.', color='red', linestyle='--', linewidth=2, zorder=10)

    # 3. Plot the lower tail (y=0 for x < -4)
    x_lower = np.linspace(-6, -4, 50)
    y_lower = np.zeros_like(x_lower)
    plt.plot(x_lower, y_lower, color='red', linestyle='--', linewidth=2, zorder=10)

    # 4. Plot the upper tail (y=x for x > 4)
    x_upper = np.linspace(4, 6, 50)
    y_upper = x_upper
    plt.plot(x_upper, y_upper, color='red', linestyle='--', linewidth=2, zorder=10)
    
    # Add vertical lines to show the boundaries
    plt.axvline(x=-4, color='gray', linestyle=':', linewidth=1, zorder=1)
    plt.axvline(x=4, color='gray', linestyle=':', linewidth=1, zorder=1)

    plt.title(f'GELU vs. Polynomial Approximation (Order {order})')
    plt.xlabel("Input Value (x)")
    plt.ylabel("Output Value (y)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 6)
    plt.xlim(-6, 6)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison plot to {save_path}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate a polynomial approximation for the GELU function.")
    parser.add_argument("--order", type=int, required=True,
                        help="The order (N) of the polynomial to generate.")
    args = parser.parse_args()

    order = args.order
    
    # Define the core approximation range
    x_min, x_max = -4, 4
    num_points = 500 # Number of points to sample for the fit

    # 1. Generate sample data from the GELU function
    x_samples = np.linspace(x_min, x_max, num_points)
    y_samples = gelu_exact(x_samples)

    # 2. Perform the polynomial fit to get the coefficients.
    #    np.polyfit finds the coefficients that minimize the squared error (MSE)
    #    between the polynomial and the y_samples, without using any distribution.
    print(f"Fitting a polynomial of order {order} to GELU in range [{x_min}, {x_max}] using direct MSE minimization...")
    coeffs = np.polyfit(x_samples, y_samples, order)
    
    # 3. Structure the output data for the JSON file
    output_data = {
        "function": "gelu_poly_approx",
        "order": order,
        "poly_range": [x_min, x_max],
        "coefficients": {f"p{order-i}": c for i, c in enumerate(coeffs)},
        "lower_tail": {"type": "constant", "value": 0.0},
        "upper_tail": {"type": "linear", "slope": 1.0, "intercept": 0.0}
    }
    
    # 4. Create directories and save the output files
    poly_dir = Path("dst_poly")
    plot_dir = Path("dst_plot")
    poly_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    
    file_path = poly_dir / f"poly_gelu_{order}_order.json"
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved polynomial function to {file_path}")

    # 5. Plot the result for visual verification
    plot_path = plot_dir / f"plot_poly_gelu_{order}_order.png"
    plot_poly_vs_original(coeffs, order, plot_path)


if __name__ == "__main__":
    main()

