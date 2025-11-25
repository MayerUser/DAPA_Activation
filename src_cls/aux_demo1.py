# aux_demo1.py
#
# Compute MSE (uniform over x in [-4, 4]) and DWMSE (distribution-weighted MSE
# also restricted to x in [-4, 4]) for:
#   - PWL DAPA GELU (vit-tiny, segments 4/6/8/10/12/14/16)
#   - Polynomial GELU (orders 4/5/6/7/8)
# and read corresponding Top-1 accuracy from dst_log.
#
# Also compute DeltaTop1 = Top1 - BASELINE_TOP1
# and Pearson correlation (r, p-value) between:
#   - DeltaTop1 vs MSE
#   - DeltaTop1 vs DWMSE (both over [-4, 4])

import json
import re
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# Config (DEMO1)
# ---------------------------------------------------------------------
DEMO1_MODELS = ["vit-tiny"]
DEMO1_SEGMENTS = [4, 6, 8, 10, 12, 14, 16]
DEMO1_SAMPLES = 256
DEMO1_POLY_ORDER = [4, 5, 6, 7, 8]

# Range used for both MSE and DWMSE
X_MIN, X_MAX = -4.0, 4.0

# vit-tiny FP32 baseline Top-1 accuracy (percent)
BASELINE_TOP1 = 75.51


# ---------------------------------------------------------------------
# GELU (exact) and helpers
# ---------------------------------------------------------------------
def gelu_exact(x: np.ndarray) -> np.ndarray:
    """
    Exact GELU as used in t2_make_poly.py:
        GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def load_gelu_pdf(model_name: str, num_samples: int):
    """
    Load GELU input PDF for a given model and sample count.
    Expects file:
        dst_pdf/{num_samples}/pdf_gelu_act_{model_name}_{num_samples}samples.npz
    """
    pdf_path = Path(f"dst_pdf/{num_samples}/pdf_gelu_act_{model_name}_{num_samples}samples.npz")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    data = np.load(pdf_path)
    hist = data["hist"]
    bin_edges = data["bin_edges"]
    return hist, bin_edges


def select_range(hist: np.ndarray, bin_edges: np.ndarray, x_min: float, x_max: float):
    """
    Restrict histogram and bin centers to [x_min, x_max].
    Returns: x_centers_selected, hist_selected, bin_width, (effective_min, effective_max)
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    mask = (bin_centers >= x_min) & (bin_centers <= x_max)
    if not np.any(mask):
        raise RuntimeError(f"No histogram bins found in range [{x_min}, {x_max}].")

    x_sel = bin_centers[mask]
    h_sel = hist[mask]
    eff_min = x_sel[0]
    eff_max = x_sel[-1]
    return x_sel, h_sel, bin_width, (eff_min, eff_max)


def compute_mse_dwmse(approx_func, hist: np.ndarray, bin_edges: np.ndarray,
                      x_min: float = X_MIN, x_max: float = X_MAX):
    """
    Compute:
      - MSE:   E_{x ~ Uniform[x_min,x_max]}[(GELU(x) - f(x))^2],
               approximated over bin centers in [x_min, x_max].
      - DWMSE: E_{x ~ p(x) | x in [x_min,x_max]}[(GELU(x) - f(x))^2],
               i.e., distribution-weighted MSE restricted to [x_min, x_max].

    Both metrics are now explicitly computed only over the range [x_min, x_max].
    """
    x, pdf_segment, bin_width, (eff_min, eff_max) = select_range(hist, bin_edges, x_min, x_max)

    true_vals = gelu_exact(x)
    approx_vals = approx_func(x)
    err_sq = (true_vals - approx_vals) ** 2

    # MSE over uniform distribution on [x_min, x_max]
    mse = np.sum(err_sq * bin_width) / (x_max - x_min)

    # DWMSE over the empirical pdf restricted to [x_min, x_max],
    # normalized so it is an expectation under p(x | x in [x_min,x_max]).
    mass = np.sum(pdf_segment * bin_width)
    if mass > 0:
        dwmse = np.sum(pdf_segment * err_sq * bin_width) / mass
    else:
        dwmse = float("nan")

    return mse, dwmse


# ---------------------------------------------------------------------
# Build approximation functions
# ---------------------------------------------------------------------
def load_pwl_func(model_name: str, segments: int, num_samples: int):
    """
    Load PWL GELU approximation for given (model, segments, samples).

    Expects JSON:
      dst_pwl/{num_samples}/pwl_gelu_act_{model_name}_{segments}seg.json
    """
    pwl_path = Path(f"dst_pwl/{num_samples}/pwl_gelu_act_{model_name}_{segments}seg.json")
    if not pwl_path.exists():
        raise FileNotFoundError(f"PWL file not found: {pwl_path}")

    with open(pwl_path, "r") as f:
        data = json.load(f)

    intervals = data["intervals"]
    params_list = data["params"]

    segments_data = []
    for (start_str, end_str), params in zip(intervals, params_list):
        if start_str == "-inf":
            s = float("-inf")
        else:
            s = float(start_str)

        if end_str == "+inf":
            e = float("inf")
        else:
            e = float(end_str)

        p0 = params.get("p0", 0.0)
        p1 = params.get("p1", 0.0)
        p2 = params.get("p2", 0.0)
        p3 = params.get("p3", 0.0)
        segments_data.append((s, e, (p0, p1, p2, p3)))

    def pwl_func(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.zeros_like(x, dtype=float)
        for s, e, (p0, p1, p2, p3) in segments_data:
            mask = (x >= s) & (x <= e)
            if not np.any(mask):
                continue
            xm = x[mask]
            y[mask] = ((p3 * xm + p2) * xm + p1) * xm + p0
        return y

    return pwl_func


def load_poly_func(order: int):
    """
    Load polynomial GELU approximation for given order.

    Expects JSON:
      dst_poly/poly_gelu_{order}_order.json
    """
    poly_path = Path(f"dst_poly/poly_gelu_{order}_order.json")
    if not poly_path.exists():
        raise FileNotFoundError(f"Poly file not found: {poly_path}")

    with open(poly_path, "r") as f:
        data = json.load(f)

    N = data["order"]
    coeffs_json = data["coefficients"]

    # Build coefficients in descending order for np.poly1d: [cN, cN-1, ..., c0]
    coeffs_desc = [coeffs_json.get(f"p{k}", 0.0) for k in range(N, -1, -1)]
    poly = np.poly1d(coeffs_desc)

    def poly_func(x: np.ndarray) -> np.ndarray:
        return poly(x)

    return poly_func


# ---------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------
# Example line:
#   "Top-1 Accuracy on 512 samples (FP32): 79.88%"
TOP1_PATTERN = re.compile(
    r"Top-1\s+Accuracy[^\n:]*:\s*([0-9.]+)\s*(%)?",
    flags=re.IGNORECASE,
)


def parse_top1_from_log(log_path: Path):
    """
    Extract Top-1 Accuracy from a log file.

    Expected patterns (examples):
      "Top-1 Accuracy on 512 samples (FP32): 79.88%"
      "Top-1 Accuracy: 81.40%"
      "Top-1 Accuracy: 0.8140"
    Returns accuracy in percent.
    """
    if not log_path.exists():
        print(f"[WARN] Log file not found: {log_path}")
        return None

    text = log_path.read_text()

    m = TOP1_PATTERN.search(text)
    if m:
        val = float(m.group(1))
        has_percent = m.group(2) is not None
        if not has_percent and val <= 1.5:
            # Interpret as fraction (0.814 -> 81.4%)
            val *= 100.0
        return val

    # Fallback: generic "Accuracy:" pattern
    m2 = re.search(r"Accuracy[^\n:]*:\s*([0-9.]+)\s*(%)?", text, flags=re.IGNORECASE)
    if m2:
        val = float(m2.group(1))
        has_percent = m2.group(2) is not None
        if not has_percent and val <= 1.5:
            val *= 100.0
        return val

    print(f"[WARN] Could not find Top-1 Accuracy in log: {log_path}")
    return None


# ---------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------
def compute_corr(entries, label):
    """
    Compute Pearson correlation between:
      DeltaTop1 vs MSE
      DeltaTop1 vs DWMSE
    for a list of entries that each have:
      "MSE", "DWMSE", "Top1", "DeltaTop1"
    Only prints a single block for 'label'.
    """
    mse_list = []
    dwmse_list = []
    delta_list = []

    for e in entries:
        top1 = e.get("Top1", None)
        delta = e.get("DeltaTop1", None)
        if top1 is None or delta is None:
            continue
        mse_list.append(e["MSE"])
        dwmse_list.append(e["DWMSE"])
        delta_list.append(delta)

    if len(delta_list) < 2:
        print(f"[WARN] Not enough points to compute correlation for {label}.")
        return None

    mse_arr = np.array(mse_list)
    dwmse_arr = np.array(dwmse_list)
    delta_arr = np.array(delta_list)

    r_mse, p_mse = pearsonr(mse_arr, delta_arr)
    r_dwmse, p_dwmse = pearsonr(dwmse_arr, delta_arr)

    print(f"\nCorrelation ({label}):")
    print(f"  DeltaTop1 vs MSE:   r = {r_mse:+.4f}, p = {p_mse:.4e}")
    print(f"  DeltaTop1 vs DWMSE: r = {r_dwmse:+.4f}, p = {p_dwmse:.4e}")

    return {
        "label": label,
        "n": len(delta_list),
        "r_MSE": float(r_mse),
        "p_MSE": float(p_mse),
        "r_DWMSE": float(r_dwmse),
        "p_DWMSE": float(p_dwmse),
    }


# ---------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------
def main():
    results = {
        "config": {
            "DEMO1_MODELS": DEMO1_MODELS,
            "DEMO1_SEGMENTS": DEMO1_SEGMENTS,
            "DEMO1_SAMPLES": DEMO1_SAMPLES,
            "DEMO1_POLY_ORDER": DEMO1_POLY_ORDER,
            "X_MIN": X_MIN,
            "X_MAX": X_MAX,
            "BASELINE_TOP1": BASELINE_TOP1,
        },
        "pwl": [],
        "poly": [],
        "correlation": {},
    }

    out_dir = Path("dst_aux")
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "demo1_metrics.json"

    for model in DEMO1_MODELS:
        print(f"\n=== DEMO1 Metrics for model: {model}, samples={DEMO1_SAMPLES} ===")

        # Load shared GELU PDF for this model
        hist, bin_edges = load_gelu_pdf(model, DEMO1_SAMPLES)

        # -------------------------
        # PWL (DAPA) cases
        # -------------------------
        print("\n--- PWL (DAPA) cases ---")
        for seg in DEMO1_SEGMENTS:
            try:
                approx_func = load_pwl_func(model, seg, DEMO1_SAMPLES)
            except FileNotFoundError as e:
                print(f"[SKIP PWL seg={seg}] {e}")
                continue

            mse, dwmse = compute_mse_dwmse(approx_func, hist, bin_edges)
            log_path = Path(f"dst_log/test_dm1_pwl_{model}_{seg}_{DEMO1_SAMPLES}.log")
            top1 = parse_top1_from_log(log_path)
            delta = top1 - BASELINE_TOP1 if top1 is not None else None

            entry = {
                "model": model,
                "segments": seg,
                "samples": DEMO1_SAMPLES,
                "MSE": mse,
                "DWMSE": dwmse,
                "Top1": top1,
                "DeltaTop1": delta,
                "log": str(log_path),
            }
            results["pwl"].append(entry)

            top1_str = f"{top1:.2f}%" if top1 is not None else "N/A"
            delta_str = f"{delta:+.2f}pt" if delta is not None else "N/A"
            print(
                f"PWL seg={seg:2d}:  MSE={mse:.6e},  DWMSE={dwmse:.6e},  "
                f"Top-1={top1_str},  ΔAcc={delta_str}"
            )

        # -------------------------
        # Polynomial cases
        # -------------------------
        print("\n--- Polynomial cases ---")
        for order in DEMO1_POLY_ORDER:
            try:
                approx_func = load_poly_func(order)
            except FileNotFoundError as e:
                print(f"[SKIP Poly order={order}] {e}")
                continue

            mse, dwmse = compute_mse_dwmse(approx_func, hist, bin_edges)
            log_path = Path(f"dst_log/test_dm1_poly_{model}_{order}_{DEMO1_SAMPLES}.log")
            top1 = parse_top1_from_log(log_path)
            delta = top1 - BASELINE_TOP1 if top1 is not None else None

            entry = {
                "model": model,
                "order": order,
                "samples": DEMO1_SAMPLES,
                "MSE": mse,
                "DWMSE": dwmse,
                "Top1": top1,
                "DeltaTop1": delta,
                "log": str(log_path),
            }
            results["poly"].append(entry)

            top1_str = f"{top1:.2f}%" if top1 is not None else "N/A"
            delta_str = f"{delta:+.2f}pt" if delta is not None else "N/A"
            print(
                f"Poly N={order:2d}:  MSE={mse:.6e},  DWMSE={dwmse:.6e},  "
                f"Top-1={top1_str},  ΔAcc={delta_str}"
            )

    # -----------------------------------------------------------------
    # Correlations + p-values: ONLY combined (PWL + Poly)
    # -----------------------------------------------------------------
    corr_all = compute_corr(results["pwl"] + results["poly"], label="All (PWL + Poly)")

    if corr_all is not None:
        results["correlation"]["all"] = corr_all

    # Save all metrics to JSON for later plotting/LaTeX tables
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved DEMO1 metrics to {out_json}")


if __name__ == "__main__":
    main()
