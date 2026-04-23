# m2_dwmse_cal.py
# Compute MSE and DWMSE over [-4,4] for PWL and Poly GELU approximations,
# AND parse loss/perplexity from log files into the same CSV.

import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.special import erf as sp_erf


# ----------------------------
# True GELU (exact, erf-based)
# ----------------------------
def gelu_exact(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + sp_erf(x / np.sqrt(2.0)))


# --------------------------------
# Approximator loaders/evaluators
# --------------------------------
class PWLApprox:
    """Evaluate piecewise polynomial y = p3 x^3 + p2 x^2 + p1 x + p0 from PWL JSON."""
    def __init__(self, json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)

        boundaries = []
        for inter in data["intervals"][:-1]:
            r = inter[1]
            if isinstance(r, str) and ("inf" in r.lower()):
                val = np.inf
            else:
                val = float(r)
            boundaries.append(val)
        self.boundaries = np.asarray(boundaries, dtype=np.float64)

        params = data["params"]
        self.p3 = np.asarray([p.get("p3", 0.0) for p in params], dtype=np.float64)
        self.p2 = np.asarray([p.get("p2", 0.0) for p in params], dtype=np.float64)
        self.p1 = np.asarray([p.get("p1", 0.0) for p in params], dtype=np.float64)
        self.p0 = np.asarray([p.get("p0", 0.0) for p in params], dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.boundaries, x, side="left")
        return (
            self.p3[idx] * (x ** 3) +
            self.p2[idx] * (x ** 2) +
            self.p1[idx] * x +
            self.p0[idx]
        )


class PolyGeluApprox:
    """Evaluate polynomial in [-4,4] + tails defined in poly JSON."""
    def __init__(self, json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.xmin, self.xmax = map(float, data["poly_range"])
        order = int(data["order"])
        coeffs_dict = data["coefficients"]
        coeffs = [coeffs_dict.get(f"p{d}", 0.0) for d in range(order, -1, -1)]
        self.poly = np.poly1d(coeffs)

        self.lower_val = float(data["lower_tail"]["value"])
        self.upper_slope = float(data["upper_tail"]["slope"])
        self.upper_intercept = float(data["upper_tail"]["intercept"])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float64)
        lower = x < self.xmin
        upper = x > self.xmax
        mid   = ~(lower | upper)
        y[lower] = self.lower_val
        y[upper] = self.upper_slope * x[upper] + self.upper_intercept
        if np.any(mid):
            y[mid] = self.poly(x[mid])
        return y


# ----------------------------
# PDF loaders (JSON-first)
# ----------------------------
def _interp_pdf_from_hist(bin_edges: np.ndarray, hist: np.ndarray):
    """Return an interpolated pdf(x) from histogram (density=True)."""
    bin_edges = bin_edges.astype(np.float64)
    hist = np.maximum(hist.astype(np.float64), 0.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    def pdf(x: np.ndarray) -> np.ndarray:
        return np.interp(x, centers, hist, left=0.0, right=0.0)
    return pdf, float(bin_edges[0]), float(bin_edges[-1])


def load_pdf_from_json(json_path: Path):
    """
    Supports flexible JSON schemas:
      - {"hist":[...], "bin_edges":[...]}
      - {"bin_centers":[...], "density":[...]}
      - {"pdf":{"hist":[...], "bin_edges":[...]}} or {"pdf":{"x":[...], "y":[...]}}
    Returns (pdf_func, a, b).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "hist" in data and "bin_edges" in data:
        be = np.asarray(data["bin_edges"], dtype=np.float64)
        h  = np.asarray(data["hist"], dtype=np.float64)
        return _interp_pdf_from_hist(be, h)

    if "bin_centers" in data and "density" in data:
        centers = np.asarray(data["bin_centers"], dtype=np.float64)
        dens    = np.maximum(np.asarray(data["density"], dtype=np.float64), 0.0)
        a, b = float(centers.min()), float(centers.max())
        def pdf(x: np.ndarray) -> np.ndarray:
            return np.interp(x, centers, dens, left=0.0, right=0.0)
        return pdf, a, b

    if "pdf" in data and isinstance(data["pdf"], dict):
        pdfd = data["pdf"]
        if "hist" in pdfd and "bin_edges" in pdfd:
            be = np.asarray(pdfd["bin_edges"], dtype=np.float64)
            h  = np.asarray(pdfd["hist"], dtype=np.float64)
            return _interp_pdf_from_hist(be, h)
        if "x" in pdfd and "y" in pdfd:
            xx = np.asarray(pdfd["x"], dtype=np.float64)
            yy = np.maximum(np.asarray(pdfd["y"], dtype=np.float64), 0.0)
            a, b = float(xx.min()), float(xx.max())
            def pdf(x: np.ndarray) -> np.ndarray:
                return np.interp(x, xx, yy, left=0.0, right=0.0)
            return pdf, a, b

    raise ValueError(f"Unrecognized PDF JSON schema: {json_path}")


def std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def find_pdf(args):
    """
    Try JSON in dst_pdf first (model/num_samples), else fallback to standard normal.
    Returns (pdf_func, (a,b), desc).
    """
    if args.pdf_json:
        p = Path(args.pdf_json)
        if p.is_file():
            f, a, b = load_pdf_from_json(p)
            return f, (a, b), f"json:{p}"

    if args.num_samples is not None and args.model is not None:
        cand = Path(f"dst_pdf/{args.num_samples}/pdf_gelu_act_{args.model}.json")
        if cand.is_file():
            f, a, b = load_pdf_from_json(cand)
            return f, (a, b), f"json:{cand}"

    for p in sorted(Path("dst_pdf").glob("**/pdf_gelu_act_*.json")):
        try:
            f, a, b = load_pdf_from_json(p)
            return f, (a, b), f"json:{p}"
        except Exception:
            continue

    return std_normal_pdf, (-np.inf, np.inf), "std_normal"


# ----------------------------
# Metric calculators
# ----------------------------
def compute_mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    e = y_hat - y_true
    return float(np.mean(e * e))


def compute_dwmse(x: np.ndarray, y_true: np.ndarray, y_hat: np.ndarray, pdf_func) -> float:
    e2 = (y_hat - y_true) ** 2
    w = pdf_func(x).astype(np.float64)
    dx = (x[-1] - x[0]) / (len(x) - 1)
    Z = np.sum(w) * dx
    if not np.isfinite(Z) or Z <= 0:
        w = np.ones_like(w)
        Z = np.sum(w) * dx
    return float(np.sum(w * e2) * dx / Z)


# ----------------------------
# Discovery of approximation files
# ----------------------------
def discover_poly(poly_dir: Path) -> list[Path]:
    return sorted(poly_dir.glob("poly_gelu_*_order.json"))


def discover_pwl(pwl_root: Path, model: str | None) -> list[Path]:
    files = []
    for sub in sorted(pwl_root.glob("*")):
        if not sub.is_dir():
            continue
        pattern = "pwl_gelu_act_*.json" if model is None else f"pwl_gelu_act_{model}_*seg.json"
        files.extend(sorted(sub.glob(pattern)))
    return files


def parse_poly_order(p: Path) -> str:
    try:
        return p.stem.split("_")[2]  # poly_gelu_<N>_order
    except Exception:
        return "?"


def parse_pwl_segs(p: Path) -> str:
    try:
        return p.stem.split("_")[-1].replace("seg", "")  # ..._<S>seg
    except Exception:
        return "?"


# ----------------------------
# Log parsing
# ----------------------------
LOG_PATTERNS = [
    # GPT-2 simple eval block:
    re.compile(
        r"mean_loss\s*=\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)\s*.*?"
        r"perplexity\s*=\s*(?P<ppl>[0-9]+(?:\.[0-9]+)?)"
        r"(?:.*?softmax=(?P<softmax>\S+))?"
        r"(?:.*?act=(?P<act>\S+))?"
        r"(?:.*?precision=(?P<precision>\S+))?"
        r"(?:.*?block_size=(?P<block_size>\d+))?",
        re.IGNORECASE | re.DOTALL
    ),
    # A/B style lines:
    re.compile(
        r"baseline\s*:\s*loss\s*=\s*(?P<base_loss>[0-9]+(?:\.[0-9]+)?)\s*\|\s*ppl\s*=\s*(?P<base_ppl>[0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE
    ),
    re.compile(
        r"patched\s*:\s*loss\s*=\s*(?P<patch_loss>[0-9]+(?:\.[0-9]+)?)\s*\|\s*ppl\s*=\s*(?P<patch_ppl>[0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE
    ),
    # Simpler single numbers
    re.compile(r"perplexity\s*=\s*(?P<ppl>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"mean_loss\s*=\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]

def parse_logs(log_dirs: list[Path]) -> list[dict]:
    entries = []
    for d in log_dirs:
        for log in sorted(d.glob("**/*.log")):
            try:
                text = log.read_text(errors="ignore")
            except Exception:
                continue

            row = {
                "kind": "log",
                "name": log.name,
                "order_or_seg": "",
                "mse": "",
                "dwmse": "",
                "source": str(log),
                "loss": "",
                "ppl": "",
                "baseline_loss": "",
                "baseline_ppl": "",
                "patched_loss": "",
                "patched_ppl": "",
                "softmax": "",
                "act": "",
                "precision": "",
                "block_size": "",
            }

            # Apply patterns
            for pat in LOG_PATTERNS:
                for m in pat.finditer(text):
                    md = m.groupdict()

                    # Detailed GPT-2 block (loss, ppl, plus knobs)
                    if "loss" in md and md.get("loss"):
                        row["loss"] = md["loss"]
                    if "ppl" in md and md.get("ppl"):
                        row["ppl"] = md["ppl"]

                    if "base_loss" in md and md.get("base_loss"):
                        row["baseline_loss"] = md["base_loss"]
                    if "base_ppl" in md and md.get("base_ppl"):
                        row["baseline_ppl"] = md["base_ppl"]

                    if "patch_loss" in md and md.get("patch_loss"):
                        row["patched_loss"] = md["patch_loss"]
                    if "patch_ppl" in md and md.get("patch_ppl"):
                        row["patched_ppl"] = md["patch_ppl"]

                    if "softmax" in md and md.get("softmax"):
                        row["softmax"] = md["softmax"]
                    if "act" in md and md.get("act"):
                        row["act"] = md["act"]
                    if "precision" in md and md.get("precision"):
                        row["precision"] = md["precision"]
                    if "block_size" in md and md.get("block_size"):
                        row["block_size"] = md["block_size"]

            # Only append if at least some metric found
            if any(row[k] for k in ["loss","ppl","baseline_loss","baseline_ppl","patched_loss","patched_ppl"]):
                entries.append(row)
    return entries


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute MSE/DWMSE for PWL & Poly GELU on [-4,4] and parse loss/PPL logs.")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--grid_points", type=int, default=2001)
    ap.add_argument("--poly_dir", type=str, default="dst_poly")
    ap.add_argument("--pwl_dir", type=str, default="dst_pwl")
    ap.add_argument("--out_csv", type=str, default="dst_metrics/dwmse_results.csv")
    # PDF
    ap.add_argument("--pdf_json", type=str, default=None, help="Explicit JSON path for histogram PDF.")
    # Logs
    ap.add_argument("--log_dirs", type=str, nargs="*", default=["dst_log_test_gpt2", "dst_log"],
                    help="Directories to scan for .log files.")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # grid & truth
    x = np.linspace(-4.0, 4.0, args.grid_points, dtype=np.float64)
    y_true = gelu_exact(x)

    # PDF
    pdf_func, (a, b), desc = find_pdf(args)
    print(f"[info] Using PDF = {desc} (coverage ~[{a:.3g}, {b:.3g}])")

    # Discover inputs
    poly_files = discover_poly(Path(args.poly_dir))
    pwl_files  = discover_pwl(Path(args.pwl_dir), args.model)

    rows = []

    # POLY
    for pf in poly_files:
        try:
            apx = PolyGeluApprox(pf)
            y_hat = apx(x)
            mse = compute_mse(y_true, y_hat)
            dwmse = compute_dwmse(x, y_true, y_hat, pdf_func)
            rows.append({
                "kind": "poly",
                "name": pf.name,
                "order_or_seg": parse_poly_order(pf),
                "mse": f"{mse:.8e}",
                "dwmse": f"{dwmse:.8e}",
                "source": str(pf),
                "loss": "", "ppl": "",
                "baseline_loss": "", "baseline_ppl": "",
                "patched_loss": "", "patched_ppl": "",
                "softmax": "", "act": "", "precision": "", "block_size": "",
            })
        except Exception as e:
            print(f"[error] POLY {pf.name}: {e}")

    # PWL
    for pf in pwl_files:
        try:
            apx = PWLApprox(pf)
            y_hat = apx(x)
            mse = compute_mse(y_true, y_hat)
            dwmse = compute_dwmse(x, y_true, y_hat, pdf_func)
            rows.append({
                "kind": "pwl",
                "name": pf.name,
                "order_or_seg": parse_pwl_segs(pf),
                "mse": f"{mse:.8e}",
                "dwmse": f"{dwmse:.8e}",
                "source": str(pf),
                "loss": "", "ppl": "",
                "baseline_loss": "", "baseline_ppl": "",
                "patched_loss": "", "patched_ppl": "",
                "softmax": "", "act": "", "precision": "", "block_size": "",
            })
        except Exception as e:
            print(f"[error] PWL {pf.name}: {e}")

    # Logs
    log_entries = parse_logs([Path(p) for p in args.log_dirs])
    for le in log_entries:
        rows.append(le)

    # sort: approximations first (by numeric order), then logs by name
    def numkey(r):
        try:
            return int(r["order_or_seg"])
        except Exception:
            return 10**9

    approx_rows = [r for r in rows if r["kind"] in ("poly","pwl")]
    log_rows    = [r for r in rows if r["kind"] == "log"]

    approx_rows.sort(key=lambda r: (r["kind"], numkey(r)))
    log_rows.sort(key=lambda r: r["name"])
    rows = approx_rows + log_rows

    # write CSV
    header = [
        "kind","name","order_or_seg","mse","dwmse","source",
        "loss","ppl","baseline_loss","baseline_ppl","patched_loss","patched_ppl",
        "softmax","act","precision","block_size"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

    # pretty print
    print("\n=== Approximation Results on [-4,4] ===")
    if approx_rows:
        for r in approx_rows:
            print(f"{r['kind']:>4} | n={r['order_or_seg']:>3} | MSE={r['mse']:>10} | DWMSE={r['dwmse']:>10} | {r['name']}")

    if log_rows:
        print("\n=== Parsed Log Metrics ===")
        for r in log_rows:
            meta = " ".join(
                f"{k}={r[k]}" for k in ["softmax","act","precision","block_size"] if r.get(k)
            )
            ab = ""
            if r.get("baseline_loss") or r.get("patched_loss"):
                ab = f" | baseline: loss={r.get('baseline_loss','')} ppl={r.get('baseline_ppl','')} | patched: loss={r.get('patched_loss','')} ppl={r.get('patched_ppl','')}"
            line = f"{r['name']} | loss={r.get('loss','')} ppl={r.get('ppl','')}{ab}"
            if meta:
                line += f" | {meta}"
            print(line)

    print(f"\n[ok] Saved CSV to: {out_csv}")


if __name__ == "__main__":
    main()
