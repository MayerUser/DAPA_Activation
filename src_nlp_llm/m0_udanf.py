# m0_udanf.py
# Piecewise-polynomial (PWL) utilities for exp / GELU and an attention softmax using PWL exp.

import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

class PWLFunction(nn.Module):
    """
    Generic PWL polynomial evaluator (up to cubic):
      y = p3*x^3 + p2*x^2 + p1*x + p0
    JSON format:
      {
        "intervals": [[x0, x1], [x1, x2], ...],
        "params":    [{"p3":..., "p2":..., "p1":..., "p0":...}, ...]
      }
    """
    def __init__(self, json_path: str):
        super().__init__()

        jp = Path(json_path)
        if not jp.is_file():
            raise FileNotFoundError(f"JSON file not found at: {json_path}")

        with open(jp, "r") as f:
            pwl_data = json.load(f)

        # Right-edge boundaries for all but last interval.
        boundaries = []
        for interval in pwl_data["intervals"][:-1]:
            boundaries.append(float(interval[1]))

        params = pwl_data["params"]
        num_segments = len(params)

        p3 = torch.zeros(num_segments)
        p2 = torch.zeros(num_segments)
        p1 = torch.zeros(num_segments)
        p0 = torch.zeros(num_segments)

        for i, p in enumerate(params):
            p3[i] = p.get("p3", 0.0)
            p2[i] = p.get("p2", 0.0)
            p1[i] = p.get("p1", 0.0)
            p0[i] = p.get("p0", 0.0)

        # Register as buffers so they follow .to(device/dtype)
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer("p3", p3.float())
        self.register_buffer("p2", p2.float())
        self.register_buffer("p1", p1.float())
        self.register_buffer("p0", p0.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep buffers aligned with x (device/dtype) without forcing fp32
        dev, dt = x.device, x.dtype
        boundaries = self.boundaries.to(device=dev, dtype=dt)
        p3 = self.p3.to(device=dev, dtype=dt)
        p2 = self.p2.to(device=dev, dtype=dt)
        p1 = self.p1.to(device=dev, dtype=dt)
        p0 = self.p0.to(device=dev, dtype=dt)

        # Segment indices
        idx = torch.bucketize(x, boundaries)

        # Coeff selection and polynomial eval
        a3 = p3[idx]
        a2 = p2[idx]
        a1 = p1[idx]
        a0 = p0[idx]
        return a3 * (x ** 3) + a2 * (x ** 2) + a1 * x + a0


# Semantic wrappers
class PWLSqrt(PWLFunction):
    def __init__(self, json_path: str):
        super().__init__(json_path)


class PWLExp(PWLFunction):
    def __init__(self, json_path: str):
        super().__init__(json_path)


class PWLGelu(PWLFunction):
    def __init__(self, json_path: str):
        super().__init__(json_path)


class PWLSoftmax(nn.Module):
    """
    Softmax using PWL approximation of exp():
      - max-shift stabilization
      - CLAMPS inputs to the PWL JSON domain to avoid overflow outside fit range
      - ReLU after PWL exp to enforce non-negativity
      - Optional debug stats
    """
    def __init__(self, json_path: str, debug: bool = True, print_first_batches: int = 3, print_kl: bool = False):
        super().__init__()
        # Load PWL exp and also parse domain bounds from the same JSON
        self.pwl_exp = PWLExp(json_path)
        with open(json_path, "r") as f:
            _p = json.load(f)
        # JSON intervals are like: [["-inf","a1"], ["a1","a2"], ..., ["aN","+inf"]]
        # We'll take the *finite* bounds of the first and last finite intervals for clamping.
        # If the ends are +/-inf, we pull the nearest finite numbers.
        intervals = _p["intervals"]
        # collect numeric bounds
        numeric_bounds = []
        for lo, hi in intervals:
            try:
                lo_f = float(lo.replace("inf", "inf"))
            except:
                lo_f = float("-inf") if "-inf" in lo else float(lo)
            try:
                hi_f = float(hi.replace("inf", "inf"))
            except:
                hi_f = float("inf") if "+inf" in hi or "inf" in hi else float(hi)
            numeric_bounds.append((lo_f, hi_f))

        # derive clamp_min/clamp_max from the nearest finite neighbors
        finite_lows  = [lo for (lo, _) in numeric_bounds if np.isfinite(lo)]
        finite_highs = [hi for (_, hi) in numeric_bounds if np.isfinite(hi)]
        # if no finite bound on a side, fall back to a safe default
        self.clamp_min = float(min(finite_lows))  if len(finite_lows)  else -150.0
        self.clamp_max = float(max(finite_highs)) if len(finite_highs) else  80.0

        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self.debug = bool(debug)
        self.print_first_batches = int(print_first_batches)
        self.print_kl = bool(print_kl)
        self._dbg_cnt = 0

    @staticmethod
    def _stats(t: torch.Tensor):
        t32 = t.detach().to(torch.float32)
        return dict(
            mean=t32.mean().item(),
            var=t32.var(unbiased=False).item(),
            min=t32.min().item(),
            max=t32.max().item(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        x = torch.clamp(x, min=-128,max=127)
        # Stable shift
        max_vals = torch.max(x, dim=-1, keepdim=True)[0]
        shifted = x - max_vals
        # --- CRITICAL: clamp to the fit domain of the PWL model ---
        # This prevents polynomial overflow for very negative scores (e.g., -3e38)
        # clamped = torch.clamp(shifted, min=self.clamp_min, max=self.clamp_max)

        # Reference (ideal) for debug/metrics only
        torch_exp = torch.exp(shifted)
        ideal_sm = torch_exp / (torch_exp.sum(dim=-1, keepdim=True) + torch.finfo(torch_exp.dtype).eps)

        # PWL path (use clamped inputs)
        pwl_exp = self.pwl_exp(shifted)
        pwl_exp = torch.relu(pwl_exp)  # ensure non-negative
        denom_p = pwl_exp.sum(dim=-1, keepdim=True)
        eps_p = torch.finfo(pwl_exp.dtype).eps if pwl_exp.dtype.is_floating_point else 1e-9
        pwl_sm = pwl_exp / (denom_p + eps_p)

        # Fallback if anything still non-finite (paranoia)
        # if not torch.isfinite(pwl_sm).all():
        #     pwl_sm = torch.softmax(shifted, dim=-1)

        if self.debug and self._dbg_cnt < self.print_first_batches:
            with torch.no_grad():
                s_in   = self._stats(x)
                s_sh   = self._stats(shifted)
                s_texp = self._stats(torch_exp)
                s_pexp = self._stats(pwl_exp)
                s_tsm  = self._stats(ideal_sm)
                s_psm  = self._stats(pwl_sm)

                mae_exp = torch.mean(torch.abs(pwl_exp - torch_exp)).item()
                mae_sm  = torch.mean(torch.abs(pwl_sm  - ideal_sm)).item()

                lines = [
                    "\n--- PWLSoftmax Debug ---",
                    f"[input logits]        mean={s_in['mean']:.6f} var={s_in['var']:.6f} min={s_in['min']:.6f} max={s_in['max']:.6f}",
                    f"[shifted logits]      mean={s_sh['mean']:.6f} var={s_sh['var']:.6f} min={s_sh['min']:.6f} max={s_sh['max']:.6f}",
                    f"[torch  exp]          mean={s_texp['mean']:.6f} var={s_texp['var']:.6f} min={s_texp['min']:.6f} max={s_texp['max']:.6f}",
                    f"[pwl    exp]          mean={s_pexp['mean']:.6f} var={s_pexp['var']:.6f} min={s_pexp['min']:.6f} max={s_pexp['max']:.6f}",
                    f"[ideal softmax]       mean={s_tsm['mean']:.6f} var={s_tsm['var']:.6f} min={s_tsm['min']:.6f} max={s_tsm['max']:.6f}",
                    f"[pwl   softmax]       mean={s_psm['mean']:.6f} var={s_psm['var']:.6f} min={s_psm['min']:.6f} max={s_psm['max']:.6f}",
                    f"[error]               MAE(exp)={mae_exp:.8f}  MAE(softmax)={mae_sm:.8f}",
                    "------------------------\n",
                ]
                if self.print_kl:
                    eps = 1e-12
                    p = torch.clamp(ideal_sm, min=eps)
                    q = torch.clamp(pwl_sm,   min=eps)
                    kl_pq = torch.mean(p * (torch.log(p) - torch.log(q))).item()
                    kl_qp = torch.mean(q * (torch.log(q) - torch.log(p))).item()
                    lines.insert(-1, f"[error]               KL(ideal||pwl)={kl_pq:.8f}  KL(pwl||ideal)={kl_qp:.8f}")
                print("\n".join(lines))
                self._dbg_cnt += 1

        return pwl_sm


class DebugSoftmax(nn.Module):
    """
    Standard numerically-stable softmax (max-shift) packaged as a module.
    Useful for exercising the patching path without changing math.
    Includes a 'calls' counter buffer for verification.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        m = torch.max(x, dim=-1, keepdim=True)[0]
        ex = torch.exp(x - m)
        denom = ex.sum(dim=-1, keepdim=True)
        eps = torch.finfo(ex.dtype).eps if ex.dtype.is_floating_point else 1e-9
        return ex / (denom + eps)
