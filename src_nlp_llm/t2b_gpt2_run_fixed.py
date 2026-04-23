# t2_gpt2_run.py
# Evaluate GPT-2 (openai-community/gpt2) on WikiText-2 only.
# Computes mean loss & perplexity; supports:
#   - Attention softmax: torch | debug | pwl-N
#       (PWL: m3_udanf_fixed.PWLSoftmaxFixed in fixed-point)
#   - MLP activation  : torch | poly-N (m1_poly_act.PolyGelu) | pwl-N
#       (PWL: m3_udanf_fixed.PWLGeluFixed in fixed-point)

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.activations import GELUActivation, NewGELUActivation

# ---- Imports for fixed-point PWL (softmax + gelu) from m3 ----
try:
    from m3_udanf_fixed import PWLSoftmaxFixed, PWLGeluFixed, DebugSoftmax
except ImportError:
    # Fallbacks (keep math identical to torch for safety)
    class PWLSoftmaxFixed(nn.Module):
        def __init__(self, json_path: str, i_bits_x: int, f_bits_x: int,
                     i_bits_y: int, f_bits_y: int, debug: bool = False):
            super().__init__()
        def forward(self, x):
            return torch.softmax(x, dim=-1)

    class PWLGeluFixed(nn.Module):
        def __init__(self, json_path: str, i_bits_x: int, f_bits_x: int,
                     i_bits_y: int, f_bits_y: int):
            super().__init__()
        def forward(self, x):
            return torch.nn.functional.gelu(x)

    class DebugSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            print("  - [DEBUG-fallback] DebugSoftmax (torch.exp softmax).")
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            max_vals = torch.max(x, dim=-1, keepdim=True)[0]
            shifted_x = x - max_vals
            exp_x = torch.exp(shifted_x)
            return exp_x / (exp_x.sum(dim=-1, keepdim=True) + 1e-9)

# ---- Import for polynomial GELU (from m1_poly_act) ----
try:
    from m1_poly_act import PolyGelu
except ImportError:
    class PolyGelu(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            return torch.nn.functional.gelu(x)


# -----------------------
# Dataset: WikiText-2 only
# -----------------------

def load_wikitext2_test(cache_dir: Optional[str] = None):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    if "test" in ds:
        return ds["test"]
    # fallback if needed
    for alt in ["validation", "valid", "dev", "train"]:
        if alt in ds:
            return ds[alt]
    return ds[list(ds.keys())[0]]


def get_text_column_name(ds) -> str:
    for c in ["text", "sentence", "content"]:
        if c in ds.column_names:
            return c
    # fallback: rename first column to "text"
    first = ds.column_names[0]
    if first != "text":
        ds = ds.map(lambda x: {"text": str(x[first])})
        return "text"
    return first


def tokenize_and_group(dataset, tokenizer, text_col: str, block_size: int = 1024):
    """
    1) Batched tokenize (NO attention_mask)
    2) Concatenate token ids
    3) Truncate to multiple of block_size
    4) Split into equal blocks for input_ids & labels
    """
    def tok(batch):
        return tokenizer(batch[text_col], return_attention_mask=False)

    tokenized = dataset.map(
        tok,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # tokenized now has ONLY "input_ids"

    def group_texts(examples):
        ids_lists = examples["input_ids"]  # list of lists
        total_ids = sum(ids_lists, [])
        total_len = (len(total_ids) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": [], "labels": []}
        total_ids = total_ids[:total_len]
        blocks = [total_ids[i:i + block_size] for i in range(0, total_len, block_size)]
        result = {"input_ids": blocks, "labels": [b.copy() for b in blocks]}
        return result

    lm_ds = tokenized.map(group_texts, batched=True)
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return lm_ds


# -----------------------
# Helpers
# -----------------------

def parse_qformat(q: str) -> Tuple[int, int]:
    """
    Parse symmetric Q-format string "I.F" -> (i_bits, f_bits).
    Example: "6.10" -> (6, 10).
    """
    try:
        parts = q.strip().replace("_", ".").split(".")
        if len(parts) != 2:
            raise ValueError
        i_bits = int(parts[0])
        f_bits = int(parts[1])
        if i_bits <= 0 or f_bits < 0:
            raise ValueError
        return i_bits, f_bits
    except Exception:
        raise ValueError(f"Invalid Q-format string '{q}'. Expected like '6.10' or '9.4'.")


# -----------------------
# PWL / Poly loaders
# -----------------------

def make_softmax_module(
    softmax_mode: str,
    num_samples: int,
    segments: Optional[int],
    pwl_root: Path,
    sm_q: Tuple[int, int],
) -> Optional[nn.Module]:
    """
    softmax_mode:
      - 'torch'  -> nn.Softmax(dim=-1)
      - 'debug'  -> DebugSoftmax()
      - 'pwl-<N>'-> load <pwl_root>/<num_samples>/pwl_exp_sm_gpt2_<N>seg.json
                    and create m3_udanf_fixed.PWLSoftmaxFixed with Q-format sm_q
    sm_q: (i_bits, f_bits) for both input/output of exp PWL.
    """
    if softmax_mode == "torch":
        return nn.Softmax(dim=-1)
    if softmax_mode == "debug":
        return DebugSoftmax()
    if softmax_mode.startswith("pwl-"):
        seg = int(softmax_mode.split("-")[1])
        segments = segments or seg
        json_path = pwl_root / f"{num_samples}" / f"pwl_exp_sm_gpt2_{segments}seg.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"PWL exp json not found: {json_path}")
        i_bits_x, f_bits_x = sm_q
        i_bits_y, f_bits_y = sm_q
        return PWLSoftmaxFixed(
            str(json_path),
            i_bits_x=i_bits_x,
            f_bits_x=f_bits_x,
            i_bits_y=i_bits_y,
            f_bits_y=f_bits_y,
            debug=False,
        )
    raise ValueError(f"Invalid --softmax mode: {softmax_mode}")


def make_act_module(
    act_mode: str,
    num_samples: int,
    segments: Optional[int],
    pwl_root: Path,
    act_q: Tuple[int, int],
) -> Optional[nn.Module]:
    """
    act_mode:
      - 'torch'   -> None (no replacement)
      - 'poly-<N>'-> load dst_poly/poly_gelu_<N>_order.json (from t0_make_poly.py)
      - 'pwl-<N>' -> load <pwl_root>/<num_samples>/pwl_gelu_act_gpt2_<N>seg.json
                    and create m3_udanf_fixed.PWLGeluFixed with Q-format act_q
    act_q: (i_bits, f_bits) for both input/output of GELU PWL.
    """
    if act_mode == "torch":
        return None
    if act_mode.startswith("poly-"):
        order = int(act_mode.split("-")[1])
        json_path = Path(f"dst_poly/poly_gelu_{order}_order.json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Poly GELU json not found: {json_path}")
        return PolyGelu(str(json_path))
    if act_mode.startswith("pwl-"):
        seg = int(act_mode.split("-")[1])
        segments = segments or seg
        json_path = pwl_root / f"{num_samples}" / f"pwl_gelu_act_gpt2_{segments}seg.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"PWL GELU json not found: {json_path}")
        i_bits_x, f_bits_x = act_q
        i_bits_y, f_bits_y = act_q
        return PWLGeluFixed(
            str(json_path),
            i_bits_x=i_bits_x,
            f_bits_x=f_bits_x,
            i_bits_y=i_bits_y,
            f_bits_y=f_bits_y,
        )
    raise ValueError(f"Invalid --act mode: {act_mode}")


# -----------------------
# Injection utilities
# -----------------------

def inject_gpt2_pwl_softmax(model: nn.Module, sm_mod: Optional[nn.Module]):
    """Patch GPT2Attention._attn to use custom softmax module sm_mod(x)."""
    if sm_mod is None:
        return

    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    import types

    def _patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            q_len, k_len = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, k_len - q_len: k_len, :k_len].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Use provided module for softmax (torch | debug | pwl)
        attn_probs = sm_mod(attn_weights)
        attn_probs = self.attn_dropout(attn_probs)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, value)
        return attn_output, attn_probs

    replaced = 0
    for mod in model.modules():
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as GPT2AttnCls
        if isinstance(mod, GPT2AttnCls):
            mod._attn = types.MethodType(_patched_attn, mod)
            replaced += 1
    print(f"[inject] Patched softmax in {replaced} GPT2Attention layers.")


def inject_act_into_gpt2_mlp(model: nn.Module, act_mod: Optional[nn.Module], device: torch.device):
    """
    Replace GELU activations in GPT-2 MLP with the provided module (PolyGelu/PWLGeluFixed).
    We replace:
      - torch.nn.GELU
      - transformers.activations.GELUActivation
      - transformers.activations.NewGELUActivation
      - module.act if it is an instance of the above types
    """
    if act_mod is None:
        print("[inject] No GELU injection (act=torch).")
        return

    class _ActWrapper(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    act_mod = act_mod.to(device)

    replaced = 0
    # Replace leaf activations by walking the module tree
    for name, module in list(model.named_modules()):
        # Replace direct activation modules
        if isinstance(module, (nn.GELU, GELUActivation, NewGELUActivation)):
            parent = _get_parent(model, name)
            if parent is not None:
                child = name.split(".")[-1]
                setattr(parent, child, _ActWrapper(act_mod))
                replaced += 1

        # In GPT-2 MLP, the attribute is usually 'act'
        if hasattr(module, "act") and isinstance(
            module.act, (nn.GELU, GELUActivation, NewGELUActivation)
        ):
            module.act = _ActWrapper(act_mod)
            replaced += 1

    print(f"[inject] Replaced {replaced} GELU activations with custom act ({act_mod.__class__.__name__}).")


def _get_parent(root: nn.Module, full_name: str) -> Optional[nn.Module]:
    parts = full_name.split(".")
    if len(parts) == 1:
        return None
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent


# -----------------------
# Evaluation
# -----------------------

@torch.no_grad()
def eval_lm(model, dataloader, device, use_fp16: bool = False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if use_fp16:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        # Sum by number of tokens in batch
        num_tokens = labels.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf")
    return mean_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on WikiText-2 (loss & PPL).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])

    # Softmax options: torch | debug | pwl-N
    parser.add_argument(
        "--softmax",
        type=str,
        default="torch",
        choices=["torch", "debug", "pwl-4", "pwl-6", "pwl-8", "pwl-10", "pwl-12", "pwl-14", "pwl-16"],
        help="Replace attention softmax (requires PWL JSON for 'pwl-*').",
    )

    # Act options: torch | poly-N | pwl-N
    parser.add_argument(
        "--act",
        type=str,
        default="torch",
        help="MLP activation: 'torch' | 'poly-N' (m1_poly_act.PolyGelu) | 'pwl-N' (m3_udanf_fixed.PWLGeluFixed)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Used to locate dst_pwl*/<num_samples>/*.json for PWL modes.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override segments for PWL file lookup (defaults to number in --softmax/--act).",
    )

    # Which PWL set to use (matches t0_make_pwl.py)
    parser.add_argument(
        "--loss",
        type=str,
        default="dwmse",
        choices=["dwmse", "mse"],
        help='Which PWL set to use: "dwmse" -> dst_pwl/, "mse" -> dst_pwl_mse/.',
    )

    # NEW: Q-format for fixed-point PWL Softmax and GELU
    parser.add_argument(
        "--sm_q",
        type=str,
        default="6.10",
        help="Symmetric Q-format for Softmax/EXP data (e.g., '6.10' -> Q6.10).",
    )
    parser.add_argument(
        "--act_q",
        type=str,
        default="9.4",
        help="Symmetric Q-format for ACT/GELU data (e.g., '9.4' -> Q9.4).",
    )

    args = parser.parse_args()

    # Parse Q-formats
    sm_q = parse_qformat(args.sm_q)
    act_q = parse_qformat(args.act_q)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16"
    print(f"Using device: {device} | Precision: {args.precision}")
    print(f"[config] sm_q={args.sm_q} (parsed={sm_q}), act_q={args.act_q} (parsed={act_q})")

    # Decide PWL root dir based on loss
    pwl_root = Path("dst_pwl") if args.loss == "dwmse" else Path("dst_pwl_mse")
    print(f"[config] PWL root: {pwl_root} (loss={args.loss})")

    # Load GPT-2
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)
    model.to(device)

    # Inject attention softmax
    sm_mod = make_softmax_module(
        args.softmax,
        num_samples=args.num_samples,
        segments=args.segments,
        pwl_root=pwl_root,
        sm_q=sm_q,
    )
    inject_gpt2_pwl_softmax(model, sm_mod)

    # Inject activation
    act_mod = make_act_module(
        args.act,
        num_samples=args.num_samples,
        segments=args.segments,
        pwl_root=pwl_root,
        act_q=act_q,
    )
    inject_act_into_gpt2_mlp(model, act_mod, device=device)

    # Dataset
    ds = load_wikitext2_test(cache_dir=args.cache_dir)
    text_col = get_text_column_name(ds)
    lm_ds = tokenize_and_group(ds, tokenizer, text_col=text_col, block_size=args.block_size)
    if len(lm_ds) == 0:
        print(f"[error] No full {args.block_size}-token blocks in WikiText-2; exiting.")
        return
    dl = DataLoader(lm_ds, batch_size=args.batch_size, shuffle=False)

    loss, ppl = eval_lm(model, dl, device=device, use_fp16=use_fp16)
    print(f"\n=== WikiText-2 Results ===")
    print(f"  mean_loss={loss:.8f}")
    print(f"  perplexity={ppl:.3f}")
    print(
        f"  (softmax={args.softmax}, act={args.act}, loss_set={args.loss}, "
        f"sm_q={args.sm_q}, act_q={args.act_q}, "
        f"precision={args.precision}, block_size={args.block_size})"
    )


if __name__ == "__main__":
    main()
