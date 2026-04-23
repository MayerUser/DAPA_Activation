# t3_llama_demo.py
import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from m3_udanf_fixed import PWLSoftmaxFixed, PWLGeluFixed, DebugSoftmax
except ImportError:
    class PWLSoftmaxFixed(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return torch.softmax(x, dim=-1)
    class PWLGeluFixed(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return torch.nn.functional.silu(x)

def load_wikitext2_test(cache_dir=None):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    return ds.get("test", ds[list(ds.keys())[0]])

def tokenize_and_group(dataset, tokenizer, block_size: int = 1024):
    tokenized = dataset.map(lambda b: tokenizer(b["text"], return_attention_mask=False), batched=True, remove_columns=dataset.column_names)
    def group_texts(examples):
        total_ids = sum(examples["input_ids"], [])
        total_len = (len(total_ids) // block_size) * block_size
        blocks = [total_ids[i:i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": blocks, "labels": [b.copy() for b in blocks]}
    
    lm_ds = tokenized.map(group_texts, batched=True)
    # [FIX]: Removed lm_ds.set_format(type="torch")
    return lm_ds

# [FIX]: Custom collate_fn to manually tensorize data, bypassing VideoReader bug
def custom_collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}

def parse_qformat(q: str) -> Tuple[int, int]:
    parts = q.strip().replace("_", ".").split(".")
    return int(parts[0]), int(parts[1])

def make_softmax_module(softmax_mode, model_name_short, num_samples, segments, pwl_root, sm_q):
    if softmax_mode == "torch": return None
    if softmax_mode.startswith("pwl-"):
        seg = int(softmax_mode.split("-")[1])
        json_path = pwl_root / f"{num_samples}" / f"pwl_exp_sm_{model_name_short}_{segments or seg}seg.json"
        return PWLSoftmaxFixed(str(json_path), sm_q[0], sm_q[1], sm_q[0], sm_q[1], debug=False)
    raise ValueError(f"Invalid softmax mode: {softmax_mode}")

def make_act_module(act_mode, model_name_short, num_samples, segments, pwl_root, act_q):
    if act_mode == "torch": return None
    if act_mode.startswith("pwl-"):
        seg = int(act_mode.split("-")[1])
        json_path = pwl_root / f"{num_samples}" / f"pwl_gelu_act_{model_name_short}_{segments or seg}seg.json"
        return PWLGeluFixed(str(json_path), act_q[0], act_q[1], act_q[0], act_q[1])
    raise ValueError(f"Invalid act mode: {act_mode}")

def inject_llama_pwl_softmax(sm_mod):
    if sm_mod is None: return
    import transformers.models.llama.modeling_llama as llama_modeling
    def custom_softmax(input, dim=None, dtype=None, **kwargs):
        res = sm_mod(input)
        return res.to(dtype) if dtype is not None else res
    llama_modeling.nn.functional.softmax = custom_softmax
    print("[inject] Globally patched llama_modeling.nn.functional.softmax")

def inject_act_into_llama_mlp(model, act_mod):
    if act_mod is None: return
    class _ActWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x): return self.inner(x)
        
    replaced = 0
    for name, module in model.named_modules():
        # LLaMA uses SiLU natively, hook into act_fn
        if hasattr(module, "act_fn") and isinstance(module.act_fn, nn.SiLU):
            module.act_fn = _ActWrapper(act_mod)
            replaced += 1
    print(f"[inject] Replaced {replaced} SiLU activations in LLaMA MLP.")

@torch.no_grad()
def eval_lm(model, dataloader, use_fp16=False):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    # Extract root device from model configured by device_map="auto"
    device = model.device

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(input_ids=input_ids, labels=labels).loss
        else:
            loss = model(input_ids=input_ids, labels=labels).loss

        num_tokens = labels.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, math.exp(mean_loss) if math.isfinite(mean_loss) else float("inf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2) # Small batch for 7B/8B
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--softmax", type=str, default="torch")
    parser.add_argument("--act", type=str, default="torch")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--loss", type=str, default="dwmse")
    parser.add_argument("--sm_q", type=str, default="6.10")
    parser.add_argument("--act_q", type=str, default="9.4")
    # Added argument to select specific llama version to keep json names matched
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()

    sm_q, act_q = parse_qformat(args.sm_q), parse_qformat(args.act_q)
    pwl_root = Path("dst_pwl") if args.loss == "dwmse" else Path("dst_pwl_mse")
    
    print(f"[config] precision={args.precision}, sm_q={args.sm_q}, act_q={args.act_q}")

    model_name = args.model_name
    model_name_short = "llama2-7b" if "Llama-2" in model_name else "llama3-8b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load with Accelerate to distribute across your two A5000s
    # Note: Removed attn_implementation="eager" for 4.35.2 support
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=args.cache_dir, 
        torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
        device_map="auto"
    )

    sm_mod = make_softmax_module(args.softmax, model_name_short, args.num_samples, args.segments, pwl_root, sm_q)
    inject_llama_pwl_softmax(sm_mod)

    act_mod = make_act_module(args.act, model_name_short, args.num_samples, args.segments, pwl_root, act_q)
    # Ensure custom module is on the root device to avoid mismatch 
    if act_mod is not None: act_mod = act_mod.to(model.device) 
    inject_act_into_llama_mlp(model, act_mod)

    ds = load_wikitext2_test(cache_dir=args.cache_dir)
    lm_ds = tokenize_and_group(ds, tokenizer, block_size=args.block_size)
    
    # [FIX]: Use custom_collate_fn
    dl = DataLoader(lm_ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    loss, ppl = eval_lm(model, dl, use_fp16=(args.precision=="fp16"))
    print(f"\n=== {model_name_short.upper()} WikiText-2 Results ===")
    print(f"  mean_loss={loss:.8f} | perplexity={ppl:.3f}")

if __name__ == "__main__":
    main()