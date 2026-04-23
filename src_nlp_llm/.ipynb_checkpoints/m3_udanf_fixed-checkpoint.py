# t3_llama_demo.py
import argparse
import math
import types
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入定点 PWL 模块
try:
    from m3_udanf_fixed import PWLSoftmaxFixed, PWLGeluFixed
except ImportError:
    PWLSoftmaxFixed, PWLGeluFixed = None, None

# 尝试导入浮点 PWL 模块
try:
    from m0_udanf import PWLSoftmax, PWLGelu
except ImportError:
    PWLSoftmax, PWLGelu = None, None

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
    return lm_ds

def custom_collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}

def parse_qformat(q: str):
    if q.lower() == "float": return None
    parts = q.strip().replace("_", ".").split(".")
    return int(parts[0]), int(parts[1])

def make_softmax_module(softmax_mode, model_name_short, num_samples, segments, pwl_root, sm_q_str):
    if softmax_mode == "torch": return None
    if softmax_mode.startswith("pwl-"):
        seg = int(softmax_mode.split("-")[1])
        json_path = pwl_root / f"{num_samples}" / f"pwl_exp_sm_{model_name_short}_{segments or seg}seg.json"
        
        # 核心修改 1：支持浮点模式
        if sm_q_str.lower() == "float":
            print(f"  - Loading FLOAT PWL Softmax from {json_path.name}")
            return PWLSoftmax(str(json_path), debug=False)
        else:
            sm_q = parse_qformat(sm_q_str)
            return PWLSoftmaxFixed(str(json_path), sm_q[0], sm_q[1], sm_q[0], sm_q[1], debug=False)
    raise ValueError(f"Invalid softmax mode: {softmax_mode}")

def make_act_module(act_mode, model_name_short, num_samples, segments, pwl_root, act_q_str):
    if act_mode == "torch": return None
    if act_mode.startswith("pwl-"):
        seg = int(act_mode.split("-")[1])
        json_path = pwl_root / f"{num_samples}" / f"pwl_gelu_act_{model_name_short}_{segments or seg}seg.json"
        
        # 核心修改 1：支持浮点模式
        if act_q_str.lower() == "float":
            print(f"  - Loading FLOAT PWL Activation from {json_path.name}")
            # LLaMA 实际上用的是 SiLU，由于 m0_udanf 中的 PWLGelu 其实只是个加载 PWL JSON 的通用层，
            # 我们直接用它加载 SiLU 的参数完全没问题。
            return PWLGelu(str(json_path)) 
        else:
            act_q = parse_qformat(act_q_str)
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
    replaced = 0
    for name, module in model.named_modules():
        # 核心修改 2：彻底放弃对 act_fn 的 isinstance 检查
        # 直接通过类名精准定位 LlamaMLP，并暴力重写它的 forward 函数！
        if module.__class__.__name__ == "LlamaMLP":
            def patched_mlp_forward(self, x):
                # LLaMA MLP 的标准公式：down_proj(act_fn(gate_proj(x)) * up_proj(x))
                gate_out = self.gate_proj(x)
                act_out = act_mod(gate_out) # 将原本的 act_fn 替换为我们的 PWL 模块
                return self.down_proj(act_out * self.up_proj(x))
            module.forward = types.MethodType(patched_mlp_forward, module)
            replaced += 1
    print(f"[inject] Replaced {replaced} SiLU activations in LLaMA MLP.")

@torch.no_grad()
def eval_lm(model, dataloader, use_fp16=False):
    model.eval()
    total_loss, total_tokens = 0.0, 0
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
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--softmax", type=str, default="torch")
    parser.add_argument("--act", type=str, default="torch")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--loss", type=str, default="dwmse")
    
    # 默认值改为 "float"，意味着如果不传，默认跑浮点
    parser.add_argument("--sm_q", type=str, default="float")
    parser.add_argument("--act_q", type=str, default="float")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()

    pwl_root = Path("dst_pwl") if args.loss == "dwmse" else Path("dst_pwl_mse")
    print(f"[config] precision={args.precision}, sm_q={args.sm_q}, act_q={args.act_q}")

    model_name = args.model_name
    model_name_short = "llama2-7b" if "Llama-2" in model_name else "llama3-8b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=args.cache_dir, 
        torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
        device_map="auto"
    )

    sm_mod = make_softmax_module(args.softmax, model_name_short, args.num_samples, args.segments, pwl_root, args.sm_q)
    inject_llama_pwl_softmax(sm_mod)

    act_mod = make_act_module(args.act, model_name_short, args.num_samples, args.segments, pwl_root, args.act_q)
    if act_mod is not None: act_mod = act_mod.to(model.device) 
    inject_act_into_llama_mlp(model, act_mod)

    ds = load_wikitext2_test(cache_dir=args.cache_dir)
    lm_ds = tokenize_and_group(ds, tokenizer, block_size=args.block_size)
    dl = DataLoader(lm_ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    loss, ppl = eval_lm(model, dl, use_fp16=(args.precision=="fp16"))
    print(f"\n=== {model_name_short.upper()} WikiText-2 Results ===")
    print(f"  mean_loss={loss:.8f} | perplexity={ppl:.3f}")

if __name__ == "__main__":
    main()