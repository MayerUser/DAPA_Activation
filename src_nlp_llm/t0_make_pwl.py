# t0_make_pwl.py
import argparse
import json
from pathlib import Path
import warnings
import types
import tempfile
import os
import atexit
import shutil

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
SEQ_LENGTH = 512

MODEL_MAPPING = {
    "gpt2": {"path": "openai-community/gpt2", "type": "language"},
    "llama2-7b": {"path": "meta-llama/Llama-2-7b-hf", "type": "language"},
    "llama3-8b": {"path": "./local_llama3_8b", "type": "language"},
}

temp_dir = tempfile.mkdtemp()
captured_data = {
    "softmax_input": {"files": [], "min": np.inf, "max": -np.inf},
    "gelu_input": {"files": [], "min": np.inf, "max": -np.inf},
}

def cleanup_temp_dir():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
atexit.register(cleanup_temp_dir)

def save_hook_data(key: str, data_tensor: torch.Tensor):
    if data_tensor is None or data_tensor.numel() == 0: return
    
    # 强制 Clamp 去除极端值保护求解器
    if "softmax" in key:
        data_tensor = torch.clamp(data_tensor, min=-128.0)
    elif "gelu" in key or "act" in key:
        data_tensor = torch.clamp(data_tensor, min=-64.0, max=64.0)
        
    data = data_tensor.detach().cpu().to(torch.float32).numpy().flatten()
    current_min, current_max = np.min(data), np.max(data)
    if current_min < captured_data[key]["min"]: captured_data[key]["min"] = current_min
    if current_max > captured_data[key]["max"]: captured_data[key]["max"] = current_max
    
    temp_path = os.path.join(temp_dir, f"{key}_{len(captured_data[key]['files'])}.npy")
    np.save(temp_path, data)
    captured_data[key]["files"].append(temp_path)

def get_model_and_attach_hooks(model_name: str, device: torch.device, cache_dir: str = None):
    model_path = MODEL_MAPPING[model_name]["path"]
    
    if "llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir).to(device)
    
    model.eval()

    # --- LLaMA 拦截逻辑 ---
    if "llama" in model_name:
        import transformers.models.llama.modeling_llama as llama_modeling
        
        # 1. 拦截 LLaMA Softmax
        original_softmax = llama_modeling.nn.functional.softmax
        def custom_llama_softmax(input, dim=None, dtype=None, **kwargs):
            max_vals = torch.max(input, dim=-1, keepdim=True)[0]
            save_hook_data("softmax_input", input - max_vals)
            return original_softmax(input, dim=dim, dtype=dtype, **kwargs)
        llama_modeling.nn.functional.softmax = custom_llama_softmax
        
        # 2. 拦截 LLaMA SiLU (直接修改 LlamaMLP 的 forward)
        for name, module in model.named_modules():
            if isinstance(module, llama_modeling.LlamaMLP):
                original_mlp = module.forward
                def patched_llama_mlp(self, x):
                    gate_out = self.gate_proj(x)
                    save_hook_data("gelu_input", gate_out) # 捕获 SiLU 输入
                    return self.down_proj(self.act_fn(gate_out) * self.up_proj(x))
                module.forward = types.MethodType(patched_llama_mlp, module)

    # --- GPT-2 拦截逻辑 ---
    for name, module in model.named_modules():
        # 1. 拦截 GPT-2 Attention
        if isinstance(module, GPT2Attention):
            original_attn = module._attn
            def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
                if self.scale_attn_weights:
                    attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
                max_vals = torch.max(attn_weights, dim=-1, keepdim=True)[0]
                save_hook_data("softmax_input", attn_weights - max_vals)
                return original_attn(query, key, value, attention_mask, head_mask)
            module._attn = types.MethodType(patched_attn, module)

        # 2. 拦截 GPT-2 GELU (因为它是函数，必须修改 GPT2MLP 的 forward)
        if isinstance(module, GPT2MLP):
            original_mlp = module.forward
            def patched_gpt2_mlp(self, hidden_states):
                hidden_states = self.c_fc(hidden_states)
                save_hook_data("gelu_input", hidden_states) # 捕获 GELU 输入
                hidden_states = self.act(hidden_states)
                hidden_states = self.c_proj(hidden_states)
                hidden_states = self.dropout(hidden_states)
                return hidden_states
            module.forward = types.MethodType(patched_gpt2_mlp, module)

    return model

def get_data_loader(num_samples: int, batch_size: int, model_name: str, cache_dir: str = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name]["path"], cache_dir=cache_dir)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    
    def tokenize_function(examples):
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["text"], truncation=False)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.select(range(min(num_samples, len(tokenized_dataset))))
        
    def group_texts(examples):
        ids = sum(examples["input_ids"], [])
        total_len = (len(ids) // SEQ_LENGTH) * SEQ_LENGTH
        if total_len == 0: return {"input_ids": []}
        res = [ids[i : i + SEQ_LENGTH] for i in range(0, total_len, SEQ_LENGTH)]
        # 彻底抛弃 attention_mask，解决 ArrowInvalid 报错
        return {"input_ids": res}
        
    # remove_columns 强制对齐，彻底杀死 Arrow 引擎报错
    processed_dataset = tokenized_dataset.map(group_texts, batched=True, remove_columns=tokenized_dataset.column_names)
    
    def collate_fn(batch):
        ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        return {"input_ids": ids}
        
    return DataLoader(processed_dataset, batch_size=batch_size, collate_fn=collate_fn)

def get_pdf_from_files(key: str, bins=131072):
    file_list = captured_data[key]["files"]
    if not file_list: return None, None
    g_min, g_max = captured_data[key]["min"] - 1e-6, captured_data[key]["max"] + 1e-6
    g_bins = np.linspace(g_min, g_max, bins + 1)
    g_hist = np.zeros(bins, dtype=np.int64)
    total = 0

    for f in tqdm(file_list, desc=f"Processing {key}"):
        data = np.load(f)
        h, _ = np.histogram(data, bins=g_bins)
        g_hist += h
        total += len(data)
        os.remove(f)
    
    return g_hist / (total * (g_bins[1] - g_bins[0])), g_bins

def get_uniform_segment_points(hist, bin_edges, num_segments: int):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = np.cumsum(hist) / np.sum(hist)
    return np.interp(np.linspace(0, 1, num_segments + 1), cdf, bin_centers)

def solve_poly(I, J, degree):
    try:
        if degree == 1:
            return dict(zip(["p1", "p0"], np.linalg.solve([[I[2], I[1]], [I[1], I[0]]], [J[1], J[0]])))
    except: return None

def generate_pwl(func, pts, hist, bin_edges, degree, loss, save_path: Path):
    pwl = {"intervals": [], "params": []}
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bw = bin_edges[1] - bin_edges[0]
    
    for i in range(len(pts) - 1):
        x_min, x_max = pts[i], pts[i+1]
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        seg_x, seg_h = bin_centers[mask], hist[mask]
        if len(seg_x) == 0: continue
        
        y = func(seg_x)
        x_pows = np.vander(seg_x, 2 * degree + 1, increasing=True).T 
        w = (x_pows * seg_h * bw) if loss == "dwmse" else (x_pows * bw)
        I, J = np.sum(w, axis=1), np.sum(w[:degree+1] * y, axis=1)
        params = solve_poly(I, J, degree)
        if params:
            pwl["intervals"].append((f"{x_min:.6f}", f"{x_max:.6f}"))
            pwl["params"].append(params)
    
    with open(save_path, 'w') as f: json.dump(pwl, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--segment_number", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--loss", type=str, default="dwmse")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_and_attach_hooks(args.model_name, device, cache_dir=args.cache_dir)
    
    bs = 4 if "llama" in args.model_name else 16
    dl = get_data_loader(args.num_samples, bs, args.model_name, cache_dir=args.cache_dir)

    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            batch_on_device = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            model(**batch_on_device)

    gelu_np = lambda x: x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))
    silu_np = lambda x: x / (1.0 + np.exp(-x))
    
    act_func = silu_np if "llama" in args.model_name else gelu_np
    
    tasks = {
        "exp_sm": {"data_key": "softmax_input", "func": np.exp, "degree": 1},
        "gelu_act": {"data_key": "gelu_input", "func": act_func, "degree": 1}
    }

    pwl_dir = Path(f"dst_pwl/{args.num_samples}")
    pwl_dir.mkdir(parents=True, exist_ok=True)

    for k, cfg in tasks.items():
        hist, bins = get_pdf_from_files(cfg["data_key"])
        if hist is None:
            print(f"!!! Error: No data for {k}. Check hooks. !!!")
            continue
        
        pts = np.unique(get_uniform_segment_points(hist, bins, args.segment_number))
        save_path = pwl_dir / f"pwl_{k}_{args.model_name}_{args.segment_number}seg.json"
        generate_pwl(cfg["func"], pts, hist, bins, cfg["degree"], args.loss, save_path)
        print(f"--- Generated {save_path} ---")

if __name__ == "__main__":
    main()