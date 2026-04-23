# t4_llama_generate.py
import argparse
import types
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 导入定点模块 ---
try:
    from m3_udanf_fixed import PWLSoftmaxFixed, PWLGeluFixed
except ImportError:
    PWLSoftmaxFixed, PWLGeluFixed = None, None

# --- 导入浮点模块 ---
try:
    from m0_udanf import PWLSoftmax, PWLGelu
except ImportError:
    PWLSoftmax, PWLGelu = None, None

def parse_qformat(q: str):
    if q is None or q.lower() == "float": return None
    parts = q.strip().replace("_", ".").split(".")
    return int(parts[0]), int(parts[1])

def make_softmax_module(softmax_mode, model_name_short, num_samples, segments, pwl_root, sm_q_str):
    if softmax_mode == "torch": return None
    if softmax_mode.startswith("pwl-"):
        seg = int(softmax_mode.split("-")[1])
        json_path = pwl_root / f"{num_samples}" / f"pwl_exp_sm_{model_name_short}_{segments or seg}seg.json"
        
        if sm_q_str is None or sm_q_str.lower() == "float":
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
        
        if act_q_str is None or act_q_str.lower() == "float":
            print(f"  - Loading FLOAT PWL Activation from {json_path.name}")
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
        # 暴力拦截 LlamaMLP 类，解决 4.35.2 下 isinstance 失败的问题
        if module.__class__.__name__ == "LlamaMLP":
            def patched_mlp_forward(self, x):
                gate_out = self.gate_proj(x)
                act_out = act_mod(gate_out)
                return self.down_proj(act_out * self.up_proj(x))
            module.forward = types.MethodType(patched_mlp_forward, module)
            replaced += 1
    print(f"[inject] Replaced {replaced} SiLU activations in LLaMA MLP.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence in chip design is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--softmax", type=str, default="torch")
    parser.add_argument("--act", type=str, default="torch")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--loss", type=str, default="dwmse")
    parser.add_argument("--sm_q", type=str, default="float")
    parser.add_argument("--act_q", type=str, default="float")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()

    pwl_root = Path("dst_pwl") if args.loss == "dwmse" else Path("dst_pwl_mse")
    
    model_name = args.model_name
    model_name_short = "llama2-7b" if "Llama-2" in model_name else "llama3-8b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=args.cache_dir, 
        torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
        device_map="auto"
    )

    # 注入定点/浮点模块
    sm_mod = make_softmax_module(args.softmax, model_name_short, args.num_samples, args.segments, pwl_root, args.sm_q)
    inject_llama_pwl_softmax(sm_mod)

    act_mod = make_act_module(args.act, model_name_short, args.num_samples, args.segments, pwl_root, args.act_q)
    if act_mod is not None: act_mod = act_mod.to(model.device) 
    inject_act_into_llama_mlp(model, act_mod)

    # 文本生成逻辑
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    
    # 强制贪心解码 (do_sample=False)
    with torch.no_grad():
        if args.precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n[{args.precision.upper()} | Softmax: {args.softmax} | Act: {args.act}]")
    print("-" * 60)
    print(f"{generated_text}")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    main()