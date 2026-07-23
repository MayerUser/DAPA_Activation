import argparse
import json
from pathlib import Path
import warnings
import types
import tempfile
import os
import atexit
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# --- 针对 4.35.2 版本的确定性导入 ---
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers.models.swin.modeling_swin import SwinSelfAttention
from transformers.models.deit.modeling_deit import DeiTSelfAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.activations import GELUActivation, NewGELUActivation

# 显式定义需要被 Hook 拦截的 Attention 类
VIT_ATTENTION_CLASSES = (ViTSelfAttention, DeiTSelfAttention)
print(f" - Note: Patching attention classes: {[c.__name__ for c in VIT_ATTENTION_CLASSES]}")

from datasets import load_dataset
from imagenet_cache import load_imagenet_validation
from scipy.integrate import quad, IntegrationWarning
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)


# --- Configuration ---
IMG_SIZE = 224
SEQ_LENGTH = 512
BATCH_SIZE = 16

# --- MODEL MAPPING ---
MODEL_MAPPING = {
    "vit-tiny": {"path": "WinKawaks/vit-tiny-patch16-224", "type": "vision"},
    "vit-small": {"path": "WinKawaks/vit-small-patch16-224", "type": "vision"},
    "vit-base": {"path": "google/vit-base-patch16-224", "type": "vision"},

    "deit-tiny": {"path": "facebook/deit-tiny-distilled-patch16-224", "type": "vision"},
    "deit-small": {"path": "facebook/deit-small-distilled-patch16-224", "type": "vision"},
    "deit-base": {"path": "facebook/deit-base-distilled-patch16-224", "type": "vision"},

    "swin-small": {"path": "microsoft/swin-small-patch4-window7-224", "type": "vision"},
    "swin-base": {"path": "microsoft/swin-base-patch4-window7-224", "type": "vision"},

    "gpt2": {"path": "openai-community/gpt2", "type": "language"},
    "bert-base": {"path": "google-bert/bert-base-uncased", "type": "language"},
}

# --- Global dictionary for file-based caching ---
print(" - Initializing temporary directory for activation caching...")
temp_dir = tempfile.mkdtemp()
captured_data = {
    "layernorm_var": {"files": [], "min": np.inf, "max": -np.inf},
    "softmax_input": {"files": [], "min": np.inf, "max": -np.inf},
    "gelu_input": {"files": [], "min": np.inf, "max": -np.inf},
}

def cleanup_temp_dir():
    print(f"\n - Cleaning up temporary directory: {temp_dir}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
atexit.register(cleanup_temp_dir)


# --- Hook Functions for Data Capture to DISK ---
def save_hook_data(key: str, data_tensor: torch.Tensor):
    if data_tensor.numel() == 0:
        return
    data = data_tensor.detach().cpu().numpy().flatten()
    current_min = np.min(data)
    current_max = np.max(data)
    if current_min < captured_data[key]["min"]:
        captured_data[key]["min"] = current_min
    if current_max > captured_data[key]["max"]:
        captured_data[key]["max"] = current_max
    temp_path = os.path.join(temp_dir, f"{key}_{len(captured_data[key]['files'])}.npy")
    np.save(temp_path, data)
    captured_data[key]["files"].append(temp_path)

def get_layernorm_var_hook(module, input_tensor, output_tensor):
    x = input_tensor[0]
    var = torch.var(x, dim=-1, unbiased=False)
    save_hook_data("layernorm_var", var)

def get_activation_input_hook(module, input_tensor, output_tensor):
    save_hook_data("gelu_input", input_tensor[0])


# --- Data Loading and Model Preparation ---
def get_model_and_attach_hooks(model_name: str, device: torch.device):
    print(f"Loading model '{model_name}'...")
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Invalid model name. Choose from {list(MODEL_MAPPING.keys())}")

    model_config = MODEL_MAPPING[model_name]
    model_path = model_config["path"]
    model_type = model_config["type"]

    print(f"Loading from Hugging Face Hub: {model_path} ({model_type} model)")

    if model_type == "vision":
        print(" - Using AutoModelForImageClassification")
        model = AutoModelForImageClassification.from_pretrained(model_path)
    elif model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif "bert" in model_name:
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model type for {model_name}")

    model.to(device)
    model.eval()

    print("Attaching hooks and modifying layers...")
    ln_hooks, sm_modified, gelu_hooks = 0, 0, 0

    import math
    def vit_new_attention_forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        max_vals = torch.max(attention_scores, dim=-1, keepdim=True)[0]
        shifted_scores = attention_scores - max_vals
        save_hook_data("softmax_input", shifted_scores)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if hasattr(self, 'dropout'):
            attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def patched_gpt2_forward(
        self, hidden_states, past_key_values=None, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False,
        output_attentions=False, cache_position=None, **kwargs
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        def split_heads(tensor, num_heads, attn_head_size):
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)
        query = split_heads(query, self.num_heads, self.head_dim)
        key = split_heads(key, self.num_heads, self.head_dim)
        value = split_heads(value, self.num_heads, self.head_dim)
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True: present = (key, value)
        else: present = None
        if hasattr(self, 'reorder_and_upcast_attn') and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask, cache_position=cache_position
            )
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            if self.scale_attn_weights:
                 attn_weights = attn_weights / torch.sqrt(
                     torch.tensor(value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device)
                 )
            if not self.is_cross_attention:
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(attn_weights.dtype)
                attn_weights = attn_weights + causal_mask
            if attention_mask is not None:
                 attn_weights = attn_weights + attention_mask
            max_vals = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            shifted_scores = attn_weights - max_vals
            save_hook_data("softmax_input", shifted_scores)
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            if head_mask is not None:
                 attn_probs = attn_probs * head_mask
            attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.num_heads * self.head_dim,)
        attn_output = attn_output.view(new_shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.register_forward_hook(get_layernorm_var_hook)
            ln_hooks += 1

        if isinstance(module, (SwinSelfAttention, nn.Softmax)) and "attention" in name:
            def shifting_softmax_hook(module, input_tensor, output_tensor):
                scores = input_tensor[0]
                max_vals = torch.max(scores, dim=-1, keepdim=True)[0]
                shifted_scores = scores - max_vals
                save_hook_data("softmax_input", shifted_scores)
            module.register_forward_hook(shifting_softmax_hook)
            sm_modified += 1
        elif isinstance(module, VIT_ATTENTION_CLASSES):
            module.forward = types.MethodType(vit_new_attention_forward, module)
            sm_modified += 1
        elif isinstance(module, GPT2Attention):
            module.forward = types.MethodType(patched_gpt2_forward, module)
            sm_modified += 1

        if isinstance(module, (nn.GELU, GELUActivation, NewGELUActivation, nn.SiLU)):
            module.register_forward_hook(get_activation_input_hook)
            gelu_hooks += 1

    print(f"✅ Attached {ln_hooks} LayerNorm hooks.")
    print(f"✅ Modified/Hooked {sm_modified} attention/softmax layers.")
    print(f"✅ Attached {gelu_hooks} GELU/SiLU/Activation hooks.")
    if sm_modified == 0:
        print("\n⚠️ Warning: Could not find any compatible attention/softmax layers.")

    return model

def get_data_loader(num_samples: int, batch_size: int, model_name: str, cache_dir: str = None):
    model_config = MODEL_MAPPING[model_name]
    model_path = model_config["path"]
    model_type = model_config["type"]

    if model_type == "vision":
        image_processor = AutoImageProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        image_mean = image_processor.image_mean
        image_std = image_processor.image_std

        crop_size = 384 if "384" in model_path else 224
        resize_size = int(crop_size / 0.875)

        print(f" - Applying transforms: Resize({resize_size}) -> CenterCrop({crop_size})")

        eval_transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

        print(f"Loading {num_samples} samples from local 'ILSVRC/imagenet-1k' cache...")
        dataset = load_imagenet_validation(num_samples, cache_dir=cache_dir)

        # --- 修改处：使用实时 Transform 绕过底层格式化 Bug ---
        def apply_transformations(examples):
            # 直接输出包含 PyTorch Tensor 的列表，不使用 torch.stack
            examples["pixel_values"] = [eval_transform(image.convert("RGB")) for image in examples["image"]]
            return examples

        # 使用 with_transform 替代 .map() 和 .set_format()
        transformed_dataset = dataset.with_transform(apply_transformations)

        def collate_fn(batch):
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            labels = torch.tensor([item.get('label', -1) for item in batch])
            return {"pixel_values": pixel_values, "labels": labels}

        return DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    elif model_type == "language":
        print(f"Loading samples from WikiText-2...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)

        def tokenize_function(examples):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer(examples["text"], truncation=False)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        if num_samples < len(tokenized_dataset):
             print(f"Selecting first {num_samples} documents...")
             tokenized_dataset = tokenized_dataset.select(range(num_samples))

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            if total_length == 0:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            if total_length < SEQ_LENGTH:
                print(f"Warning: Total tokens ({total_length}) is less than sequence length ({SEQ_LENGTH}). Padding...")
                remainder = SEQ_LENGTH - total_length
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                concatenated['input_ids'].extend([pad_token_id] * remainder)
                if 'attention_mask' in concatenated:
                     concatenated['attention_mask'].extend([0] * remainder)
                else:
                     concatenated['attention_mask'] = [1] * (total_length) + [0] * remainder
                total_length = len(concatenated['input_ids'])
            total_length = (total_length // SEQ_LENGTH) * SEQ_LENGTH
            result = {k: [t[i:i+SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)] for k, t in concatenated.items()}
            result["labels"] = result["input_ids"].copy()
            return result

        processed_dataset = tokenized_dataset.map(group_texts, batched=True)
        processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        def collate_fn(batch):
             input_ids = torch.stack([item['input_ids'] for item in batch])
             attention_mask = torch.stack([item['attention_mask'] for item in batch])
             labels = torch.stack([item['labels'] for item in batch])
             if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
             return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        return DataLoader(processed_dataset, batch_size=batch_size, collate_fn=collate_fn)


# --- PDF, PWL, and Plotting ---
def get_pdf_from_files(key: str, bins=131072):
    file_list = captured_data[key]["files"]
    if not file_list:
        print("  - WARNING: No data files found.")
        return None, None
    global_min = captured_data[key]["min"]
    global_max = captured_data[key]["max"]
    global_min -= 1e-6
    global_max += 1e-6
    if global_max <= global_min:
        print("  - WARNING: PDF data is constant or invalid range.")
        return None, None
    print(f"  - Generating high-accuracy PDF with {bins} bins from {len(file_list)} temp files...")
    print(f"  - Global range: [{global_min:.4f}, {global_max:.4f}]")
    global_bins = np.linspace(global_min, global_max, bins + 1)
    global_hist = np.zeros(bins, dtype=np.int64)
    total_count = 0
    for f in tqdm(file_list, desc=f"Processing {key} files"):
        try:
            batch_data = np.load(f)
            batch_hist, _ = np.histogram(batch_data, bins=global_bins)
            global_hist += batch_hist
            total_count += len(batch_data)
            os.remove(f)
        except Exception as e:
            print(f"\nWarning: Could not process file {f}. Error: {e}")
    bin_width = global_bins[1] - global_bins[0]
    if total_count == 0 or bin_width == 0:
        print("  - WARNING: No data processed, cannot normalize histogram.")
        return None, None
    density_hist = global_hist / (total_count * bin_width)
    captured_data[key]["files"] = []
    return density_hist, global_bins

def create_pdf_func(hist, bin_edges):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    def pdf_func(x):
        return np.interp(x, bin_centers, hist, left=0, right=0)
    return pdf_func

def plot_pdf(hist: np.ndarray, bin_edges: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, hist, linewidth=2)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PDF plot to {save_path}")

def get_uniform_segment_points_from_hist(hist, bin_edges, num_segments: int):
    print("  - Calculating percentiles from histogram...")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cumulative_hist = np.cumsum(hist)
    if cumulative_hist[-1] == 0:
        print("  - WARNING: Empty histogram, cannot determine segment points.")
        return np.linspace(bin_edges[0], bin_edges[-1], num_segments + 1)
    normalized_cumulative_hist = cumulative_hist / cumulative_hist[-1]
    percentiles_to_find = np.linspace(0, 100, num_segments + 1)
    segment_points = np.interp(
        percentiles_to_find / 100.0,
        normalized_cumulative_hist,
        bin_centers
    )
    return segment_points

def solve_poly_system(I, J, degree):
    try:
        if degree == 1:
            A = np.array([[I[2], I[1]], [I[1], I[0]]])
            B = np.array([J[1], J[0]])
            p1, p0 = np.linalg.solve(A, B)
            return {"p1": p1, "p0": p0}
        elif degree == 2:
            A = np.array([[I[4], I[3], I[2]], [I[3], I[2], I[1]], [I[2], I[1], I[0]]])
            B = np.array([J[2], J[1], J[0]])
            p2, p1, p0 = np.linalg.solve(A, B)
            return {"p2": p2, "p1": p1, "p0": p0}
        elif degree == 3:
            A = np.array([[I[6], I[5], I[4], I[3]], [I[5], I[4], I[3], I[2]],
                          [I[4], I[3], I[2], I[1]], [I[3], I[2], I[1], I[0]]])
            B = np.array([J[3], J[2], J[1], J[0]])
            p3, p2, p1, p0 = np.linalg.solve(A, B)
            return {"p3": p3, "p2": p2, "p1": p1, "p0": p0}
    except np.linalg.LinAlgError:
        return None

def fast_fit_poly(func, x_min, x_max, hist, bin_edges, degree, loss: str = "dwmse"):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    start_idx = np.searchsorted(bin_centers, x_min, side='left')
    end_idx = np.searchsorted(bin_centers, x_max, side='right')

    if start_idx >= end_idx:
        if degree == 1: return fit_linear_segment_unweighted(func, x_min, x_max)
        if degree == 2: return fit_quadratic_segment_unweighted(func, x_min, x_max)
        if degree == 3: return fit_cubic_segment_unweighted(func, x_min, x_max)

    segment_centers = bin_centers[start_idx:end_idx]
    segment_hist = hist[start_idx:end_idx]
    segment_func_vals = func(segment_centers)

    max_k = 2 * degree + 1
    x_powers = np.vander(segment_centers, max_k, increasing=True).T

    if loss == "dwmse":
        weighted_x_powers = x_powers * segment_hist * bin_width
    else:
        weighted_x_powers = x_powers * bin_width

    I = np.sum(weighted_x_powers, axis=1)
    J = np.sum(weighted_x_powers[:degree+1] * segment_func_vals, axis=1)

    params = solve_poly_system(I, J, degree)
    if params is None:
        if degree == 1: return fit_linear_segment_unweighted(func, x_min, x_max)
        if degree == 2: return fit_quadratic_segment_unweighted(func, x_min, x_max)
        if degree == 3: return fit_cubic_segment_unweighted(func, x_min, x_max)
    return params

def fit_linear_segment_unweighted(func, x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    y = np.array([func(val) for val in x])
    p1, p0 = np.polyfit(x, y, 1)
    return {"p1": p1, "p0": p0}

def fit_quadratic_segment_unweighted(func, x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    y = np.array([func(val) for val in x])
    p2, p1, p0 = np.polyfit(x, y, 2)
    return {"p2": p2, "p1": p1, "p0": p0}

def fit_cubic_segment_unweighted(func, x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    y = np.array([func(val) for val in x])
    p3, p2, p1, p0 = np.polyfit(x, y, 3)
    return {"p3": p3, "p2": p2, "p1": p1, "p0": p0}

def generate_and_save_pwl(fit_function, target_func, segment_points, hist, bin_edges, save_path: Path):
    pwl_data = {"intervals": [], "params": []}
    for i in range(len(segment_points) - 1):
        x_min, x_max = segment_points[i], segment_points[i+1]
        if x_min >= x_max:
            continue
        params = fit_function(target_func, x_min, x_max, hist, bin_edges)
        if params is None:
            continue
        start_label = "-inf" if i == 0 and segment_points[0] < 0 else f"{x_min:.6f}"
        end_label = "+inf" if i == len(segment_points) - 2 else f"{x_max:.6f}"
        pwl_data["intervals"].append((start_label, end_label))
        pwl_data["params"].append(params)
    with open(save_path, 'w') as f:
        json.dump(pwl_data, f, indent=4)
    print(f"Saved PWL data to {save_path}")
    return pwl_data

def plot_pwl_vs_original(target_func, pwl_data, segment_points, title: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    x_min_plot, x_max_plot = segment_points[0], segment_points[-1]
    if "EXP" in title and x_max_plot > 5:
        x_max_plot = 5
    if "Activation" in title:
        x_min_plot, x_max_plot = max(x_min_plot, -4), min(x_max_plot, 4)
    x_orig = np.linspace(x_min_plot, x_max_plot, 400)
    y_orig = target_func(x_orig)
    plt.plot(x_orig, y_orig, label='Original Function', color='blue', linewidth=3, zorder=5)

    for i, params in enumerate(pwl_data["params"]):
        seg_start = float(pwl_data["intervals"][i][0].replace('-inf', str(x_min_plot)))
        seg_end = float(pwl_data["intervals"][i][1].replace('+inf', str(x_max_plot)))
        x_approx = np.linspace(seg_start, seg_end, 50)
        p3, p2, p1, p0 = params.get('p3', 0), params.get('p2', 0), params.get('p1', 0), params.get('p0', 0)
        y_approx = p3*(x_approx**3) + p2*(x_approx**2) + p1*x_approx + p0
        if "EXP" in title:
            y_approx[y_approx < 0] = 0
        plt.plot(
            x_approx,
            y_approx,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Approximation' if i == 0 else "",
            zorder=10
        )

    for point in segment_points[1:-1]:
        plt.axvline(x=point, color='gray', linestyle=':', linewidth=1, zorder=1)

    plt.title(title)
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    if "SQRT" in title:
        plt.ylim(bottom=0)
    if "EXP" in title:
        plt.ylim(-1, 30)
    plt.xlim(x_min_plot, x_max_plot)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PWL comparison plot to {save_path}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate approximations for functions in transformer models.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODEL_MAPPING.keys()),
        help="The transformer model variant to use."
    )
    parser.add_argument(
        "--segment_number",
        type=int,
        required=True,
        choices=[4, 6, 8, 10, 12, 14, 16],
        help="Number of segments for the approximation."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of samples to use for capturing distribution."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to a shared Hugging Face cache directory."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dwmse",
        choices=["dwmse", "mse"],
        help="Loss metric for fitting: 'dwmse' (distribution-weighted MSE) or 'mse' (plain MSE over x)."
    )

    args = parser.parse_args()

    pdf_dir = Path(f"dst_pdf/{args.num_samples}")
    plot_dir = Path(f"dst_plot/{args.num_samples}")

    pwl_root = "dst_pwl" if args.loss == "dwmse" else "dst_pwl_mse"
    pwl_dir = Path(f"{pwl_root}/{args.num_samples}")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pwl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using loss = {args.loss}, PWL output dir = {pwl_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model_and_attach_hooks(args.model_name, device)
    data_loader = get_data_loader(args.num_samples, BATCH_SIZE, args.model_name, cache_dir=args.cache_dir)

    print("\nRunning inference to capture activation distributions...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}

            is_deit_distilled = "deit" in args.model_name

            if is_deit_distilled:
                outputs = model.deit(pixel_values=batch['pixel_values'])
                hidden_states = outputs.last_hidden_state
                logits_cls = model.cls_classifier(hidden_states[:, 0, :])
                logits_dist = model.distillation_classifier(hidden_states[:, 1, :])
                _ = (logits_cls + logits_dist) / 2  # Trigger hooks
            else:
                model(**batch)

    print("Inference complete. Processing captured / loaded PDF data...\n")

    def silu_numpy(x):
        return x / (1 + np.exp(-x))

    activation_function = silu_numpy if "bert" in args.model_name else (
        lambda x: x*0.5*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))
    )

    functions_to_approximate = {
        "exp_sm": {
            "data_key": "softmax_input",
            "func": np.exp,
            "name": "EXP (Softmax)",
            "degree": 1
        },
        "gelu_act": {
            "data_key": "gelu_input",
            "func": activation_function,
            "name": "Activation",
            "degree": 1
        }
    }

    for func_key, config in functions_to_approximate.items():
        data_key = config["data_key"]
        print(f"--- Processing for {config['name']} ({data_key}) ---")

        pdf_save_path = pdf_dir / f"pdf_{func_key}_{args.model_name}_{args.num_samples}samples.npz"

        if pdf_save_path.exists():
            print(f"  - Found existing PDF data: {pdf_save_path}, loading instead of regenerating.")
            data = np.load(pdf_save_path)
            hist = data["hist"]
            bin_edges = data["bin_edges"]
        else:
            print("  - No existing PDF found, generating from captured data...")
            hist, bin_edges = get_pdf_from_files(data_key)
            if hist is None:
                print(f"Warning: No data captured for {data_key}. Skipping.")
                continue

            global_min = captured_data[data_key]["min"]
            global_max = captured_data[data_key]["max"]

            np.savez_compressed(
                pdf_save_path,
                hist=hist,
                bin_edges=bin_edges,
                global_min=global_min,
                global_max=global_max
            )
            print(f"  - Saved new PDF data to {pdf_save_path}")

        segment_points = get_uniform_segment_points_from_hist(hist, bin_edges, args.segment_number)
        segment_points = np.unique(segment_points)

        pwl_save_path = pwl_dir / f"pwl_{func_key}_{args.model_name}_{args.segment_number}seg.json"

        pwl_data = generate_and_save_pwl(
            lambda f, x1, x2, h, be: fast_fit_poly(
                f, x1, x2, h, be, config['degree'], loss=args.loss
            ),
            config['func'],
            segment_points,
            hist,
            bin_edges,
            pwl_save_path
        )

        if pwl_data['params']:
            plot_save_path = plot_dir / (
                f"plot_pwl_vs_orig_{func_key}_{args.model_name}_{args.segment_number}seg.png"
            )
            plot_title = (
                f"Approximation vs Original {config['name']} "
                f"({args.model_name}, {args.segment_number} Segments, loss={args.loss})"
            )
            plot_pwl_vs_original(config['func'], pwl_data, segment_points, plot_title, plot_save_path)

        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()