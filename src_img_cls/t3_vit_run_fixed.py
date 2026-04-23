# t3_vit_run_fixed.py

import argparse
import types
from pathlib import Path
import math
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
# --- MODIFICATION: Make imports backward-compatible ---
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers.models.swin.modeling_swin import SwinSelfAttention
from transformers.models.deit.modeling_deit import DeiTSelfAttention  # <--- ADDED IMPORT
from transformers.activations import GELUActivation, NewGELUActivation

# Dynamically import ViTSdpaSelfAttention if it exists in the installed transformers version
try:
    from transformers.models.vit.modeling_vit import ViTSdpaSelfAttention
    # --- ADDED DeiTSelfAttention TO TUPLE ---
    ATTENTION_CLASSES_TO_PATCH = (ViTSelfAttention, ViTSdpaSelfAttention, DeiTSelfAttention)
    print(" - Note: Found 'ViTSdpaSelfAttention', supporting modern ViT/DeiT architectures.")
except ImportError:
    print(" - Note: 'ViTSdpaSelfAttention' not found. Supporting older ViT/DeiT architectures.")
    ViTSdpaSelfAttention = None # Define as None if it doesn't exist
    # --- ADDED DeiTSelfAttention TO TUPLE ---
    ATTENTION_CLASSES_TO_PATCH = (ViTSelfAttention, DeiTSelfAttention)
# --- END MODIFICATION ---


from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Import PWL function implementations and config parameters
import config
# --- MODIFICATION: Import fixed-point modules ---
from m3_udanf_fixed import PWLGeluFixed, PWLSqrtFixed, PWLSoftmaxFixed, DebugSoftmax
# --- END MODIFICATION ---


# --- MODEL MAPPING (REVISED) ---
MODEL_MAPPING = {
    # Vision Models
    "vit-tiny": {"path": "WinKawaks/vit-tiny-patch16-224", "type": "vision"},
    "vit-small": {"path": "WinKawaks/vit-small-patch16-224", "type": "vision"},
    "vit-base": {"path": "google/vit-base-patch16-224", "type": "vision"},
    
    # --- Use the distilled models ---
    "deit-tiny": {"path": "facebook/deit-tiny-distilled-patch16-224", "type": "vision"},
    "deit-small": {"path": "facebook/deit-small-distilled-patch16-224", "type": "vision"},
    "deit-base": {"path": "facebook/deit-base-distilled-patch16-224", "type": "vision"}, 
    
    "swin-small": {"path": "microsoft/swin-small-patch4-window7-224", "type": "vision"},
    "swin-base": {"path": "microsoft/swin-base-patch4-window7-224", "type": "vision"},
}
# --- END REVISION ---


# --- Custom Modules for FIXED-POINT PWL Integration ---

class PWLLayerNormFixed(nn.Module):
    def __init__(self, original_layernorm: nn.LayerNorm, sqrt_json_path: str,
                 i_bits_x: int, f_bits_x: int, i_bits_y: int, f_bits_y: int):
        super().__init__()
        self.normalized_shape = original_layernorm.normalized_shape
        self.eps = original_layernorm.eps
        self.elementwise_affine = original_layernorm.elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(original_layernorm.weight.clone())
            self.bias = nn.Parameter(original_layernorm.bias.clone())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.pwl_sqrt = PWLSqrtFixed(
            sqrt_json_path, 
            i_bits_x, f_bits_x, i_bits_y, f_bits_y
        )
        print(f"  - Replacing LayerNorm with PWLLayerNormFixed using {Path(sqrt_json_path).name}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        mean = x_float.mean(dim=-1, keepdim=True)
        var = torch.var(x_float, dim=-1, keepdim=True, unbiased=False)
        inv_std = 1.0 / self.pwl_sqrt(var + self.eps)
        normalized_x = (x_float - mean) * inv_std
        if self.elementwise_affine:
            return (normalized_x * self.weight.to(torch.float32) + self.bias.to(torch.float32)).to(input_dtype)
        return normalized_x.to(input_dtype)

# --- Model Modification Logic ---

def replace_modules_in_model(
    model, model_name, 
    sqrt_impl, softmax_impl, act_impl, 
    num_samples, debug_softmax_flag,
    act_bits_config, sqrt_bits_config, sm_bits_config
):
    print("--- Modifying model ---")
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and sqrt_impl != 'torch':
            if not sqrt_bits_config:
                raise ValueError("--sqrt_q is required when --sqrt is not 'torch'")
            num_segments = sqrt_impl.split('-')[-1]
            json_path = f"dst_pwl/{num_samples}/pwl_sqrt_ln_{model_name}_{num_segments}seg.json"
            replacements[name] = PWLLayerNormFixed(module, json_path, **sqrt_bits_config)
        if isinstance(module, (nn.GELU, GELUActivation, NewGELUActivation, nn.SiLU)) and act_impl != 'torch':
            if not act_bits_config:
                raise ValueError("--act_q is required when --act is not 'torch'")
            if act_impl.startswith('pwl-'):
                num_segments = act_impl.split('-')[-1]
                json_path = f"dst_pwl/{num_samples}/pwl_gelu_act_{model_name}_{num_segments}seg.json"
                replacements[name] = PWLGeluFixed(json_path, **act_bits_config)
                print(f"  - Queuing FIXED-POINT PWL GELU for '{name}' with {Path(json_path).name}")
    for name, new_module in replacements.items():
        name_parts = name.split('.')
        parent_name = ".".join(name_parts[:-1])
        child_name = name_parts[-1]
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, child_name, new_module)
        print(f"  - [DEBUG] Replaced '{name}' successfully.")

    if softmax_impl != 'torch':
        if not sm_bits_config:
            raise ValueError("--sm_q is required when --softmax is not 'torch'")
        modified_count = 0
        
        # --- (This is the minimal, correct patch from before) ---
        def patched_forward_vit(self, hidden_states, head_mask=None, output_attentions=False):
            mixed_query_layer = self.query(hidden_states)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(mixed_query_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.custom_softmax(attention_scores) # <-- OUR CHANGE
            if hasattr(self, 'dropout'):
                attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs

        def patched_forward_swin(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            B_, N, C = hidden_states.shape
            num_attention_heads = self.num_attention_heads
            attention_head_size = int(C / num_attention_heads)
            def shape(tensor): return tensor.view(B_, N, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            query, key, value = shape(self.query(hidden_states)), shape(self.key(hidden_states)), shape(self.value(hidden_states))
            attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(attention_head_size)
            attention_scores = attention_scores + self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            ).permute(2, 0, 1).contiguous().unsqueeze(0)
            if attention_mask is not None:
                num_win = attention_mask.shape[0]
                attention_scores = attention_scores.view(B_ // num_win, num_win, num_attention_heads, N, N) + attention_mask.unsqueeze(1).unsqueeze(0)
                attention_scores = attention_scores.view(-1, num_attention_heads, N, N)
            attention_probs = self.custom_softmax(attention_scores)
            attention_probs = self.dropout(attention_probs)
            context_layer = (attention_probs @ value).transpose(1, 2).reshape(B_, N, C)
            return (context_layer, attention_probs) if output_attentions else (context_layer,)

        for name, module in model.named_modules():
            patch_function = None
            if isinstance(module, ATTENTION_CLASSES_TO_PATCH):
                patch_function = patched_forward_vit
            elif isinstance(module, SwinSelfAttention):
                patch_function = patched_forward_swin
            if patch_function:
                if softmax_impl == 'debug':
                    custom_softmax_module = DebugSoftmax()
                else:
                    json_path = f"dst_pwl/{num_samples}/pwl_exp_sm_{model_name}_{softmax_impl.split('-')[-1]}seg.json"
                    custom_softmax_module = PWLSoftmaxFixed(
                        json_path, **sm_bits_config, debug=debug_softmax_flag
                    )
                module.custom_softmax = custom_softmax_module
                module.forward = types.MethodType(patch_function, module)
                modified_count += 1
                print(f"  - [DEBUG] Patched forward method of '{name}' ({type(module).__name__})")
        if modified_count == 0:
            print("  - WARNING: No compatible attention layers were found to patch.")
    print("--- Model modification complete ---\n")
    return model

# --- Data Loading and Evaluation ---

def get_data_loader(num_samples: int, batch_size: int, model_name: str, cache_dir: str = None):
    model_path = MODEL_MAPPING[model_name]["path"]
    
    # --- CORRECTED TRANSFORMS (from DeiT paper) ---
    image_processor = AutoImageProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std
    
    # Use 384 crop for deit-base-384, 224 for all others
    crop_size = 384 if "384" in model_path else 224
    resize_size = int(crop_size / 0.875) # This is 256 for 224, 438 for 384
    
    print(f" - Applying transforms: Resize({resize_size}) -> CenterCrop({crop_size})")

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
    # --- END CORRECTION ---

    print(f"Loading {num_samples} samples from 'ILSVRC/imagenet-1k' (streaming)...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation', streaming=True, cache_dir=cache_dir).take(num_samples)

    def apply_transformations(examples):
        processed_images = [eval_transform(image.convert("RGB")) for image in examples["image"]]
        examples["pixel_values"] = torch.stack(processed_images)
        return examples

    transformed_dataset = dataset.map(apply_transformations, batched=True, remove_columns=["image"])

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item.get('label', -1) for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    return DataLoader(transformed_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# --- REVISED: evaluate_model ---
def evaluate_model(model, data_loader, device, precision, model_name: str):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    autocast_dtype = torch.float16 if precision == 'fp16' else torch.float32

    # Check if this is a DeiT distilled model
    is_deit_distilled = "deit" in model_name

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({precision.upper()})"):
            images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                
                if is_deit_distilled:
                    # --- CORRECTED DeiT-Distilled LOGIC ---
                    # 1. Get hidden states from the model's base (which is `.deit`)
                    outputs = model.deit(pixel_values=images)
                    hidden_states = outputs.last_hidden_state

                    # 2. Get logits from the TWO separate classifier heads
                    #    [CLS] token (index 0) goes to `.cls_classifier`
                    #    [DIST] token (index 1) goes to `.distillation_classifier`
                    logits_cls = model.cls_classifier(hidden_states[:, 0, :])
                    logits_dist = model.distillation_classifier(hidden_states[:, 1, :])

                    # 3. Average the logits for the final prediction
                    outputs_logits = (logits_cls + logits_dist) / 2
                    # --- END CORRECTION ---
                else:
                    # --- Standard ViT/Swin LOGIC ---
                    outputs = model(pixel_values=images)
                    outputs_logits = outputs.logits
                
            predictions = torch.argmax(outputs_logits, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100 * correct / total
# --- END REVISION ---

# --- Helper function to parse Q-format string ---

def parse_q_format(q_str: str) -> dict:
    """Parses a Q-format string like '9.4' into a dict for the module."""
    if not q_str:
        return {} # Return empty dict if no Q-format is specified
    try:
        i_str, f_str = q_str.split('.')
        i_bits = int(i_str)
        f_bits = int(f_str)
        # Assumes symmetric Q-format for X and Y
        return {"i_bits_x": i_bits, "f_bits_x": f_bits, "i_bits_y": i_bits, "f_bits_y": f_bits}
    except Exception:
        raise ValueError(f"Invalid Q-format string: '{q_str}'. Expected format 'I.F' (e.g., '9.4').")

# --- Main Execution ---

def main():
    
    pwl_choices = ['torch', 'pwl-4', 'pwl-6', 'pwl-8', 'pwl-10', 'pwl-12', 'pwl-14', 'pwl-16']
    
    parser = argparse.ArgumentParser(description="Test Vision Transformer models with FIXED-POINT PWL functions.")
    
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_MAPPING.keys())
    parser.add_argument("--num_samples", type=int, default=256, 
                        help="Number of samples used to *generate* the PWL files (e.g., 256).")
    
    # Implementation choices
    parser.add_argument("--sqrt", type=str, default='torch', choices=pwl_choices,
                        help="SQRT implementation to use.")
    parser.add_argument("--softmax", type=str, default='torch', choices=pwl_choices,
                        help="Softmax/EXP implementation to use.")
    parser.add_argument("--act", type=str, default='torch', choices=pwl_choices,
                        help="Activation/GELU implementation to use.")
    
    # New Q-Number arguments
    parser.add_argument("--sqrt_q", type=str, default=None, 
                        help="Symmetric Q-format for SQRT data (e.g., '4.12')")
    parser.add_argument("--sm_q", type=str, default=None, 
                        help="Symmetric Q-format for Softmax/EXP data (e.g., '6.10')")
    parser.add_argument("--act_q", type=str, default=None, 
                        help="Symmetric Q-format for ACT/GELU data (e.g., '9.4')")
    
    # Evaluation Parameters
    parser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'fp16'])
    parser.add_argument("--debug_softmax", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None, 
                        help="Path to a shared Hugging Face cache directory.")
    
    args = parser.parse_args()

    # --- Build Q-number configs from new args ---
    act_bits_config = parse_q_format(args.act_q)
    sm_bits_config = parse_q_format(args.sm_q)
    sqrt_bits_config = parse_q_format(args.sqrt_q)

    # --- Print Configuration ---
    print("--- FIXED-POINT Test Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Precision: {args.precision.upper()}")
    print(f"PWL files generated with {args.num_samples} samples.")
    print(f"SQRT: {args.sqrt}" + (f" (Q{args.sqrt_q})" if args.sqrt_q else ""))
    print(f"Softmax: {args.softmax}" + (f" (Q{args.sm_q})" if args.sm_q else ""))
    print(f"Activation: {args.act}" + (f" (Q{args.act_q})" if args.act_q else ""))
    if args.cache_dir:
        print(f"Using shared cache directory: {args.cache_dir}")
    print(f"Test samples: {config.SAMPLE_NUM}, Batch size: {config.BATCH_SIZE}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if args.precision == 'fp16' else torch.float32

    # --- Use AutoModel, this is stable and correct ---
    print(f" - Using AutoModelForImageClassification (standard head) for {args.model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_MAPPING[args.model_name]["path"], 
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )
    # --- END CORRECTION ---

    model = replace_modules_in_model(
        model, args.model_name, 
        args.sqrt, args.softmax, args.act, 
        args.num_samples, args.debug_softmax,
        act_bits_config, sqrt_bits_config, sm_bits_config
    )
    
    eval_data_loader = get_data_loader(config.SAMPLE_NUM, config.BATCH_SIZE, args.model_name, cache_dir=args.cache_dir)

    # --- REVISED: Pass model_name to evaluation ---
    accuracy = evaluate_model(model, eval_data_loader, device, args.precision, args.model_name)
    
    print("\n--- Results ---")
    print(f"Top-1 Accuracy on {config.SAMPLE_NUM} samples ({args.precision.upper()}): {accuracy:.2f}%")

if __name__ == "__main__":
    main()