import argparse
import types
from pathlib import Path
import math
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from imagenet_cache import load_imagenet_validation
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- 针对 4.35.2 版本的确定性导入 ---
from transformers.models.vit.modeling_vit import ViTSelfAttention
from transformers.models.swin.modeling_swin import SwinSelfAttention
from transformers.models.deit.modeling_deit import DeiTSelfAttention
from transformers.activations import GELUActivation, NewGELUActivation

# 显式定义需要被替换 forward 函数的类
ATTENTION_CLASSES_TO_PATCH = (ViTSelfAttention, DeiTSelfAttention)

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import config
from m0_udanf import PWLGelu, PWLSqrt, PWLSoftmax, DebugSoftmax
from m1_poly_act import PolyGelu

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
}

# --- Custom Modules for PWL Integration ---
class PWLLayerNorm(nn.Module):
    def __init__(self, original_layernorm: nn.LayerNorm, sqrt_json_path: str):
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
        
        self.pwl_sqrt = PWLSqrt(sqrt_json_path)
        print(f"  - Replacing LayerNorm with PWLLayerNorm using {Path(sqrt_json_path).name}")

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
    model,
    model_name,
    sqrt_impl,
    softmax_impl,
    act_impl,
    num_samples,
    debug_softmax_flag,
    loss: str
):
    print("--- Modifying model ---")

    base_pwl_dir = "dst_pwl" if loss == "dwmse" else "dst_pwl_mse"
    print(f"  - Using PWL json base dir: {base_pwl_dir}")

    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and sqrt_impl != 'torch':
            num_segments = sqrt_impl.split('-')[-1]
            json_path = f"{base_pwl_dir}/{num_samples}/pwl_sqrt_ln_{model_name}_{num_segments}seg.json"
            replacements[name] = PWLLayerNorm(module, json_path)

        if isinstance(module, (nn.GELU, GELUActivation, NewGELUActivation, nn.SiLU)) and act_impl != 'torch':
            if act_impl.startswith('pwl-'):
                num_segments = act_impl.split('-')[-1]
                json_path = f"{base_pwl_dir}/{num_samples}/pwl_gelu_act_{model_name}_{num_segments}seg.json"
                replacements[name] = PWLGelu(json_path)
                print(f"  - Queuing PWL GELU/Activation replacement for '{name}' with {Path(json_path).name}")
            elif act_impl.startswith('poly-'):
                order = act_impl.split('-')[-1]
                json_path = f"dst_poly/poly_gelu_{order}_order.json"
                replacements[name] = PolyGelu(json_path)
                print(f"  - Queuing Polynomial GELU replacement for '{name}' with {Path(json_path).name}")

    for name, new_module in replacements.items():
        name_parts = name.split('.')
        parent_name = ".".join(name_parts[:-1])
        child_name = name_parts[-1]
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, child_name, new_module)
        print(f"  - [DEBUG] Replaced '{name}' successfully.")

    if softmax_impl != 'torch':
        modified_count = 0
        
        def patched_forward_vit(self, hidden_states, head_mask=None, output_attentions=False):
            mixed_query_layer = self.query(hidden_states)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(mixed_query_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.custom_softmax(attention_scores)
            
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

            def shape(tensor):
                return tensor.view(B_, N, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)

            query = shape(self.query(hidden_states))
            key = shape(self.key(hidden_states))
            value = shape(self.value(hidden_states))

            attention_scores = (query @ key.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(attention_head_size)
            
            attention_scores = attention_scores + self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1
            ).permute(2, 0, 1).contiguous().unsqueeze(0)

            if attention_mask is not None:
                num_win = attention_mask.shape[0]
                attention_scores = attention_scores.view(
                    B_ // num_win, num_win, num_attention_heads, N, N
                ) + attention_mask.unsqueeze(1).unsqueeze(0)
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
                    seg = softmax_impl.split('-')[-1]
                    json_path = f"{base_pwl_dir}/{num_samples}/pwl_exp_sm_{model_name}_{seg}seg.json"
                    custom_softmax_module = PWLSoftmax(json_path, debug=debug_softmax_flag)
                
                module.custom_softmax = custom_softmax_module
                module.forward = types.MethodType(patch_function, module)
                modified_count += 1
                print(f"  - [DEBUG] Patched forward method of '{name}' ({type(module).__name__})")
        
        if modified_count == 0:
            print("  - WARNING: No compatible attention layers were found to patch.")

    print("--- Model modification complete ---\n")
    return model

# --- Data Loading ---
def get_data_loader(num_samples: int, batch_size: int, model_name: str, cache_dir: str = None):
    model_path = MODEL_MAPPING[model_name]["path"]
    
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

def evaluate_model(model, data_loader, device, precision, model_name: str):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    autocast_dtype = torch.float16 if precision == 'fp16' else torch.float32

    is_deit_distilled = "deit" in model_name

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({precision.upper()})"):
            images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                if is_deit_distilled:
                    outputs = model.deit(pixel_values=images)
                    hidden_states = outputs.last_hidden_state

                    logits_cls = model.cls_classifier(hidden_states[:, 0, :])
                    logits_dist = model.distillation_classifier(hidden_states[:, 1, :])

                    outputs_logits = (logits_cls + logits_dist) / 2
                else:
                    outputs = model(pixel_values=images)
                    outputs_logits = outputs.logits
                
            predictions = torch.argmax(outputs_logits, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return 100 * correct / total

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Test Vision Transformer models with PWL functions.")
    
    all_models = list(MODEL_MAPPING.keys())
    pwl_choices = ['torch', 'pwl-4', 'pwl-6', 'pwl-8', 'pwl-10', 'pwl-12', 'pwl-14', 'pwl-16']
    poly_choices = ['poly-4', 'poly-5', 'poly-6', 'poly-7', 'poly-8']
    act_choices = pwl_choices + poly_choices
    softmax_choices = ['debug'] + pwl_choices

    parser.add_argument("--model_name", type=str, required=True, choices=all_models)
    parser.add_argument("--sqrt", type=str, default='torch', choices=pwl_choices)
    parser.add_argument("--softmax", type=str, default='torch', choices=softmax_choices)
    parser.add_argument("--act", type=str, default='torch', choices=act_choices)
    parser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'fp16'])
    parser.add_argument("--debug_softmax", action="store_true")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument(
        "--pwl_samples",
        type=int,
        default=None,
        help="Number of samples used to generate the PWL files. Defaults to --num_samples.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to a shared Hugging Face cache directory.")
    parser.add_argument(
        "--loss",
        type=str,
        default="dwmse",
        choices=["dwmse", "mse"],
        help='Which approximations to use: "dwmse" -> dst_pwl, "mse" -> dst_pwl_mse.'
    )

    args = parser.parse_args()
    pwl_samples = args.pwl_samples if args.pwl_samples is not None else args.num_samples
    uses_pwl = any(impl.startswith("pwl-") for impl in (args.sqrt, args.softmax, args.act))

    # --- 硬件状态强警示与自动判定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*40)
    print(f"[*] 硬件检测状态: 当前分配的设备是 {device.type.upper()}")
    if device.type == "cpu":
        print("    -> ⚠️ 警告: 未检测到可用 GPU，PyTorch 将完全在 CPU 上执行！")
        print("    -> 提示: 请检查是否正确申请了 GPU 资源或 CUDA 环境变量是否正确配置。")
    print("="*40 + "\n")

    print("--- Test Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Precision: {args.precision.upper()}")
    print(f"SQRT: {args.sqrt}, Softmax: {args.softmax}, Activation: {args.act}")
    print(f"Loss (PWL source): {args.loss}  (dst_pwl if dwmse, dst_pwl_mse if mse)")
    if uses_pwl:
        print(f"PWL samples: {pwl_samples}")
    if args.cache_dir:
        print(f"Using shared cache directory: {args.cache_dir}")
    print(f"Test samples: {args.num_samples}, Batch size: {config.BATCH_SIZE}\n")

    torch_dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    
    # --- 第一时间把模型放进 GPU ---
    print(f" - Loading AutoModelForImageClassification directly to {device.type.upper()}...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_MAPPING[args.model_name]["path"], 
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    ).to(device)

    model = replace_modules_in_model(
        model,
        args.model_name,
        args.sqrt,
        args.softmax,
        args.act,
        pwl_samples,
        args.debug_softmax,
        args.loss
    )
    
    eval_data_loader = get_data_loader(
        args.num_samples,
        config.BATCH_SIZE,
        args.model_name,
        cache_dir=args.cache_dir
    )

    accuracy = evaluate_model(model, eval_data_loader, device, args.precision, args.model_name)
    
    print("\n--- Results ---")
    print(f"Top-1 Accuracy on {args.num_samples} samples ({args.precision.upper()}): {accuracy:.2f}%")

if __name__ == "__main__":
    main()
