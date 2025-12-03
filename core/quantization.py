import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple


def quantize_to_ternary(
    model: nn.Module, 
    threshold_percentile: float = 0.70,
    layer_wise: bool = True
) -> nn.Module:
    
    print(f"[QUANTIZATION] Starting ternary quantization (1.58-bit)...")
    print(f"  Threshold: {threshold_percentile:.0%} (keep top {(1-threshold_percentile)*100:.0f}%)")
    
    quantized_model = copy.deepcopy(model)
    
    stats = {
        'total_params': 0,
        'non_zero_params': 0,
        'positive_params': 0,
        'negative_params': 0,
        'layer_stats': {}
    }
    
    for name, param in quantized_model.named_parameters():
        if 'weight' not in name or len(param.shape) < 2:
            continue  # Skip biases and 1D layers
        
        w = param.data
        stats['total_params'] += w.numel()
        
        if layer_wise:
            # Per-layer threshold
            threshold = torch.quantile(torch.abs(w.flatten()), threshold_percentile)
        else:
            # Global threshold (computed once, used here)
            threshold = threshold_percentile  # Assumes pre-computed
        
        # Create ternary weights
        ternary = torch.zeros_like(w)
        ternary[w > threshold] = 1.0
        ternary[w < -threshold] = -1.0
        
        # Update parameter
        param.data = ternary
        
        # Statistics
        non_zero = (ternary != 0).sum().item()
        positive = (ternary == 1).sum().item()
        negative = (ternary == -1).sum().item()
        sparsity = 1.0 - (non_zero / w.numel())
        
        stats['non_zero_params'] += non_zero
        stats['positive_params'] += positive
        stats['negative_params'] += negative
        
        stats['layer_stats'][name] = {
            'sparsity': sparsity,
            'positive_ratio': positive / w.numel(),
            'negative_ratio': negative / w.numel(),
            'params': w.numel()
        }
        
        print(f"    {name:40s}: {sparsity*100:5.1f}% sparse, "
              f"{positive:7d} pos, {negative:7d} neg")
    
    # Overall stats
    overall_sparsity = 1.0 - (stats['non_zero_params'] / stats['total_params'])
    print(f"\n  Overall sparsity: {overall_sparsity*100:.1f}%")
    print(f"  Positive weights: {stats['positive_params']:,}")
    print(f"  Negative weights: {stats['negative_params']:,}")
    print(f"  Zero weights: {stats['total_params'] - stats['non_zero_params']:,}")
    
    return quantized_model, stats # pyright: ignore[reportReturnType]


def estimate_ternary_size(model: nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'params': total_params,
        'fp32_mb': total_params * 4 / (1024**2),
        'fp16_mb': total_params * 2 / (1024**2),
        'int8_mb': total_params * 1 / (1024**2),
        'ternary_mb': total_params * 2 / 8 / (1024**2),  # 2 bits per param
        'ternary_ideal_mb': total_params * 1.58 / 8 / (1024**2)  # Theoretical
    }


def mixed_precision_quantization(
    model: nn.Module,
    critical_layers: list,
    threshold_percentile: float = 0.70
) -> nn.Module:
    quantized_model = copy.deepcopy(model)
    
    for name, param in quantized_model.named_parameters():
        if 'weight' not in name or len(param.shape) < 2:
            continue
        
        # Check if critical layer
        is_critical = any(crit in name for crit in critical_layers)
        
        if is_critical:
            # Keep in INT8 (or FP16)
            w = param.data
            scale = w.abs().max() / 127.0
            quantized = torch.round(w / scale * 127.0).clamp(-127, 127)
            param.data = quantized / 127.0 * scale
            print(f"  {name}: INT8 (critical)")
        else:
            # Ternary quantization
            w = param.data
            threshold = torch.quantile(torch.abs(w.flatten()), threshold_percentile)
            ternary = torch.zeros_like(w)
            ternary[w > threshold] = 1.0
            ternary[w < -threshold] = -1.0
            param.data = ternary
            sparsity = (ternary == 0).float().mean()
            print(f"  {name}: Ternary ({sparsity*100:.0f}% sparse)")
    
    return quantized_model