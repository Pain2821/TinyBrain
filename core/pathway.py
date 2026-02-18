import torch
import torch.nn as nn
from typing import Optional
from .quantization import estimate_ternary_size


class Pathway(nn.Module):
    
    def __init__(
        self, 
        base_model: nn.Module,
        task_name: str,
        task_idx: int,
        quantization_stats: Optional[dict] = None
    ):
        super().__init__()
        
        self.task_name = task_name
        self.task_idx = task_idx
        self.quantization_stats = quantization_stats or {}
        
        # Store frozen model
        self.network = base_model
        
        # Freeze all parameters
        for param in self.network.parameters():
            param.requires_grad = False
        
        # Calculate memory
        size_info = estimate_ternary_size(self.network)
        self.memory_mb = size_info['ternary_mb']
        self.num_params = size_info['params']
        
        # Performance tracking
        self.usage_count = 0
        self.accuracy_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.network(x)
    
    def get_sparsity(self) -> float:
        total = 0
        zeros = 0
        
        for param in self.network.parameters():
            total += param.numel()
            zeros += (param.data == 0).sum().item()
        
        return zeros / total if total > 0 else 0.0
    
    def record_usage(self, count: int = 1):
        self.usage_count += count
    
    def record_accuracy(self, acc: float):
        self.accuracy_history.append(acc)
    
    def get_stats(self) -> dict:
        return {
            'task_name': self.task_name,
            'task_idx': self.task_idx,
            'memory_mb': self.memory_mb,
            'num_params': self.num_params,
            'sparsity': self.get_sparsity(),
            'usage_count': self.usage_count,
            'avg_accuracy': sum(self.accuracy_history) / len(self.accuracy_history) 
                           if self.accuracy_history else 0.0
        }
    
    def __repr__(self):
        return (f"Pathway(task={self.task_name}, "
                f"memory={self.memory_mb:.2f}MB, "
                f"sparsity={self.get_sparsity()*100:.1f}%)")
