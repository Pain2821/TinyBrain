import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Router(nn.Module):
    
    def __init__(
        self, 
        feature_dim: int = 1280,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        softmax_temperature: float = 1.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.softmax_temperature = max(softmax_temperature, 1e-6)
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating head (will be resized as pathways added)
        self.gating_head = None
        self.num_pathways = 0
        
        # Statistics
        self.routing_history = []
    
    def add_pathway(self):
        self.num_pathways += 1
        device = next(self.feature_net.parameters()).device
        
        if self.gating_head is None:
            # First pathway
            self.gating_head = nn.Linear(self.hidden_dim, 1).to(device)
            nn.init.xavier_uniform_(self.gating_head.weight)
            nn.init.zeros_(self.gating_head.bias)
        else:
            # Expand existing head
            old_weight = self.gating_head.weight.data.clone()
            old_bias = self.gating_head.bias.data.clone()
            
            # Create new larger head
            self.gating_head = nn.Linear(self.hidden_dim, self.num_pathways).to(device)
            
            # Copy old weights
            with torch.no_grad():
                self.gating_head.weight.data[:self.num_pathways-1] = old_weight
                self.gating_head.bias.data[:self.num_pathways-1] = old_bias
                
                # Initialize new pathway
                nn.init.xavier_uniform_(self.gating_head.weight.data[-1:])
                nn.init.zeros_(self.gating_head.bias.data[-1:])
        
        print(f"  [Router] Added pathway {self.num_pathways}, total: {self.num_pathways}")

    @staticmethod
    def top_k_gates(
        gates: torch.Tensor,
        top_k: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if top_k is None or top_k <= 0 or top_k >= gates.size(1):
            indices = torch.arange(gates.size(1), device=gates.device).unsqueeze(0).expand(gates.size(0), -1)
            return gates, indices, gates

        topk_vals, topk_idx = torch.topk(gates, k=top_k, dim=1)
        topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-8)

        sparse = torch.zeros_like(gates)
        sparse.scatter_(1, topk_idx, topk_vals)
        return sparse, topk_idx, topk_vals
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_pathways == 0:
            return None # type: ignore
        
        # Process features
        h = self.feature_net(features)
        
        # Compute gating logits
        logits = self.gating_head(h) # type: ignore
        
        # Softmax to get weights
        gates = F.softmax(logits / self.softmax_temperature, dim=1)
        
        return gates
    
    def get_pathway_usage(self) -> torch.Tensor:
        if len(self.routing_history) == 0:
            return torch.zeros(self.num_pathways)

        aligned = []
        for gates in self.routing_history:
            if gates.numel() == self.num_pathways:
                aligned.append(gates)
            elif gates.numel() < self.num_pathways:
                pad = torch.zeros(self.num_pathways - gates.numel(), dtype=gates.dtype)
                aligned.append(torch.cat([gates, pad], dim=0))
            else:
                aligned.append(gates[:self.num_pathways])

        all_gates = torch.stack(aligned)
        avg_usage = all_gates.mean(dim=0)
        
        return avg_usage
    
    def record_routing(self, gates: torch.Tensor):
        self.routing_history.append(gates.mean(dim=0).detach().cpu())
        
        # Keep only recent history (memory management)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
