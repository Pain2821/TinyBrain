import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
from typing import List, Tuple, Optional

from .pathway import Pathway
from .router import Router
from .quantization import quantize_to_ternary, estimate_ternary_size


class BitBrain(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        base_arch: str = 'mobilenet_v2',
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.base_arch = base_arch
        self.device = device
        self.task_history = []
        
        # Fast Learner (plastic)
        self.fast_learner = self._create_base_network(num_classes)
        
        # Pathway Bank (frozen experts)
        self.pathways = nn.ModuleList()
        
        # Router (gating)
        self.router = Router(feature_dim=self._get_feature_dim())
        
        # Move to device
        self.to(device)
        
        # Config
        self.fast_learner_weight = 0.7  # During training
        self.pathway_weight = 0.3
        
        print(f"[BitBrain] Initialized")
        print(f"  Architecture: {base_arch}")
        print(f"  Device: {device}")
        print(f"  Fast Learner size: {self._get_model_size_mb():.1f} MB")
    
    def _create_base_network(self, num_classes: int) -> nn.Module:
        if self.base_arch == 'mobilenet_v2':
            try:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
                base = models.mobilenet_v2(weights=weights)
            except:
                base = models.mobilenet_v2(pretrained=True)
            
            # Replace classifier
            base.classifier = nn.Linear(1280, num_classes) # type: ignore
            nn.init.xavier_uniform_(base.classifier.weight) # type: ignore
            nn.init.zeros_(base.classifier.bias) # type: ignore
            
            return base
        else:
            raise NotImplementedError(f"Architecture {self.base_arch} not supported")
    
    def _get_feature_dim(self) -> int:
        if self.base_arch == 'mobilenet_v2':
            return 1280
        return 1280
    
    def _get_model_size_mb(self) -> float:
        size_info = estimate_ternary_size(self.fast_learner)
        return size_info['fp32_mb']
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_arch == 'mobilenet_v2':
            features = self.fast_learner.features(x) # type: ignore
            pooled = F.adaptive_avg_pool2d(features, 1)
            return pooled.view(pooled.size(0), -1)
        
        raise NotImplementedError()
    
    # def forward(self, x: torch.Tensor, return_routing_info: bool = False):
    #     """
    #     Forward pass with proper single-pathway handling
        
    #     FIX: When only 1 pathway exists:
    #     - During training: Use it normally to train router
    #     - During testing: Rely more on Fast Learner (pathway might be degraded)
    #     """
    #     batch_size = x.size(0)
        
    #     # Extract features
    #     features = self.extract_features(x)
        
    #     # Fast Learner prediction
    #     fast_pred = self.fast_learner.classifier(features) # type: ignore
        
    #     # No pathways - use Fast Learner only
    #     if len(self.pathways) == 0:
    #         if return_routing_info:
    #             info = {
    #                 'source': 'fast_learner_only',
    #                 'gates': None,
    #                 'num_pathways': 0,
    #                 'fast_weight': 1.0,
    #                 'pathway_weight': 0.0
    #             }
    #             return fast_pred, info
    #         return fast_pred
        
    #     # Router gates
    #     gates = self.router(features)  # [batch, num_pathways]
    #     self.router.record_routing(gates)
        
    #     # Pathway predictions
    #     pathway_preds = []
    #     for pathway in self.pathways:
    #         pred = pathway(x)
    #         pathway_preds.append(pred)
        
    #     pathway_preds = torch.stack(pathway_preds, dim=1)  # [B, P, C]
        
    #     # Weighted combination
    #     pathway_output = torch.einsum('bp,bpc->bc', gates, pathway_preds)
        
    #     # FIX: Adaptive combination based on number of pathways
    #     if len(self.pathways) == 1:
    #         # Single pathway - might be degraded by quantization
    #         if self.training:
    #             # During training: use normal weights to train router
    #             fast_weight = 0.7
    #             pathway_weight = 0.3
    #         else:
    #             # During testing: trust Fast Learner more
    #             # Pathway is quantized and might be broken
    #             fast_weight = 0.9  # Trust Fast Learner heavily
    #             pathway_weight = 0.1  # Pathway might be broken
                
    #         output = fast_weight * fast_pred + pathway_weight * pathway_output
            
    #     elif len(self.pathways) <= 3:
    #         # Few pathways - gradually trust them more
    #         if self.training:
    #             fast_weight = 0.6
    #             pathway_weight = 0.4
    #         else:
    #             fast_weight = 0.5
    #             pathway_weight = 0.5
                
    #         output = fast_weight * fast_pred + pathway_weight * pathway_output
            
    #     else:
    #         # Many pathways - trust pathway routing
    #         if self.training:
    #             fast_weight = 0.5
    #             pathway_weight = 0.5
    #         else:
    #             fast_weight = 0.3  # Pathways have learned multiple tasks
    #             pathway_weight = 0.7
                
    #         output = fast_weight * fast_pred + pathway_weight * pathway_output
        
    #     if return_routing_info:
    #         info = {
    #             'gates': gates.detach().cpu(),
    #             'fast_pred': fast_pred.detach().cpu(),
    #             'pathway_pred': pathway_output.detach().cpu(),
    #             'num_pathways': len(self.pathways),
    #             'source': f'{len(self.pathways)}_pathways',
    #             'fast_weight': fast_weight,
    #             'pathway_weight': pathway_weight
    #         }
    #         return output, info
        
    #     return output

    def forward(self, x: torch.Tensor, return_routing_info: bool = False):
        """
        Forward pass with CORRECTED routing
        
        FIXES:
        1. Proper weight normalization (no double weighting)
        2. Entropy-based adaptive weighting (optional)
        3. Per-sample confidence adjustment
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.extract_features(x)
        
        # Fast Learner prediction
        fast_pred = self.fast_learner.classifier(features) # type: ignore
        
        # No pathways - use Fast Learner only
        if len(self.pathways) == 0:
            if return_routing_info:
                info = {
                    'source': 'fast_learner_only',
                    'gates': None,
                    'num_pathways': 0,
                    'fast_weight': 1.0,
                    'pathway_weight': 0.0
                }
                return fast_pred, info
            return fast_pred
        
        # Router gates (already normalized via softmax)
        gates = self.router(features)  # [B, P]
        self.router.record_routing(gates)
        
        # Pathway predictions
        pathway_preds = []
        for pathway in self.pathways:
            pred = pathway(x)
            pathway_preds.append(pred)
        
        pathway_preds = torch.stack(pathway_preds, dim=1)  # [B, P, C]
        
        # Weighted pathway output (gates already sum to 1!)
        pathway_output = torch.einsum('bp,bpc->bc', gates, pathway_preds)
        
        # ==== ADAPTIVE WEIGHTING STRATEGY ====
        
        # Base weights by pathway count
        if len(self.pathways) == 1:
            base_fast = 0.7
            base_pathway = 0.3
        elif len(self.pathways) <= 3:
            base_fast = 0.5
            base_pathway = 0.5
        else:
            base_fast = 0.3
            base_pathway = 0.7
        
        # Optional: Adjust by training/testing
        if not self.training:
            # During testing, might want different behavior
            if len(self.pathways) == 1:
                base_fast = 0.8  # Single pathway might be degraded
                base_pathway = 0.2
        
        # Optional: Entropy-based adjustment
        USE_ENTROPY = True  # Set this as config parameter
        
        if USE_ENTROPY:
            # Calculate routing confidence (inverse of entropy)
            epsilon = 1e-8
            entropy = -(gates * torch.log(gates + epsilon)).sum(dim=1)  # [B]
            max_entropy = torch.log(torch.tensor(len(self.pathways), dtype=torch.float32, device=gates.device))
            normalized_entropy = entropy / max_entropy  # [B] in [0, 1]
            
            # Adjust weights per sample
            # High entropy (uncertain) → trust Fast Learner more
            # Low entropy (confident) → trust Pathways more
            entropy_adj = normalized_entropy.unsqueeze(1)  # [B, 1]
            
            fast_weight = base_fast + 0.15 * entropy_adj
            pathway_weight = base_pathway - 0.15 * entropy_adj
        else:
            # Static weights
            fast_weight = torch.tensor([[base_fast]], device=x.device).expand(batch_size, 1)
            pathway_weight = torch.tensor([[base_pathway]], device=x.device).expand(batch_size, 1)
        
        # CRITICAL: Normalize weights to sum to 1
        total_weight = fast_weight + pathway_weight
        fast_weight = fast_weight / total_weight
        pathway_weight = pathway_weight / total_weight
        
        # Final prediction with proper weighting
        output = fast_weight * fast_pred + pathway_weight * pathway_output
        
        if return_routing_info:
            info = {
                'gates': gates.detach().cpu(),
                'fast_pred': fast_pred.detach().cpu(),
                'pathway_pred': pathway_output.detach().cpu(),
                'num_pathways': len(self.pathways),
                'source': f'{len(self.pathways)}_pathways',
                'fast_weight_mean': fast_weight.mean().item(),
                'pathway_weight_mean': pathway_weight.mean().item(),
                'fast_weight_std': fast_weight.std().item() if USE_ENTROPY else 0.0,
                'pathway_weight_std': pathway_weight.std().item() if USE_ENTROPY else 0.0,
            }
            
            if USE_ENTROPY:
                info['entropy_mean'] = normalized_entropy.mean().item()
                info['entropy_std'] = normalized_entropy.std().item()
            
            return output, info
        
        return output

    # def consolidate_task(
    #     self,
    #     task_name: str,
    #     threshold_percentile: float = 0.70,
    #     quantize: bool = True
    # ):
    #     print(f"\n{'='*60}")
    #     print(f"[CONSOLIDATION] Task: {task_name}")
    #     print(f"{'='*60}\n")
        
    #     # Copy Fast Learner
    #     pathway_network = copy.deepcopy(self.fast_learner)
        
    #     # Quantize to ternary
    #     if quantize:
    #         pathway_network, quant_stats = quantize_to_ternary(
    #             pathway_network, 
    #             threshold_percentile
    #         ) # type: ignore
    #     else:
    #         print("  [WARNING] Quantization disabled - using FP32")
    #         quant_stats = {}
        
    #     # Create Pathway
    #     task_idx = len(self.pathways)
    #     pathway = Pathway(
    #         pathway_network,
    #         task_name,
    #         task_idx,
    #         quant_stats
    #     )
        
    #     # Add to bank
    #     self.pathways.append(pathway)
        
    #     # Update router
    #     self.router.add_pathway()
        
    #     # Record task
    #     self.task_history.append(task_name)
        
    #     print(f"\n  ✓ Pathway created: {pathway}")
    #     print(f"  ✓ Total pathways: {len(self.pathways)}")
    #     print(f"  ✓ Total memory: {self.get_total_memory():.2f} MB\n")
    #     print(f"{'='*60}\n")

    def consolidate_task(
        self,
        task_name: str,
        threshold_percentile: float = 0.70,
        quantize: bool = True,
        skip_classifier: bool = True  # NEW: Don't quantize classifier
    ):
        """
        Consolidate with option to preserve classifier
        """
        print(f"\n{'='*60}")
        print(f"[CONSOLIDATION] Task: {task_name}")
        print(f"{'='*60}\n")
        
        # Copy Fast Learner
        pathway_network = copy.deepcopy(self.fast_learner)
        
        # Quantize to ternary
        if quantize:
            # Use FIXED quantization that preserves classifier
            pathway_network, quant_stats = quantize_to_ternary(
                pathway_network, 
                threshold_percentile,
                skip_classifier=skip_classifier  # NEW
            )
        else:
            print("  [WARNING] Quantization disabled - using FP32")
            quant_stats = {}
        
        # Create Pathway
        task_idx = len(self.pathways)
        pathway = Pathway(
            pathway_network,
            task_name,
            task_idx,
            quant_stats
        )
        
        # Add to bank
        self.pathways.append(pathway)
        
        # Update router
        self.router.add_pathway()
        
        # Record task
        self.task_history.append(task_name)
        
        print(f"\n  ✓ Pathway created: {pathway}")
        print(f"  ✓ Total pathways: {len(self.pathways)}")
        print(f"  ✓ Total memory: {self.get_total_memory():.2f} MB\n")
        print(f"{'='*60}\n")


    def reset_fast_learner(
        self,
        new_num_classes: Optional[int] = None,
        keep_backbone: bool = True
    ):
        if new_num_classes is None:
            new_num_classes = self.num_classes
        
        print(f"[RESET] Fast Learner for new task ({new_num_classes} classes)")
        
        if keep_backbone:
            # Reset classifier only
            if self.base_arch == 'mobilenet_v2':
                self.fast_learner.classifier = nn.Linear(
                    1280, new_num_classes
                ).to(self.device)
                nn.init.xavier_uniform_(self.fast_learner.classifier.weight)
                nn.init.zeros_(self.fast_learner.classifier.bias)
            print(f"  ✓ Classifier reset (backbone preserved)\n")
        else:
            # Full reset
            self.fast_learner = self._create_base_network(new_num_classes)
            self.fast_learner.to(self.device)
            print(f"  ✓ Full network reset\n")
    
    def get_total_memory(self) -> float:
        fast_mb = self._get_model_size_mb()
        pathway_mb = sum(p.memory_mb for p in self.pathways) # type: ignore
        router_mb = estimate_ternary_size(self.router)['fp32_mb']
        
        return fast_mb + pathway_mb + router_mb
    
    def get_memory_breakdown(self) -> dict:
        return {
            'fast_learner_mb': self._get_model_size_mb(),
            'pathways_mb': sum(p.memory_mb for p in self.pathways), # type: ignore
            'router_mb': estimate_ternary_size(self.router)['fp32_mb'],
            'total_mb': self.get_total_memory(),
            'num_pathways': len(self.pathways),
            'pathways': [p.get_stats() for p in self.pathways] # type: ignore
        }
    
    def analyze_routing(self, print_results: bool = True) -> dict:
        if len(self.pathways) == 0:
            return {}
        
        usage = self.router.get_pathway_usage()
        
        results = {}
        for i, (pathway, use) in enumerate(zip(self.pathways, usage)):
            results[pathway.task_name] = {
                'usage_percent': float(use * 100),
                'total_calls': pathway.usage_count,
                'memory_mb': pathway.memory_mb
            }
        
        if print_results:
            print("\n[ROUTING ANALYSIS]")
            print(f"{'Task':<15} {'Usage %':<10} {'Calls':<10} {'Memory (MB)':<12}")
            print("-" * 50)
            for task_name, stats in results.items():
                print(f"{task_name:<15} {stats['usage_percent']:>8.1f}% "
                      f"{stats['total_calls']:>8d}   "
                      f"{stats['memory_mb']:>10.2f}")
            print()
        
        return results