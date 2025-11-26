# # core/router.py
# from typing import List, Tuple, Optional
# import numpy as np
# import torch
# from core.memory_bank import MemoryBank
# from core.config import ROUTER_THRESHOLD
# from core.utils_logging import log

# class Router:
#     def __init__(self, memory_bank: MemoryBank, threshold: float = ROUTER_THRESHOLD):
#         self.mb = memory_bank
#         self.threshold = threshold
#         log(f"Router initialized with threshold={threshold}")

#     def route(self, stage_activations: List[Tuple[int, torch.Tensor]], nn_predicted_class: Optional[str] = None) -> Tuple[Optional[str], float]:
#         if not self.mb.pathways:
#             return None, 0.0

#         # Filter pathways by predicted class if provided
#         if nn_predicted_class:
#             valid_pathways = {}
#             for tag, node_ids in self.mb.pathways.items():
#                 for nid in node_ids:
#                     node = self.mb.nodes.get(nid)
#                     if node and node.class_label == nn_predicted_class:
#                         valid_pathways[tag] = node_ids
#                         break
#             if not valid_pathways:
#                 return None, 0.0
#         else:
#             valid_pathways = self.mb.pathways

#         best_tag = None
#         best_score = 0.0
#         for tag, node_ids in valid_pathways.items():
#             pathway_nodes = [self.mb.nodes[nid] for nid in node_ids if nid in self.mb.nodes]
#             if not pathway_nodes:
#                 continue
#             stage_similarities = []
#             for stage_idx, activation in stage_activations:
#                 act = activation.numpy() if isinstance(activation, torch.Tensor) else activation
#                 act = act.astype(np.float32)
#                 act = act / (np.linalg.norm(act) + 1e-9)
#                 stage_nodes = [n for n in pathway_nodes if n.stage_idx == stage_idx]
#                 if not stage_nodes:
#                     continue
#                 centroids = np.stack([n.centroid for n in stage_nodes])
#                 avg_centroid = centroids.mean(axis=0)
#                 sim = float(np.dot(act, avg_centroid))
#                 stage_similarities.append(sim)
#             if stage_similarities:
#                 pathway_score = float(np.mean(stage_similarities))
#                 if pathway_score > best_score:
#                     best_score = pathway_score
#                     best_tag = tag

#         if best_score >= self.threshold:
#             return best_tag, best_score
#         return None, best_score


# import numpy as np
# import torch
# from typing import List, Tuple, Optional

# from core.memory_bank import MemoryBank
# from core.utils_logging import log
# from core.config import ROUTER_THRESHOLD


# class Router:
#     """
#     Auto-calibrated Router for BitBrain.
#     Computes ideal threshold based on actual memory bank similarity statistics.
#     """

#     def __init__(self, memory_bank: MemoryBank, threshold: Optional[float] = None):
#         self.mb = memory_bank

#         # Use manual threshold only if provided
#         if threshold is not None:
#             self.threshold = threshold
#             log(f"[Router] Manual threshold used: {self.threshold:.3f}")
#         else:
#             # Auto-calibrate based on stored nodes
#             self.threshold = self._auto_calibrate_threshold()
#             log(f"[Router] Auto-calibrated threshold: {self.threshold:.3f}")

#     # ----------------------------------------------------------------------
#     # AUTO-CALIBRATION
#     # ----------------------------------------------------------------------
#     def _auto_calibrate_threshold(self) -> float:
#         """
#         Computes threshold automatically by examining similarities
#         between centroids of nodes in the memory bank.

#         Idea:
#         - Similar nodes (same stage & class) → higher similarity
#         - Unrelated nodes → lower similarity
#         - Threshold = mid-way between these groups
#         """

#         if len(self.mb.nodes) < 2:
#             log("[Router] Not enough nodes for calibration. Using default.")
#             return ROUTER_THRESHOLD

#         similarities = []

#         node_list = list(self.mb.nodes.values())

#         # Compare each centroid to all others
#         for i in range(len(node_list)):
#             for j in range(i + 1, len(node_list)):
#                 a = node_list[i].centroid
#                 b = node_list[j].centroid

#                 sim = float(np.dot(a, b))

#                 # Only consider nodes from SAME stage — important!
#                 if node_list[i].stage_idx == node_list[j].stage_idx:
#                     similarities.append(sim)

#         if len(similarities) == 0:
#             log("[Router] No per-stage similarity pairs. Using default.")
#             return ROUTER_THRESHOLD

#         sims = np.array(similarities)

#         # Compute statistics
#         median_sim = float(np.median(sims))
#         mean_sim = float(np.mean(sims))

#         # Recommended threshold = 70% of mean similarity
#         calibrated = round(mean_sim * 0.70, 3)

#         log(f"[Router] Calibration stats: mean={mean_sim:.3f}, median={median_sim:.3f}")
#         log(f"[Router] Suggested threshold = 0.7 * mean = {calibrated:.3f}")

#         # Avoid tiny thresholds
#         calibrated = max(calibrated, 0.10)

#         return calibrated

#     # ----------------------------------------------------------------------

#     def route(
#         self,
#         stage_activations: List[Tuple[int, torch.Tensor]],
#         nn_predicted_class: Optional[str] = None
#     ) -> Tuple[Optional[str], float]:

#         if not self.mb.pathways:
#             return None, 0.0

#         # Filter pathways by class
#         if nn_predicted_class:
#             valid_pathways = {
#                 tag: ids
#                 for tag, ids in self.mb.pathways.items()
#                 if any(self.mb.nodes[n].class_label == nn_predicted_class for n in ids)
#             }
#             if not valid_pathways:
#                 return None, 0.0
#         else:
#             valid_pathways = self.mb.pathways

#         best_tag = None
#         best_score = 0.0

#         for tag, node_ids in valid_pathways.items():
#             pathway_nodes = [self.mb.nodes[n] for n in node_ids if n in self.mb.nodes]
#             if not pathway_nodes:
#                 continue

#             sims = []

#             for stage_idx, act in stage_activations:
#                 vec = act.numpy().astype(np.float32)
#                 vec = vec / (np.linalg.norm(vec) + 1e-9)

#                 # Nodes from this stage only
#                 stage_nodes = [n for n in pathway_nodes if n.stage_idx == stage_idx]
#                 if not stage_nodes:
#                     continue

#                 centroids = np.stack([n.centroid for n in stage_nodes])
#                 avg_centroid = centroids.mean(axis=0)

#                 sims.append(float(np.dot(vec, avg_centroid)))

#             if sims:
#                 score = float(np.mean(sims))
#                 if score > best_score:
#                     best_score = score
#                     best_tag = tag

#         # Decision
#         if best_score >= self.threshold:
#             return best_tag, best_score

#         return None, best_score


# core/router.py

import numpy as np
import torch
from typing import List, Tuple, Optional

from core.memory_bank import MemoryBank
from core.utils_logging import log
from core.config import ROUTER_THRESHOLD


class Router:
    """
    BitBrain Router (SAFE VERSION)
    - Only compares nodes from the SAME stage.
    - Avoids dimension mismatch.
    - Allows lifelong learning without deleting memory_bank.
    """

    def __init__(self, memory_bank: MemoryBank, threshold: float = ROUTER_THRESHOLD):
        self.mb = memory_bank
        self.threshold = threshold
        log(f"[Router] Initialized with threshold={self.threshold}")

    # ----------------------------------------------------------------------
    # FIXED SAFE ROUTING
    # ----------------------------------------------------------------------
    def route(
        self,
        stage_activations: List[Tuple[int, torch.Tensor]],
        nn_predicted_class: Optional[str] = None
    ) -> Tuple[Optional[str], float]:

        if not self.mb.pathways:
            return None, 0.0

        # FILTER BY PREDICTED CLASS
        if nn_predicted_class:
            valid_pathways = {
                tag: node_ids
                for tag, node_ids in self.mb.pathways.items()
                if any(self.mb.nodes[n].class_label == nn_predicted_class for n in node_ids)
            }

            if not valid_pathways:
                return None, 0.0

        else:
            valid_pathways = self.mb.pathways

        best_tag = None
        best_score = 0.0

        # -------------------------------------------------------
        # SAFE MATCHING: ONLY compare SAME-STAGE nodes
        # -------------------------------------------------------
        for tag, node_ids in valid_pathways.items():

            pathway_nodes = [self.mb.nodes[n] for n in node_ids if n in self.mb.nodes]
            stage_scores = []

            for stage_idx, activation in stage_activations:

                # Only compare nodes from SAME STAGE
                same_stage_nodes = [
                    n for n in pathway_nodes
                    if n.stage_idx == stage_idx
                ]

                if not same_stage_nodes:
                    continue

                # Normalize activation
                act = activation.numpy().astype(np.float32)
                act = act / (np.linalg.norm(act) + 1e-9)

                # Build centroid matrix
                centroids = np.stack([n.centroid for n in same_stage_nodes])
                avg_centroid = centroids.mean(axis=0)

                # SHAPES ALWAYS MATCH HERE → NO MORE ERRORS
                sim = float(np.dot(act, avg_centroid))
                stage_scores.append(sim)

            # Average similarity across matched stages
            if stage_scores:
                pathway_score = float(np.mean(stage_scores))

                if pathway_score > best_score:
                    best_score = pathway_score
                    best_tag = tag

        # DECISION
        if best_score >= self.threshold:
            return best_tag, best_score

        return None, best_score
