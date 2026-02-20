from dataclasses import dataclass
import torch


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    backbone: str = "mobilenet_v2"
    pretrained: bool = True
    stage_indices: tuple[int, ...] = (3, 6, 13, 18)
    img_size: int = 224

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs_per_task: int = 6
    batch_size: int = 32
    num_workers: int = 4
    log_every: int = 20

    tau_reuse: float = 0.85
    tau_route: float = 0.60
    gamma_high: float = 0.85
    gamma_low: float = 0.60
    delta_conf: float = 0.15
    use_confidence_logic: bool = True
    max_clusters: int = 5
    min_samples_for_cluster: int = 8
    max_samples_per_class: int = 200
    quantize: bool = True

    task_aware_routing: bool = True
    eval_time_batches: int = 5
