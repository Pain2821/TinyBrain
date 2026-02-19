from dataclasses import dataclass
import torch


@dataclass
class BitBrainMVPConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs_per_task: int = 6
    num_workers: int = 4
    batch_size: int = 32
    stage_indices: tuple[int, ...] = (3, 6, 13, 18)
    reuse_threshold: float = 0.85
    route_threshold: float = 0.60
    high_conf: float = 0.85
    low_conf: float = 0.60
    conf_gap: float = 0.15
    k_max: int = 6
    max_samples_per_class: int = 200
    random_state: int = 42
    log_every: int = 20
