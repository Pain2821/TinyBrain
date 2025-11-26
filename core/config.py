# core/config.py
"""
Global configuration / constants for BitBrain v4.
Change values here instead of scanning multiple files.
"""
from pathlib import Path
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4
DATA_DIR = "data"

# Paths
BASE_DIR = Path(".").resolve()
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MEMORY_BANK_DIR = BASE_DIR / "memory_bank"
STAGING_DIR = BASE_DIR / "tmp_bitbrain_staging"

# Training
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Consolidation / Memory
MIN_EXAMPLES_PER_CLASS = 10        # lower threshold to make consolidation trigger in small datasets
REUSE_THRESHOLD = 0.75             # cosine similarity cutoff to reuse a node
CONSOLIDATION_SAMPLE_SIZE = 32     # how many samples to add to buffer per batch (max)
CLUSTER_MIN_SIZE = 5               # min cluster points to create a node
MAX_STAGE_CLUSTERS = 4             # max k for kmeans

# Router
ROUTER_THRESHOLD = 0.30            # conservative default based on observed sim ranges

# Quantization
INT8_MAX = 127.0

# Logging / runtime
VERBOSE = True

# Torch device object
TORCH_DEVICE = torch.device(DEVICE)
