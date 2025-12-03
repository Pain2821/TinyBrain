from pathlib import Path
import torch

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
BASE_DIR = Path('.').resolve()
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints_internal'
LOG_DIR = BASE_DIR / 'logs_internal'

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Model architecture
BASE_ARCH = 'mobilenet_v2'
FEATURE_DIM = 1280

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
IMG_SIZE = 224

# Continual learning
EPOCHS_PER_TASK = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Consolidation
QUANTIZE_PATHWAYS = True  # Set False to test without quantization
THRESHOLD_PERCENTILE = 0.70  # Keep top 30% weights
LAYER_WISE_THRESHOLD = True

# Router
ROUTER_LR_SCALE = 0.1  # Router learns slower than Fast Learner
FAST_LEARNER_WEIGHT_TRAIN = 0.7  # During training
PATHWAY_WEIGHT_TRAIN = 0.3

# Memory
KEEP_BACKBONE_ON_RESET = True  # Transfer learning between tasks

# Logging
VERBOSE = True
PRINT_EVERY = 20  # Print every N batches