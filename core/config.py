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
QUANTIZE_PATHWAYS = True
THRESHOLD_PERCENTILE = 0.70  # Keep for backbone
SKIP_CLASSIFIER_QUANTIZATION = True  # NEW: Don't quantize classifier
CLASSIFIER_THRESHOLD = 0.50  # If quantizing classifier, use softer threshold

# Router
ROUTER_LR_SCALE = 0.1
FAST_LEARNER_WEIGHT_TRAIN = 0.7
PATHWAY_WEIGHT_TRAIN = 0.3

# NEW: Adaptive weights based on number of pathways
SINGLE_PATHWAY_FAST_WEIGHT = 0.9  # Trust Fast Learner more with 1 pathway
SINGLE_PATHWAY_PATH_WEIGHT = 0.1

MULTI_PATHWAY_FAST_WEIGHT = 0.3   # Trust pathways more with many pathways
MULTI_PATHWAY_PATH_WEIGHT = 0.7

# Memory
KEEP_BACKBONE_ON_RESET = True  # Transfer learning between tasks

# Logging
VERBOSE = True
PRINT_EVERY = 20  # Print every N batches

# Routing strategy
USE_ENTROPY_ROUTING = True  # Adaptive weighting based on confidence
ENTROPY_ADJUSTMENT_SCALE = 0.15  # How much entropy affects weights

# Base weights by pathway count
WEIGHTS_SINGLE_PATHWAY = {'fast': 0.7, 'pathway': 0.3}
WEIGHTS_FEW_PATHWAYS = {'fast': 0.5, 'pathway': 0.5}    # 2-3 pathways
WEIGHTS_MANY_PATHWAYS = {'fast': 0.3, 'pathway': 0.7}   # 4+ pathways

# Testing adjustments
TEST_SINGLE_PATHWAY_BOOST_FAST = 0.1  # Extra weight to Fast Learner with 1 pathway