"""
Central Configuration for Facial Verification Siamese Network
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ANCHOR_PATH = DATA_DIR / "anchor"
POSITIVE_PATH = DATA_DIR / "positive"
NEGATIVE_PATH = DATA_DIR / "negative"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they do not exist
for dir_path in [CHECKPOINT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
IMG_SIZE = 100  # Target size for the network (100x100)
IMG_CHANNELS = 3  # RGB

# ============================================================================
# DATA AUGMENTATION (On-The-Fly)
# ============================================================================
class AugmentationConfig:
    """Configuration for on-the-fly data augmentation"""
    BRIGHTNESS_MAX_DELTA = 0.30  # ±30%
    CONTRAST_LOWER = 0.8
    CONTRAST_UPPER = 1.2
    SATURATION_LOWER = 0.8
    SATURATION_UPPER = 1.2
    FLIP_PROB = 0.5  # 50% chance for horizontal flip
    NUM_AUGMENTATIONS = 10

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4  # Adam optimizer
EPOCHS = 50  # Target number of epochs
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum improvement required

# Train/Test Split
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class EmbeddingConfig:
    """Embedding network architecture"""
    FILTERS_1 = 64
    KERNEL_1 = (10, 10)
    POOL_1 = (2, 2)

    FILTERS_2 = 128
    KERNEL_2 = (7, 7)
    POOL_2 = (2, 2)

    FILTERS_3 = 128
    KERNEL_3 = (4, 4)
    POOL_3 = (2, 2)

    FILTERS_4 = 256
    KERNEL_4 = (4, 4)

    DENSE_UNITS = 4096
    ACTIVATION = 'sigmoid'
    L2_WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.35

# ============================================================================
# INFERENCE
# ============================================================================
VERIFICATION_THRESHOLD = 0.25  # Threshold for Siamese network output
WEBCAM_DEVICE = 0  # Default webcam index (adjustable)
WEBCAM_FRAME_SIZE = 250  # Crop size to 250x250px
WEBCAM_OFFSET_X = 200
WEBCAM_OFFSET_Y = 120

# ============================================================================
# GPU/CPU MANAGEMENT
# ============================================================================
class GPUConfig:
    """GPU Memory Management"""
    ENABLE_GPU = True
    GPU_MEMORY_GROWTH = True  # Prevents Out-Of-Memory (OOM) errors
    GPU_MEMORY_LIMIT_MB = None
    DEVICE_COUNT = 1  # Number of GPUs

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================
VERBOSE_LEVEL = 1
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
PLOT_INTERVAL = 1  # Plot metrics every N epochs
ENABLE_TENSORBOARD = True

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
class DatasetConfig:
    """Parameters for data loading and preprocessing"""
    PRELOAD_DATA = False
    MAX_IMAGES_PER_CLASS = 400
    NEGATIVE_SAMPLE_SIZE = 2000  # Random negative images per dataset load call
    SHUFFLE_BUFFER = 10000  # Shuffle buffer size
    PREFETCH_BUFFER = 8  # tf.data prefetch buffer size for performance optimization
    NUM_PARALLEL_CALLS = 4  # Number of parallel data loading threads
    AUGMENTATION_REPEATS = 1  # Number of times each sample is repeated with different augmentations