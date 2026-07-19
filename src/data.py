"""
Data Loading, Preprocessing and On-The-Fly Augmentation
For Facial Verification Siamese Network

Strategy: On-The-Fly Augmentation while Training
"""

import os
import random
import numpy as np

# Supress TensorFlow C++ INFO/WARNING Logs (z. B. Metal/NUMA Warning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Iterator, Optional

import config
from config import AugmentationConfig, DatasetConfig


# ============================================================================
# IMAGE LOADING & PREPROCESSING
# ============================================================================

def load_image(file_path: str) -> tf.Tensor:
    """
    Load single image from file path
    
    Args:
        file_path: path to image file (.jpg, .png)
    
    Returns:
        tf.Tensor: Unprocessed Image (Original size)
    """
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img, channels=config.IMG_CHANNELS)
    return img


def preprocess_image(img: tf.Tensor) -> tf.Tensor:
    """
    Preprocessing: Resize to 100x100 and normalization
    
    Args:
        img: tf.Tensor (any size)
    
    Returns:
        tf.Tensor: Resized (100x100x3) and normalized (0-1)
    """
    # Resize to target size
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    return img


def preprocess_image_from_path(file_path: str) -> tf.Tensor:
    """
    Load and Preprocess combined (for tf.data.Dataset.map)
    
    Args:
        file_path: path to image file
    
    Returns:
        tf.Tensor: Preprocessed Image
    """
    img = load_image(file_path)
    return preprocess_image(img)


# ============================================================================
# ON-THE-FLY AUGMENTATION (Core Feature)
# ============================================================================

def augment_image(img: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    """
    Augment single image with random transformations
    (Stateless for Reproducibility)
    
    Args:
        img: tf.Tensor (any size, values 0-1)
        seed: Optional Seed for Reproducibility (None = random)
    
    Returns:
        tf.Tensor: Augmented Image (same shape as Input)
    """
    if seed is None:
        seed = (np.random.randint(0, 100), np.random.randint(0, 100))
    else:
        seed = (seed, seed)
    
    # Random Brightness
    img = tf.image.stateless_random_brightness(
        img, 
        max_delta=AugmentationConfig.BRIGHTNESS_MAX_DELTA,
        seed=seed
    )
    
    # Random Contrast
    img = tf.image.stateless_random_contrast(
        img,
        lower=AugmentationConfig.CONTRAST_LOWER,
        upper=AugmentationConfig.CONTRAST_UPPER,
        seed=(seed[0] + 1, seed[1] + 1)
    )
    
    # Random Saturation
    img = tf.image.stateless_random_saturation(
        img,
        lower=AugmentationConfig.SATURATION_LOWER,
        upper=AugmentationConfig.SATURATION_UPPER,
        seed=(seed[0] + 2, seed[1] + 2)
    )
    
    # Random Horizontal Flip (50%)
    img = tf.image.stateless_random_flip_left_right(
        img,
        seed=(seed[0] + 3, seed[1] + 3)
    )
    
    # Clip to [0, 1] (if Augmentation limits exceeded)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img


def augment_image_batch(images: tf.Tensor) -> tf.Tensor:
    """
    Augment single image with random transformations

    Args:
        images: tf.Tensor (Batch, H, W, 3)
    
    Returns:
        tf.Tensor: Augmented Batch
    """
    return tf.map_fn(lambda x: augment_image(x), images, dtype=tf.float32)


def augment_image_stateless(img: tf.Tensor) -> tf.Tensor:
    """
    Augment single image with a TF-generated seed, so that each
    execution (per element / per epoch) produces different random values.
    """
    # create a Seed inside the TF-Graphen (changes per call)
    seed = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
    # Random Brightness
    img = tf.image.stateless_random_brightness(
        img,
        max_delta=AugmentationConfig.BRIGHTNESS_MAX_DELTA,
        seed=seed
    )
    # Random Contrast
    seed_c = tf.stack([seed[0] + 1, seed[1] + 1])
    img = tf.image.stateless_random_contrast(
        img,
        lower=AugmentationConfig.CONTRAST_LOWER,
        upper=AugmentationConfig.CONTRAST_UPPER,
        seed=seed_c
    )
    # Random Saturation
    seed_s = tf.stack([seed[0] + 2, seed[1] + 2])
    img = tf.image.stateless_random_saturation(
        img,
        lower=AugmentationConfig.SATURATION_LOWER,
        upper=AugmentationConfig.SATURATION_UPPER,
        seed=seed_s
    )
    # Random Horizontal Flip (stateless alternative)
    seed_f = tf.stack([seed[0] + 3, seed[1] + 3])
    # use stateless_random_uniform to decide whether to flip
    flip_val = tf.random.stateless_uniform([], seed=seed_f, minval=0, maxval=1)
    img = tf.cond(flip_val < AugmentationConfig.FLIP_PROB,
                  lambda: tf.image.flip_left_right(img),
                  lambda: img)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img



# ============================================================================
# DATASET CREATION & PIPELINE
# ============================================================================

def load_image_paths(
    folder_path: str,
    max_images: Optional[int] = config.DatasetConfig.MAX_IMAGES_PER_CLASS,
    random_sample: bool = False
) -> List[str]:
    """
    Load image paths from a folder.
    
    Args:
        folder_path: Path to the image folder
        max_images: Maximum number of images to return.
            None disables the limit.
        random_sample: If True, a random subset will be sampled.
    
    Returns:
        List[str]: List of image paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    paths = [str(p) for p in folder.iterdir() if p.suffix in image_extensions]
    
    paths = sorted(paths)
    if max_images is not None:
        if random_sample and len(paths) > max_images:
            paths = random.sample(paths, max_images)
        else:
            paths = paths[:max_images]
    return paths


def create_triplet_dataset(
    anchor_paths: List[str],
    positive_paths: List[str],
    negative_paths: List[str],
    apply_augmentation: bool = True,
    augmentation_repeats: int = config.DatasetConfig.AUGMENTATION_REPEATS
) -> tf.data.Dataset:
    """
    Create Triplet Dataset for Siamese Training with multiple augmentations
    Format: (anchor_img, positive_img, label=1.0), (anchor_img, negative_img, label=0.0)
    
    Args:
        anchor_paths: List of Anchor image paths
        positive_paths: List of Positive image paths
        negative_paths: List of Negative image paths (ALL are used!)
        apply_augmentation: Apply augmentation (True)
        augmentation_repeats: How often each sample is augmented and repeated (e.g., 3)
    
    Returns:
        tf.data.Dataset: (anchor, img, label) Triplets
    """
    num_anchors = len(anchor_paths)
    num_positives = len(positive_paths)
    num_negatives = len(negative_paths)
    
    print(f"\n Dataset Balancing:")
    print(f"  Anchor/Positive Pairs: {num_anchors}")
    print(f"  Negative Samples:     {num_negatives}")
    
    # ========================================================================
    # Positive Pairs (Anchor, Positive, Label=1)
    # Use ALL Anchor/Positive Pairs (no limit anymore!)
    # ========================================================================
    positive_dataset = tf.data.Dataset.from_tensor_slices((
        anchor_paths,
        positive_paths,
        np.ones(num_anchors, dtype=np.float32)
    ))
    if apply_augmentation and augmentation_repeats > 1:
        positive_dataset = positive_dataset.repeat(augmentation_repeats)
    
    # ========================================================================
    # Negative Pairs (Anchor, Negative, Label=0)
    # If more Negatives than Anchors: Repeat Anchor-Paths
    # ========================================================================
    # Calculate how often Anchors need to be repeated
    if num_negatives > num_anchors:
        repeat_factor = (num_negatives + num_anchors - 1) // num_anchors  # Ceiling
        anchor_paths_repeated = (
            tf.data.Dataset.from_tensor_slices(anchor_paths)
            .repeat(repeat_factor)
            .take(num_negatives)
        )
    else:
        anchor_paths_repeated = tf.data.Dataset.from_tensor_slices(
            anchor_paths[:num_negatives]
        )
    
    negative_dataset = tf.data.Dataset.zip((
        anchor_paths_repeated,
        tf.data.Dataset.from_tensor_slices(negative_paths),
        tf.data.Dataset.from_tensor_slices(np.zeros(num_negatives, dtype=np.float32))
    ))
    
    if apply_augmentation and augmentation_repeats > 1:
        negative_dataset = negative_dataset.repeat(augmentation_repeats)
    
    print(f"  Total Pairs per Epoch: {num_anchors + num_negatives}")
    if num_negatives > num_anchors:
        print(f"  Anchors repeated {repeat_factor}x for negative pairing")
    
    # Combine both datasets
    dataset = positive_dataset.concatenate(negative_dataset)
    
    # Preprocessing & Optional Augmentation
    def process_triplet(anchor_path, img_path, label):
        """Load and preprocess images with optional augmentation"""
        anchor_img = load_image(anchor_path)
        img = load_image(img_path)
        
        # Preprocess (Resize + Normalization)
        anchor_img = preprocess_image(anchor_img)
        img = preprocess_image(img)
        
        # On-The-Fly Augmentation
        if apply_augmentation:
            # Use TF-based stateless augmentation so each map call yields different results
            anchor_img = augment_image_stateless(anchor_img)
            img = augment_image_stateless(img)
        
        return anchor_img, img, label
    
    dataset = dataset.map(
        process_triplet,
        num_parallel_calls=DatasetConfig.NUM_PARALLEL_CALLS
    )
    
    return dataset


def prepare_dataset_pipeline(
    dataset: tf.data.Dataset,
    shuffle_buffer: int = DatasetConfig.SHUFFLE_BUFFER,
    batch_size: int = config.BATCH_SIZE,
    prefetch_buffer: int = DatasetConfig.PREFETCH_BUFFER,
    train_split: float = config.TRAIN_SIZE,
    cache: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare Dataset for Training
    - Cache: Reduces Reload Time
    - Shuffle: Random Order
    - Batch: Group Data
    - Prefetch: Load next batch while Current batch is training

    Args:
        dataset: Input Dataset
        shuffle_buffer: Buffer size for Shuffling
        batch_size: Batch size
        prefetch_buffer: Prefetch Buffer
        train_split: Train/Test Split Ratio
        cache: Dataset cache
    
    Returns:
        Tuple[train_dataset, test_dataset]
    """
    # Cache, if no On-The-Fly-Augmentation required.
    if cache:
        dataset = dataset.cache()
    
    # Shuffle
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    
    # Calculate Length
    dataset_size = len(list(dataset))
    train_size = int(dataset_size * train_split)
    
    # Split
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    
    # Batch + Prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(prefetch_buffer)
    test_dataset = test_dataset.batch(batch_size).prefetch(prefetch_buffer)
    
    return train_dataset, test_dataset


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_all_datasets(
    anchor_dir: str = str(config.ANCHOR_PATH),
    positive_dir: str = str(config.POSITIVE_PATH),
    negative_dir: str = str(config.NEGATIVE_PATH),
    apply_augmentation: bool = True,
    max_images_per_class: Optional[int] = config.DatasetConfig.MAX_IMAGES_PER_CLASS,
    train_split: float = config.TRAIN_SIZE,
    batch_size: int = config.BATCH_SIZE,
    negative_sample_size: Optional[int] = config.DatasetConfig.NEGATIVE_SAMPLE_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load all data and create Pipeline with IMAGE-LEVEL SPLIT
    
    Important: Anchor/Positive will be split on IMAGE-Edge , not on Pair-Edge.
    This ensures that no single anchor image appears simultaneously in Train and Test.
    
    Args:
        anchor_dir: Path to Anchor folder
        positive_dir: Path to Positive folder
        negative_dir: Path to Negative folder
        apply_augmentation: use Augmentation
        max_images_per_class: Limit pro class; None uses all images
        train_split: Train/Test Split Ratio (z.B. 0.7 = 70% Training)
        batch_size: Batch size
        negative_sample_size: How many Negative per Dataset (None = all)
    
    Returns:
        Tuple[train_dataset, test_dataset]
    """
    # ======== 1. LOADING ALL PATHS ========
    print(f"Loading Anchor Images from: {anchor_dir}")
    anchor_paths = load_image_paths(anchor_dir, max_images=max_images_per_class)
    print(f"  ✓ {len(anchor_paths)} Anchor Images loaded")
    
    print(f"Loading Positive Images from: {positive_dir}")
    positive_paths = load_image_paths(positive_dir, max_images=max_images_per_class)
    print(f"  ✓ {len(positive_paths)} Positive Images loaded")
    
    print(f"Loading Negative Images from: {negative_dir}")
    negative_paths = load_image_paths(
        negative_dir,
        max_images=negative_sample_size,
        random_sample=True
    )
    print(f"  ✓ {len(negative_paths)} Negative Images loaded")
    
    # ======== 2. IMAGE-LEVEL SPLIT ========
    num_images = len(anchor_paths)
    split_idx = int(num_images * train_split)
    
    # Shuffle for random distribution on IMAGE-Edge
    indices = list(range(num_images))
    np.random.shuffle(indices)
    
    train_indices = sorted(indices[:split_idx])
    test_indices = sorted(indices[split_idx:])
    
    train_anchor_paths = [anchor_paths[i] for i in train_indices]
    train_positive_paths = [positive_paths[i] for i in train_indices]
    test_anchor_paths = [anchor_paths[i] for i in test_indices]
    test_positive_paths = [positive_paths[i] for i in test_indices]

    # Split negatives separately to avoid data leakage between train and test.
    if len(negative_paths) <= 1:
        train_negative_paths = negative_paths
        test_negative_paths = negative_paths
        print("\n Only 1 negative sample available - train/test must share it.")
    else:
        shuffled_negative_paths = negative_paths.copy()
        random.shuffle(shuffled_negative_paths)
        negative_split_idx = int(len(shuffled_negative_paths) * train_split)
        negative_split_idx = max(1, min(len(shuffled_negative_paths) - 1, negative_split_idx))
        train_negative_paths = shuffled_negative_paths[:negative_split_idx]
        test_negative_paths = shuffled_negative_paths[negative_split_idx:]
    
    print(f"\n Image-Level Split):")
    print(f"  Training: {len(train_anchor_paths)} Anchors + {len(train_positive_paths)} Positives")
    print(f"  Test:     {len(test_anchor_paths)} Anchors + {len(test_positive_paths)} Positives")
    print(f"  Training Negatives: {len(train_negative_paths)}")
    print(f"  Test Negatives:     {len(test_negative_paths)}")
    
    # ======== 3. CREATE SEPARATE DATASETS ========
    print(f"\nCreate Train dataset...")
    train_dataset = create_triplet_dataset(
        train_anchor_paths, train_positive_paths, train_negative_paths,
        apply_augmentation=apply_augmentation
    )
    
    print(f"\nCreate Test dataset...")
    test_dataset = create_triplet_dataset(
        test_anchor_paths, test_positive_paths, test_negative_paths,
        apply_augmentation=False  # No Augmentation for valid Test-Metrics
    )
    
    # ======== 4. BATCH + PREFETCH ========
    print(f"\nPrepare Train Pipeline...")
    train_dataset = train_dataset.shuffle(DatasetConfig.SHUFFLE_BUFFER) \
        .batch(batch_size) \
        .prefetch(DatasetConfig.PREFETCH_BUFFER)
    
    print(f"Prepare Test Pipeline...")
    test_dataset = test_dataset.batch(batch_size) \
        .prefetch(DatasetConfig.PREFETCH_BUFFER)
    
    print(f"  ✓ Train Dataset: {len(list(train_dataset))} Batches")
    print(f"  ✓ Test Dataset: {len(list(test_dataset))} Batches")
    
    return train_dataset, test_dataset


def get_dataset_stats(
    anchor_dir: str = str(config.ANCHOR_PATH),
    positive_dir: str = str(config.POSITIVE_PATH),
    negative_dir: str = str(config.NEGATIVE_PATH),
    max_images_per_class: Optional[int] = config.DatasetConfig.MAX_IMAGES_PER_CLASS
) -> dict:
    """
    Get statistics about available data
    
    Returns:
        dict: Amount of images per category + expected training images
    """
    anchor_count = len(load_image_paths(anchor_dir, max_images=max_images_per_class))
    positive_count = len(load_image_paths(positive_dir, max_images=max_images_per_class))
    negative_total = len(load_image_paths(negative_dir, max_images=None))
    negative_per_epoch = config.DatasetConfig.NEGATIVE_SAMPLE_SIZE
    
    total_pairs_per_epoch = anchor_count + negative_per_epoch
    
    return {
        'anchor': anchor_count,
        'positive': positive_count,
        'negative_total': negative_total,
        'negative_per_epoch': negative_per_epoch,
        'total_pairs_per_epoch': total_pairs_per_epoch,
        'negative': negative_per_epoch,
        'min': min(anchor_count, positive_count, negative_per_epoch),
        'total_pairs': total_pairs_per_epoch,
    }
