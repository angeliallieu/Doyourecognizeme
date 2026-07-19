"""
Utility functions for Visualization, Evaluation und Inference
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple, Optional
from pathlib import Path

import config


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_training_history(history: dict, figsize: Tuple[int, int] = (15, 5)):
    """
    Plot Training History (Loss, Precision, Recall)
    
    Args:
        history: dict mit Keys: 'train_loss', 'val_loss', 'train_precision', etc
        figsize: Figure Size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    axes[0].plot(history['epochs'], history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['epochs'], history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision
    axes[1].plot(history['epochs'], history['train_precision'], label='Train Precision', marker='o')
    axes[1].plot(history['epochs'], history['val_precision'], label='Val Precision', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Training & Validation Precision')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Recall
    axes[2].plot(history['epochs'], history['train_recall'], label='Train Recall', marker='o')
    axes[2].plot(history['epochs'], history['val_recall'], label='Val Recall', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Training & Validation Recall')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_sample_pair(
    anchor_img: np.ndarray,
    verification_img: np.ndarray,
    label: float,
    prediction: Optional[float] = None,
    title: str = ""
):
    """
    Visualize Anchor & Verification Image Pair
    
    Args:
        anchor_img: np.ndarray (H, W, 3) normalized [0, 1]
        verification_img: np.ndarray (H, W, 3) normalized [0, 1]
        label: True Label (0 or 1)
        prediction: Predicted Similarity (optional)
        title: Title für Plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Anchor Image
    axes[0].imshow(anchor_img)
    axes[0].set_title("Anchor Image")
    axes[0].axis('off')
    
    # Verification Image
    axes[1].imshow(verification_img)
    if prediction is not None:
        label_text = f"True: {'Same' if label == 1 else 'Different'}\nPred: {prediction:.4f}"
        axes[1].set_title(f"Verification Image\n{label_text}")
    else:
        axes[1].set_title(f"Verification Image\nLabel: {'Same (1)' if label == 1 else 'Different (0)'}")
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_augmentations(
    original_img: np.ndarray,
    augmented_imgs: List[np.ndarray],
    title: str = "Image Augmentations (On-The-Fly)"
):
    """
    visualize original and augmented Version
    
    Args:
        original_img: Original Image (H, W, 3)
        augmented_imgs: List of augmented Images
        title: Plot Title
    """
    n_aug = len(augmented_imgs)
    n_cols = 4
    n_rows = (n_aug + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Augmented
    for i, img in enumerate(augmented_imgs):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Augmentation {i+1}")
        axes[i + 1].axis('off')
    
    # Hide unused subplots
    for i in range(n_aug + 1, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = config.VERIFICATION_THRESHOLD
) -> dict:
    """
    Compute Confusion Matrix based on Threshold
    
    Args:
        y_true: True Labels (0 or 1)
        y_pred: Predicted Similarities [0, 1]
        threshold: Threshold for Classification
    
    Returns:
        dict: TP, TN, FP, FN, Accuracy, Precision, Recall
    """
    y_pred_binary = (y_pred >= threshold).astype(int).flatten()
    y_true_binary = y_true.astype(int).flatten()
    
    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, dict]:
    """
    Find optimal threshold based on F1 Score
    
    Args:
        y_true: True Labels
        y_pred: Predicted Similarities
        thresholds: Thresholds to test (default: 0.1-0.9)
    
    Returns:
        Tuple[best_threshold, best_metrics]
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        metrics = compute_confusion_matrix(y_true, y_pred, threshold)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot ROC Curve
    
    Args:
        y_true: True Labels
        y_pred: Predicted Similarities
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf(), roc_auc


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def verify_faces(
    model: tf.keras.Model,
    anchor_img: np.ndarray,
    verification_img: np.ndarray,
    threshold: float = config.VERIFICATION_THRESHOLD
) -> Tuple[float, bool]:
    """
    Verification of Faces
    
    Args:
        model: Trained Siamese Model
        anchor_img: Anchor Image (100x100x3, normalized)
        verification_img: Verification Image (100x100x3, normalized)
        threshold: Similarity Threshold
    
    Returns:
        Tuple[similarity_score, is_same_person]
    """
    # Add Batch Dimension
    anchor_batch = np.expand_dims(anchor_img, axis=0)
    verification_batch = np.expand_dims(verification_img, axis=0)
    
    # Predict
    similarity = model.predict([anchor_batch, verification_batch], verbose=0)[0][0]
    is_same = similarity >= threshold
    
    return float(similarity), bool(is_same)


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def get_batch_sample(dataset: tf.data.Dataset, batch_idx: int = 0) -> Tuple:
    """
    extract Sample from dataset
    
    Args:
        dataset: tf.data.Dataset
        batch_idx: Welcher Batch
    
    Returns:
        Tuple: (anchor_img, verification_img, label) for first sample in batch
    """
    for i, batch in enumerate(dataset):
        if i == batch_idx:
            anchor, verification, label = batch
            return (
                anchor[0].numpy(),
                verification[0].numpy(),
                label[0].numpy()
            )
    
    raise IndexError(f"Batch index {batch_idx} out of range")


def print_dataset_info(dataset: tf.data.Dataset, name: str = "Dataset"):
    """Print Dataset Information"""
    print(f"\n{name} Info:")
    print(f"  Element Spec: {dataset.element_spec}")
    
    # first Element Details
    for anchor, verification, label in dataset.take(1):
        print(f"  Anchor Shape: {anchor.shape}")
        print(f"  Verification Shape: {verification.shape}")
        print(f"  Label Shape: {label.shape}")
        print(f"  Anchor Range: [{anchor.numpy().min():.4f}, {anchor.numpy().max():.4f}]")
        print(f"  Label Values: {np.unique(label.numpy())}")
