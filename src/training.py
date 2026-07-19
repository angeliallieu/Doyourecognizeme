"""
Training Loop for Siamese Network with Early Stopping and Checkpoint Management

Features:
- Custom Training Loop with @tf.function for Performance
- Early Stopping based on Validation Loss
- saved Checkpoint of all N Epochs
- Metric Tracking (Loss, Precision, Recall)
- Cluster-compatibility (no plotting in Training Loop)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List
import json
from datetime import datetime

import config
from config import EmbeddingConfig, DatasetConfig


# ============================================================================
# CUSTOM METRICS & LOSS TRACKING
# ============================================================================

class TrainingHistory:
    """Tracking of Training Metrics"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'epochs': []
        }
    
    def add_epoch(self, epoch: int, metrics: dict):
        """add Epoch Metrics """
        self.history['epochs'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))
    
    def get_best_epoch(self, metric: str = 'val_loss') -> Tuple[int, float]:
        """take best Epoch based on Metric"""
        if metric == 'val_loss':
            idx = np.argmin(self.history['val_loss'])
        elif metric == 'val_precision':
            idx = np.argmax(self.history['val_precision'])
        else:
            idx = 0
        
        return self.history['epochs'][idx], self.history[metric][idx]
    
    def save_json(self, path: str):
        """save History as JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training History saved: {path}")
    
    def load_json(self, path: str):
        """Load history from JSON"""
        with open(path, 'r') as f:
            self.history = json.load(f)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStoppingCallback:
    """Simple Early Stopping implementation"""
    
    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        metric: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Args:
            patience: Amount of Epochs without fixing before Stop
            min_delta: Minimal fix to count as fix
            metric: which Metrics tracked ('val_loss', 'val_precision', etc)
            mode: 'min' for Loss, 'max' for Accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.wait_count = 0
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_epoch = 0
    
    def step(self, epoch: int, metric_value: float) -> bool:
        """
        Check if training should stop
        
        Returns:
            bool: True if stops, False if continues
        """
        if self.mode == 'min':
            if metric_value < self.best_value - self.min_delta:
                self.best_value = metric_value
                self.best_epoch = epoch
                self.wait_count = 0
                return False
        else:  # max
            if metric_value > self.best_value + self.min_delta:
                self.best_value = metric_value
                self.best_epoch = epoch
                self.wait_count = 0
                return False
        
        self.wait_count += 1
        
        if self.wait_count >= self.patience:
            print(f"\n⚠️ Early Stopping! Best {self.metric}={self.best_value:.6f} for Epoch {self.best_epoch}")
            return True
        
        return False


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    epoch: int,
    checkpoint_dir: str = str(config.CHECKPOINT_DIR),
    history: Optional[TrainingHistory] = None
) -> str:
    """
    save model Checkpoint
    
    Args:
        model: trained model
        optimizer: Optimizer State
        epoch: current Epoch
        checkpoint_dir: directory für Checkpoints
        history: Training History (optional)
    
    Returns:
        str: path to saved Checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"siamese_epoch_{epoch:04d}"
    model.save(str(checkpoint_path) + '.keras')

    # save also if history is available
    if history:
        history_path = checkpoint_dir / f"history_epoch_{epoch:04d}.json"
        history.save_json(str(history_path))
    
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str
) -> tf.keras.Model:
    """
    Load model from Checkpoint
    
    Args:
        checkpoint_path: path to Checkpoint
    
    Returns:
        Model: loaded model
    """
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"✓ Model loaded from: {checkpoint_path}")
    return model


def get_latest_checkpoint(
    checkpoint_dir: str = str(config.CHECKPOINT_DIR)
) -> Optional[str]:
    """
    Find latest Checkpoint in the directory
    
    Args:
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Optional[str]: path to latest Checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob("siamese_epoch_*"))
    if not checkpoints:
        return None
    
    return str(checkpoints[-1])


def get_best_checkpoint(
    checkpoint_dir: str = str(config.CHECKPOINT_DIR)
) -> Optional[str]:
    """
    Find Checkpoint with the lowest Validation Loss
    
    Reads the history_epoch_*.json files and selects the checkpoint
    with the lowest val_loss.
    
    Args:
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Optional[str]: path to best Checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Find all History files
    history_files = sorted(checkpoint_dir.glob("history_epoch_*.json"))
    if not history_files:
        # Fallback: Load the latest Checkpoint
        return get_latest_checkpoint(str(checkpoint_dir))
    
    best_val_loss = float('inf')
    best_checkpoint = None
    
    for history_file in history_files:
        try:
            # Extract Epoch number from filename
            epoch_str = history_file.stem.split('_')[-1]
            checkpoint_path = checkpoint_dir / f"siamese_epoch_{epoch_str}.keras"
            
            if not checkpoint_path.exists():
                continue
            
            # Load History
            with open(str(history_file), 'r') as f:
                history = json.load(f)
            
            # Find val_loss
            val_losses = history.get('val_loss', [])
            if val_losses:
                current_val_loss = min(val_losses) if isinstance(val_losses, list) else val_losses
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_checkpoint = str(checkpoint_path)
        
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            continue
    
    # Fallback: no val_loss found
    if best_checkpoint is None:
        return get_latest_checkpoint(str(checkpoint_dir))
    
    return best_checkpoint


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

@tf.function
def train_step(
    batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    train_precision: Precision,
    train_recall: Recall
) -> tf.Tensor:
    """
    Single Training Step with GradientTape
    
    Args:
        batch: (anchor_imgs, verification_imgs, labels)
        model: Siamese Network
        loss_fn: Loss Function (Binary Crossentropy)
        optimizer: Optimizer (Adam)
        train_precision: Precision Metric
        train_recall: Recall Metric
    
    Returns:
        tf.Tensor: Loss value
    """
    anchor_img, verification_img, label = batch
    
    with tf.GradientTape() as tape:
        # Forward Pass
        yhat = model([anchor_img, verification_img], training=True)
        
        # Calculate Loss
        loss = loss_fn(label, yhat)
    
    # Gradients & Update
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Update Metrics
    train_precision.update_state(label, yhat)
    train_recall.update_state(label, yhat)
    
    return loss


@tf.function
def test_step(
    batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    val_precision: Precision,
    val_recall: Recall
) -> tf.Tensor:
    """
    Single Validation Step (no gradient update)
    
    Args:
        batch: (anchor_imgs, verification_imgs, labels)
        model: Siamese Network
        loss_fn: Loss Function
        val_precision: Precision Metric
        val_recall: Recall Metric
    
    Returns:
        tf.Tensor: Loss value
    """
    anchor_img, verification_img, label = batch
    
    # Forward Pass (no GradientTape)
    yhat = model([anchor_img, verification_img], training=False)
    
    # Loss
    loss = loss_fn(label, yhat)
    
    # Update Metrics
    val_precision.update_state(label, yhat)
    val_recall.update_state(label, yhat)
    
    return loss


def train_epoch(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
    epoch: int,
    verbose: int = config.VERBOSE_LEVEL
) -> Dict[str, float]:
    """
    Train one complete epoch

    Args:
        train_data: Training Dataset
        val_data: Validation Dataset
        model: Siamese Network
        loss_fn: Loss Function
        optimizer: Optimizer
        epoch: Epoch Number (for Logging)
        verbose: Logging Level
    
    Returns:
        dict: Metrics for this Epoch
    """
    # Initialize Metrics
    train_loss = tf.keras.metrics.Mean()
    train_precision = Precision()
    train_recall = Recall()
    
    val_loss = tf.keras.metrics.Mean()
    val_precision = Precision()
    val_recall = Recall()
    
    # ======== TRAINING ========
    if verbose >= 1:
        print(f"Epoch {epoch} - Training...")
    
    for batch_idx, batch in enumerate(train_data):
        loss = train_step(batch, model, loss_fn, optimizer, train_precision, train_recall)
        train_loss.update_state(loss)
        
        if verbose >= 2 and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss={train_loss.result():.6f}")
    
    # ======== VALIDATION ========
    if verbose >= 1:
        print(f"Epoch {epoch} - Validation...")
    
    for batch in val_data:
        loss = test_step(batch, model, loss_fn, val_precision, val_recall)
        val_loss.update_state(loss)
    
    # ======== METRICS ========
    metrics = {
        'train_loss': float(train_loss.result()),
        'val_loss': float(val_loss.result()),
        'train_precision': float(train_precision.result()),
        'val_precision': float(val_precision.result()),
        'train_recall': float(train_recall.result()),
        'val_recall': float(val_recall.result()),
    }
    
    if verbose >= 1:
        print(f"  Loss: {metrics['train_loss']:.6f} | Val Loss: {metrics['val_loss']:.6f}")
        print(f"  Precision: {metrics['train_precision']:.4f} | Val Precision: {metrics['val_precision']:.4f}")
        print(f"  Recall: {metrics['train_recall']:.4f} | Val Recall: {metrics['val_recall']:.4f}")
    
    return metrics


def train_model(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    checkpoint_dir: str = str(config.CHECKPOINT_DIR),
    checkpoint_interval: int = config.SAVE_INTERVAL,
    early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
    verbose: int = config.VERBOSE_LEVEL,
    resume_from_checkpoint: Optional[str] = None
) -> TrainingHistory:
    """
    main training Loop with Early Stopping and Checkpointing
    
    Args:
        train_data: Training Dataset
        val_data: Validation Dataset
        model: Siamese Network
        epochs: Maximum amount Epoch
        learning_rate: Learning Rate for Adam
        checkpoint_dir: directory for Checkpoints
        checkpoint_interval: save Checkpoint all N Epoch
        early_stopping_patience: Epochs fix before Stops
        verbose: Logging Level
        resume_from_checkpoint: Optional Checkpoint to resume training

    Returns:
        TrainingHistory: Training Metrics
    """

    if not resume_from_checkpoint:
        checkpoint_path = Path(checkpoint_dir)
        old_checkpoints = list(checkpoint_path.glob("siamese_epoch_*.keras"))
        old_histories = list(checkpoint_path.glob("history_epoch_*.json"))
        old_training_history = list(checkpoint_path.glob("training_history.json"))

        all_old_files = old_checkpoints + old_histories + old_training_history

        if all_old_files:
            print(f"\n⚠️ Old Checkpoints found ({len(all_old_files)} files)")
            print(f"   Deleting to avoid confusion...")
            for f in all_old_files:
                try:
                    f.unlink()
                    print(f"   ✓ Deleted: {f.name}")
                except Exception as e:
                    print(f"   ✗ Error deleting {f.name}: {e}")
            print(f"   → Training starts FRESH (Epoch 1)\n")
        else:
            print(f"\n✓ No old Checkpoints found → Fresh Start\n")

    # Setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    history = TrainingHistory()
    early_stopper = EarlyStoppingCallback(patience=early_stopping_patience)
    
    start_epoch = 1
    
    # Resume from Checkpoint if available
    if resume_from_checkpoint:
        print(f"Resume Training von: {resume_from_checkpoint}")
        model = load_checkpoint(resume_from_checkpoint)
        # try to start Epoch from Checkpoint Name extraction
        try:
            start_epoch = int(resume_from_checkpoint.split('_')[-1]) + 1
        except:
            pass
    
    print(f"\n{'='*80}")
    print(f"START TRAINING")
    print(f"{'='*80}")
    print(f"Epochs: {start_epoch} - {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Checkpoint Interval: {checkpoint_interval}")
    print(f"{'='*80}\n")
    
    # ======== TRAINING LOOP ========
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        # Train & Validate
        metrics = train_epoch(train_data, val_data, model, loss_fn, optimizer, epoch, verbose)
        history.add_epoch(epoch, metrics)
        
        # Early Stopping Check
        if early_stopper.step(epoch, metrics['val_loss']):
            print(f"\n✓ Training done (Early Stopping)")
            break
        
        # Save Checkpoint
        if epoch % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, history)
            print(f"✓ Checkpoint saved (Epoch {epoch})")
    
    print(f"\n{'='*80}")
    print(f"TRAINING DONE")
    print(f"{'='*80}")
    
    # save final History
    history_path = Path(checkpoint_dir) / "training_history.json"
    history.save_json(str(history_path))
    
    return history
