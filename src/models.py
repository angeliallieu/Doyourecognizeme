"""
Model Architecture for Facial Verification Siamese Network
Based on Nicholas Renottes Tutorial

Architecture:
1. Embedding Layer: Conv2D + MaxPooling + Dense(4096)
2. L1 Distance Layer: Calculates L1 Distance between two Embeddings
3. Classification Layer: Dense(1, sigmoid) for binary classification
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import regularizers
from typing import Tuple

import config
from config import EmbeddingConfig


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class L1Dist(Layer):
    """
    Custom Layer for L1 Distance Calculation (Manhattan Distance)
    
    Input: (embedding1, embedding2)
    Output: Absolute Difference between the Embeddings
    
    Formula: L1_distance = |embedding1 - embedding2|
    """
    
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        """Calculate L1 Distance"""
        # Convert lists to tensors if necessary
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]

        return tf.math.abs(input_embedding - validation_embedding)
    
    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Calculate Output Shape"""
        return input_shape[0]


# ============================================================================
# EMBEDDING NETWORK BUILDER
# ============================================================================

def create_embedding_network() -> Model:
    """
    Create Embedding Network (Siamese Network Basis)
    
    Architecture (based on Nicholas Renotte):
    - Input: (100, 100, 3)
    - Conv2D(64, 10x10) + MaxPooling(2x2)
    - Conv2D(128, 7x7) + MaxPooling(2x2)
    - Conv2D(128, 4x4) + MaxPooling(2x2)
    - Conv2D(256, 4x4)
    - Flatten + Dense(4096, sigmoid)
    - Output: (4096,)
    
    Returns:
        Model: Embedding Network
    """
    inp = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS), name='input_image')
    l2 = regularizers.l2(EmbeddingConfig.L2_WEIGHT_DECAY)
    
    # Block 1
    c1 = Conv2D(
        EmbeddingConfig.FILTERS_1,
        EmbeddingConfig.KERNEL_1,
        activation='relu',
        kernel_regularizer=l2,
        name='conv2d_1'
    )(inp)
    m1 = MaxPooling2D(
        pool_size=EmbeddingConfig.POOL_1,
        padding='same',
        name='maxpool_1'
    )(c1)
    
    # Block 2
    c2 = Conv2D(
        EmbeddingConfig.FILTERS_2,
        EmbeddingConfig.KERNEL_2,
        activation='relu',
        kernel_regularizer=l2,
        name='conv2d_2'
    )(m1)
    m2 = MaxPooling2D(
        pool_size=EmbeddingConfig.POOL_2,
        padding='same',
        name='maxpool_2'
    )(c2)
    
    # Block 3
    c3 = Conv2D(
        EmbeddingConfig.FILTERS_3,
        EmbeddingConfig.KERNEL_3,
        activation='relu',
        kernel_regularizer=l2,
        name='conv2d_3'
    )(m2)
    m3 = MaxPooling2D(
        pool_size=EmbeddingConfig.POOL_3,
        padding='same',
        name='maxpool_3'
    )(c3)
    
    # Block 4 (Final Embedding)
    c4 = Conv2D(
        EmbeddingConfig.FILTERS_4,
        EmbeddingConfig.KERNEL_4,
        activation='relu',
        kernel_regularizer=l2,
        name='conv2d_4'
    )(m3)
    
    # Flatten and Dense Layer
    f1 = Flatten(name='flatten')(c4)
    f1 = Dropout(EmbeddingConfig.DROPOUT_RATE, name='embedding_dropout')(f1)
    d1 = Dense(
        EmbeddingConfig.DENSE_UNITS,
        activation=EmbeddingConfig.ACTIVATION,
        kernel_regularizer=l2,
        name='dense_embedding'
    )(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding_network')


# ============================================================================
# SIAMESE NETWORK BUILDER
# ============================================================================

def create_siamese_network() -> Model:
    """
    Create Complete Siamese Network for Facial Verification
    
    Architecture:
    - Input 1: Anchor Image (100x100x3)
    - Input 2: Verification Image (100x100x3)
    - Both processed by the same Embedding Network
    - L1 Distance Layer between Embeddings
    - Classification: Dense(1, sigmoid)
    
    Output: Similarity Score [0, 1]
      - Close to 0: Different People
      - Close to 1: Same Person
    
    Returns:
        Model: Complete Siamese Network
    """
    # Create Embedding Network (used twice)
    embedding = create_embedding_network()
    
    # Inputs
    input_image = Input(
        shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS),
        name='anchor_image'
    )
    verification_image = Input(
        shape=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS),
        name='verification_image'
    )
    
    # Embeddings (both trough same Embedding Network - Shared Weights)
    inp_embedding = embedding(input_image)
    val_embedding = embedding(verification_image)
    
    # L1 Distance
    siamese_layer = L1Dist(name='l1_distance')
    distances = siamese_layer(inp_embedding, val_embedding)
    distances = Dropout(EmbeddingConfig.DROPOUT_RATE, name='classifier_dropout')(distances)
    
    # Classification Layer
    classifier = Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=regularizers.l2(EmbeddingConfig.L2_WEIGHT_DECAY),
        name='similarity_score'
    )(distances)
    
    return Model(
        inputs=[input_image, verification_image],
        outputs=classifier,
        name='siamese_network'
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_model_summary(model: Model, verbose: int = 1) -> None:
    """
    Print detailed Model Summary
    
    Args:
        model: Keras Model
        verbose: 0=quiet, 1=normal, 2=detailed
    """
    print("\n" + "="*80)
    print(f"MODEL SUMMARY: {model.name}")
    print("="*80)
    model.summary()
    print("="*80 + "\n")


def get_model_info(model: Model) -> dict:
    """
    Get Model Information
    
    Returns:
        dict: Model statistics
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'name': model.name,
        'layers': len(model.layers),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'input_shape': model.input_shape if isinstance(model.input_shape, tuple) else [s for s in model.input_shape],
        'output_shape': model.output_shape,
    }


def load_siamese_model(checkpoint_path: str) -> Model:
    """
    Load trained Siamese Model from Checkpoint
    
    Args:
        checkpoint_path: path to Checkpoint/SavedModel
    
    Returns:
        Model: Loaded model
    """
    if checkpoint_path.endswith('.h5'):
        # Keras HDF5 Format
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        # SavedModel Format
        model = tf.keras.models.load_model(checkpoint_path)
    
    return model


def save_siamese_model(
    model: Model,
    save_path: str,
    save_format: str = 'saved_model'
) -> None:
    """
    Save Siamese Model
    
    Args:
        model: Keras Model
        save_path: target path
        save_format: 'saved_model' oder 'h5'
    """
    if save_format == 'h5':
        model.save(save_path + '.h5')
    else:
        model.save(save_path)
    
    print(f"✓ Model saved: {save_path}")
