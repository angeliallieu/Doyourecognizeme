"""
Streamlit Face Verification App
===============================
An interactive application for face verification using a trained Siamese Network.

Features:
- Camera photo capture
- Face detection before verification
- Adjustable similarity threshold
- Automatic square cropping
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
from pathlib import Path
import random

# Project imports
import config
from src import data, training, utils
from src.models import L1Dist


# ============================================================================
# FACE DETECTION
# ============================================================================

@st.cache_resource
def load_face_cascade():
    """Load OpenCV Haar Cascade for face detection"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


def detect_face(image: np.ndarray, face_cascade) -> tuple:
    """
    Detects face in image and returns the bounding box.

    Args:
        image: Input image (RGB or BGR)
        face_cascade: OpenCV CascadeClassifier

    Returns:
        tuple: (face_detected, x, y, w, h) or (False, 0, 0, 0, 0)
    """
    # Convert to grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return True, x, y, w, h

    return False, 0, 0, 0, 0


def crop_face_to_square(
    image: np.ndarray, x: int, y: int, w: int, h: int, padding: float = 0.4 ) -> np.ndarray:
    """
    Creates a square face crop around a detected bounding box.

    Bounding-box coordinates are safely limited to the image boundaries.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot crop an empty image.")

    img_h, img_w = image.shape[:2]

    if img_h == 0 or img_w == 0:
        raise ValueError("Image has invalid dimensions.")

    # Convert bounding box to valid image coordinates.
    left = max(0, x)
    top = max(0, y)
    right = min(img_w, x + w)
    bottom = min(img_h, y + h)

    # The detected box lies completely outside the image or is invalid.
    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid face bounding box: x={x}, y={y}, w={w}, h={h}"
        )

    # Center of the valid face box.
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    face_w = right - left
    face_h = bottom - top

    # Desired crop size, including context around the face.
    desired_size = int(max(face_w, face_h) * (1 + padding))

    # A square crop cannot be larger than the image's shortest side.
    crop_size = min(desired_size, img_w, img_h)

    if crop_size <= 0:
        raise ValueError("Calculated crop size is invalid.")

    # Position a square crop around the face center.
    crop_x1 = center_x - crop_size // 2
    crop_y1 = center_y - crop_size // 2

    # Shift the crop back into the image if it would exceed an edge.
    crop_x1 = max(0, min(crop_x1, img_w - crop_size))
    crop_y1 = max(0, min(crop_y1, img_h - crop_size))

    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if cropped.size == 0:
        raise ValueError(
            f"Face crop is empty. Crop coordinates: "
            f"({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})"
        )

    return cropped



# ============================================================================
# MODEL LOADING
# ============================================================================

def get_best_epoch_info() -> dict:
    """Analyzes the training history and finds the best epoch"""
    history_path = config.CHECKPOINT_DIR / "training_history.json"

    if not history_path.exists():
        return None

    try:
        import json
        with open(history_path, 'r') as f:
            history = json.load(f)

        val_losses = history.get('val_loss', [])
        epochs = history.get('epochs', [])

        if not val_losses or not epochs:
            return None

        # Find best epoch
        best_idx = np.argmin(val_losses)
        best_epoch = epochs[best_idx]
        best_val_loss = val_losses[best_idx]

        # Check if checkpoint for this epoch exists
        checkpoint_path = config.CHECKPOINT_DIR / f"siamese_epoch_{best_epoch:04d}.keras"
        checkpoint_exists = checkpoint_path.exists()

        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'checkpoint_exists': checkpoint_exists,
            'total_epochs': len(epochs)
        }
    except Exception:
        return None

@st.cache_resource
def load_model():
    """Loads the trained Siamese Network"""
    best_checkpoint = training.get_best_checkpoint(str(config.CHECKPOINT_DIR))

    if best_checkpoint is None:
        best_checkpoint = str(config.CHECKPOINT_DIR / "siamese_final.keras")

    if not Path(best_checkpoint).exists():
        return None, f"No model found in: {config.CHECKPOINT_DIR}", None

    try:
        model = keras.models.load_model(
            best_checkpoint,
            custom_objects={'L1Dist': L1Dist}
        )

        # Extract epoch from checkpoint name
        checkpoint_name = Path(best_checkpoint).stem
        if 'epoch_' in checkpoint_name:
            loaded_epoch = int(checkpoint_name.split('_')[-1])
        elif 'final' in checkpoint_name:
            loaded_epoch = 'final'
        else:
            loaded_epoch = 'unknown'

        return model, best_checkpoint, loaded_epoch
    except Exception as load_error:
        return None, str(load_error), None


@st.cache_data
def load_anchor_images():
    """Loads all available anchor images"""
    try:
        anchor_paths = data.load_image_paths(str(config.ANCHOR_PATH), max_images=None)
        return anchor_paths
    except Exception as e:
        return []


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Prepares an image for the model.

    Args:
        image: Input image (any size, RGB, 0-255)

    Returns:
        np.ndarray: Preprocessed image (100x100x3, normalized 0-1)
    """
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)

    # Resize and normalize
    processed = data.preprocess_image(img_tensor)

    return processed.numpy()


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Doyourecognizeme",
        page_icon="📷",
        layout="wide"
    )

    st.title("📷 Face Verification")
    st.markdown("*Verify if a captured photo matches the stored anchor images.*")

    # Load resources
    face_cascade = load_face_cascade()
    model, model_info, loaded_epoch = load_model()
    anchor_paths = load_anchor_images()
    best_epoch_info = get_best_epoch_info()

    # Sidebar configuration
    st.sidebar.header("Settings")

    # Threshold slider
    threshold = st.sidebar.slider(
        "Verification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.VERIFICATION_THRESHOLD,
        step=0.01,
        help="Similarity score must be >= threshold for 'Verified'"
    )

    # Anchor image selection
    st.sidebar.subheader("Anchor Image")

    if not anchor_paths:
        st.sidebar.warning("! No anchor images found!")
        selected_anchor_idx = None
    else:
        anchor_selection_mode = st.sidebar.radio(
            "Selection Mode",
            ["Random", "Manual"],
            help="Choose how the anchor image is selected"
        )

        if anchor_selection_mode == "Random":
            if st.sidebar.button("🔀 Choose New Anchor"):
                st.session_state['anchor_idx'] = random.randint(0, len(anchor_paths) - 1)

            if 'anchor_idx' not in st.session_state:
                st.session_state['anchor_idx'] = random.randint(0, len(anchor_paths) - 1)

            selected_anchor_idx = st.session_state['anchor_idx']
        else:
            selected_anchor_idx = st.sidebar.selectbox(
                "Select Anchor Image",
                range(len(anchor_paths)),
                format_func=lambda i: Path(anchor_paths[i]).name
            )

    st.sidebar.info(f"{len(anchor_paths)} anchor images available")

    # Status
    st.sidebar.divider()
    st.sidebar.subheader("📊 Status")

    if model is None:
        st.sidebar.error(f"❌ Model not loaded: {model_info}")
    else:
        st.sidebar.success("Model loaded")
        st.sidebar.info(f"{Path(model_info).name}")

        # Show info about best epoch
        if best_epoch_info:
            st.sidebar.caption(f"Loaded Epoch: **{loaded_epoch}**")

            if not best_epoch_info['checkpoint_exists']:
                st.sidebar.warning(
                    f"⚠Best epoch was **{best_epoch_info['best_epoch']}** "
                    f"(val_loss: {best_epoch_info['best_val_loss']:.4f}), "
                    f"but no checkpoint exists!"
                )
            else:
                if loaded_epoch == best_epoch_info['best_epoch']:
                    st.sidebar.caption(f"Best epoch loaded (val_loss: {best_epoch_info['best_val_loss']:.4f})")

    # Main area
    col1, col2 = st.columns(2)

    # Display anchor image
    with col1:
        st.subheader("Anchor Image (Reference)")

        if selected_anchor_idx is not None and anchor_paths:
            anchor_path = anchor_paths[selected_anchor_idx]
            anchor_img = data.load_image(anchor_path).numpy().astype(np.uint8)
            anchor_processed = preprocess_for_model(anchor_img)

            st.image(anchor_img, caption=f"Anchor: {Path(anchor_path).name}", width=300)
        else:
            st.warning("No anchor image selected")
            anchor_processed = None

    # Camera capture
    with col2:
        st.subheader("Take A Photo")

        camera_image = st.camera_input("Place your face in the middle")

    # Verification
    st.divider()

    if camera_image is not None and model is not None and anchor_processed is not None:
        # Load image
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        captured_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        # Face detection
        face_detected, fx, fy, fw, fh = detect_face(captured_image, face_cascade)

        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])

        with result_col1:
            st.subheader("Captured Image")

            # Display image with face bounding box
            display_image = captured_image.copy()
            if face_detected:
                cv2.rectangle(display_image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
                st.image(display_image, caption="Face detected ✓", width='stretch')
            else:
                st.image(display_image, caption="No face detected ✗", width='stretch')

        with result_col2:
            st.subheader(" Cropped Face")

            if face_detected:
                # Crop face to square
                cropped_face = crop_face_to_square(captured_image, fx, fy, fw, fh)
                st.image(cropped_face, caption=f"Size: {cropped_face.shape[0]}x{cropped_face.shape[1]}", width='stretch')

                # Process for model
                verification_processed = preprocess_for_model(cropped_face)
            else:
                st.warning("⚠️ No face detected for verification!")
                verification_processed = None

        with result_col3:
            st.subheader("🔍 Verification Result")

            if face_detected and verification_processed is not None:
                # Perform verification
                similarity, is_verified = utils.verify_faces(
                    model,
                    anchor_processed,
                    verification_processed,
                    threshold=threshold
                )

                # Display result
                if is_verified:
                    st.success("# ✅ VERIFIED")
                else:
                    st.error("# ❌ NOT VERIFIED")

                # Metrics
                st.metric("Similarity Score", f"{similarity:.4f}")
                st.metric("Threshold", f"{threshold:.2f}")

                # Progress bar
                st.progress(min(similarity, 1.0))

                # Details
                with st.expander("📊 Details"):
                    st.write(f"**Similarity Score:** {similarity:.6f}")
                    st.write(f"**Threshold:** {threshold:.4f}")
                    st.write(f"**Difference:** {similarity - threshold:.6f}")
                    st.write(f"**Result:** {'Same Person' if is_verified else 'Different Person'}")
            else:
                st.warning("Verification not possible - no face detected!")
                st.info("Please take a photo where your face is clearly visible.")

    elif camera_image is None:
        st.info("Please take a photo with the camera to start verification.")

    elif model is None:
        st.error("The model could not be loaded. Please check if the trained model is available.")

    elif anchor_processed is None:
        st.warning("No anchor image available. Please add anchor images to the `data/anchor` folder.")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
            Face Verification App | Siamese Network | 
            Input Size: {img_size}x{img_size} | 
            Threshold: {threshold:.2f}
        </small>
    </div>
    """.format(img_size=config.IMG_SIZE, threshold=threshold), unsafe_allow_html=True)


if __name__ == "__main__":
    main()



