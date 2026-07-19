"""
collect_data.py

Capture new images for a Siamese Face Verification dataset.

Controls:
    A = Save current frame as an Anchor image
    P = Save current frame as a Positive image
    Q = Quit the application
"""

import sys
import os
import uuid
from pathlib import Path

import cv2

# =============================================================================
# PROJECT SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# =============================================================================
# CONFIGURATION
# =============================================================================

WEBCAM_DEVICE = 0          # Default webcam
FRAME_SIZE = 250           # Size of the cropped square

# Fixed crop position
OFFSET_X = 200
OFFSET_Y = 120

# =============================================================================
# FUNCTIONS
# =============================================================================

def print_header():
    """Print application information."""

    print("=" * 80)
    print("WEBCAM DATA COLLECTION")
    print("=" * 80)

    print(f"\nAnchor directory : {config.ANCHOR_PATH}")
    print(f"Positive directory: {config.POSITIVE_PATH}")

    print("\nControls:")
    print("  A - Save Anchor image")
    print("  P - Save Positive image")
    print("  Q - Quit")

    print("\nOpening webcam...\n")


def save_image(image, folder):
    """
    Save an image with a unique filename.

    Args:
        image: Image to save.
        folder: Destination directory.

    Returns:
        str: Saved filename.
    """

    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(str(folder), filename)

    cv2.imwrite(filepath, image)

    return filename


def run_collection():
    """Start webcam and collect images."""

    cap = cv2.VideoCapture(WEBCAM_DEVICE)

    if not cap.isOpened():
        print("ERROR: Unable to open webcam.")
        print("Try changing WEBCAM_DEVICE to another camera index.")
        return None

    print("✓ Webcam connected.\n")

    anchor_count = 0
    positive_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Unable to read frame from webcam.")
            break

        # Crop image
        cropped = frame[
            OFFSET_Y:OFFSET_Y + FRAME_SIZE,
            OFFSET_X:OFFSET_X + FRAME_SIZE
        ]

        # Draw overlay
        display = frame.copy()

        cv2.rectangle(
            display,
            (OFFSET_X, OFFSET_Y),
            (OFFSET_X + FRAME_SIZE, OFFSET_Y + FRAME_SIZE),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "A = Anchor | P = Positive | Q = Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Anchor: {anchor_count}   Positive: {positive_count}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Webcam Data Collection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):

            filename = save_image(cropped, config.ANCHOR_PATH)

            anchor_count += 1

            print(f"✓ Anchor image #{anchor_count} saved ({filename})")

        elif key == ord("p"):

            filename = save_image(cropped, config.POSITIVE_PATH)

            positive_count += 1

            print(f"✓ Positive image #{positive_count} saved ({filename})")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return anchor_count, positive_count


def print_summary(anchor_count, positive_count):
    """Print a summary after image collection."""

    anchor_files = list(config.ANCHOR_PATH.glob("*.jpg"))
    positive_files = list(config.POSITIVE_PATH.glob("*.jpg"))

    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)

    print(f"New Anchor images   : {anchor_count}")
    print(f"New Positive images : {positive_count}")
    print(f"Total new images    : {anchor_count + positive_count}")

    print()

    print(f"Anchor images total : {len(anchor_files)}")
    print(f"Positive images total: {len(positive_files)}")

    print("\nSaved to:")

    print(f"  Anchor   : {config.ANCHOR_PATH}")
    print(f"  Positive : {config.POSITIVE_PATH}")

    print("=" * 80)

    if anchor_count + positive_count > 0:
        print("\nYou can now retrain your Siamese Network using the updated dataset.")
    else:
        print("\nNo new images were captured.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header()

    result = run_collection()

    if result is None:
        return

    anchor_count, positive_count = result

    print_summary(anchor_count, positive_count)


if __name__ == "__main__":
    main()