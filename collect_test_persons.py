"""
Multi-Person Data Collection Script
Collects images of different test subjects via webcam OR folder
Saves to: data/test_persons/person_1/, person_2/, etc.

Usage:
    python scripts/collect_test_persons.py
"""

import os
import cv2
import uuid
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def create_person_directory(person_id):
    """Create directory for the test subject"""
    person_dir = Path(config.PROJECT_ROOT) / "data" / "test_persons" / f"person_{person_id}"
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir


def collect_from_webcam(person_id, num_images=30):
    """
    Collect images of a test subject via webcam

    Controls:
    - 'c' = Capture image
    - 'q' = Quit/Exit
    """
    person_dir = create_person_directory(person_id)

    print(f"\n{'='*80}")
    print(f"WEBCAM COLLECTION - Person {person_id}")
    print(f"{'='*80}")
    print(f"\nTarget Directory: {person_dir}")
    print(f"Target Count: {num_images} images")
    print(f"\nControls:")
    print(f"  'c' = Capture image")
    print(f"  'q' = Quit/Exit")
    print(f"\n{'='*80}\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ ERROR: Webcam could not be opened!")
        return False

    captured_count = 0

    try:
        while captured_count < num_images:
            ret, frame = cap.read()

            if not ret:
                print("✗ Error reading from webcam")
                break

            # Calculate centered crop coordinates
            frame_height, frame_width = frame.shape[:2]
            x_start = (frame_width - config.WEBCAM_FRAME_SIZE) // 2
            y_start = (frame_height - config.WEBCAM_FRAME_SIZE) // 2
            x_end = x_start + config.WEBCAM_FRAME_SIZE
            y_end = y_start + config.WEBCAM_FRAME_SIZE

            # Ensure coordinates are within bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(frame_width, x_end)
            y_end = min(frame_height, y_end)

            cropped_frame = frame[y_start:y_end, x_start:x_end]

            # Draw rectangle on display
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Person {person_id} - Captured: {captured_count}/{num_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                display_frame,
                "Press 'c' to capture, 'q' to quit",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow('Image Collection', display_frame)

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Save image
                img_path = person_dir / f"{uuid.uuid1()}.jpg"
                cv2.imwrite(str(img_path), cropped_frame)
                captured_count += 1
                print(f"✓ Image {captured_count}/{num_images} saved")

            elif key == ord('q'):
                print(f"\n WARNING: Capture canceled - {captured_count}/{num_images} images saved")
                break

        if captured_count == num_images:
            print(f"\n✓ Successfully saved {num_images} images!")
            return True

    except KeyboardInterrupt:
        print(f"\n WARNING: Capture interrupted - {captured_count}/{num_images} images saved")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_count == num_images


def collect_from_folder(person_id, folder_path, num_images=30):
    """
    Collect images of a test subject from a folder

    Args:
        person_id: Person ID (1-10)
        folder_path: Path to the folder containing images
        num_images: Number of images to copy
    """
    person_dir = create_person_directory(person_id)
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"✗ ERROR: Folder not found: {folder_path}")
        return False

    print(f"\n{'='*80}")
    print(f"FOLDER COLLECTION - Person {person_id}")
    print(f"{'='*80}")
    print(f"\nSource Folder: {folder_path}")
    print(f"Target Directory: {person_dir}")
    print(f"Target Count: {num_images} images\n")

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in folder_path.iterdir()
                   if f.suffix in image_extensions]

    if len(image_files) < num_images:
        print(f" WARNING: Folder only contains {len(image_files)} images, but {num_images} requested")
        print(f"   Copying {len(image_files)} available images")
        num_images = len(image_files)

    try:
        for i, img_file in enumerate(image_files[:num_images]):
            # Read image
            img = cv2.imread(str(img_file))

            if img is None:
                print(f" Could not read image: {img_file.name}")
                continue

            # Resize to 250x250 (if larger)
            if img.shape[0] > 250 or img.shape[1] > 250:
                img = cv2.resize(img, (250, 250))

            # Save with a unique filename
            dest_path = person_dir / f"{uuid.uuid1()}.jpg"
            cv2.imwrite(str(dest_path), img)

            print(f"✓ Image {i+1}/{num_images} copied: {img_file.name}")

        print(f"\n✓ Successfully copied {num_images} images!")
        return True

    except Exception as e:
        print(f"✗ ERROR while copying: {e}")
        return False


def main():
    """Main Function"""
    print("\n" + "="*80)
    print("MULTI-PERSON TEST DATA COLLECTION")
    print("="*80)

    # 1. User Input: Person ID
    while True:
        try:
            person_id = input("\nPerson ID (1-10): ").strip()
            person_id = int(person_id)
            if 1 <= person_id <= 10:
                break
            else:
                print("✗ Please enter a number between 1 and 10")
        except ValueError:
            print("✗ Invalid input, please enter a number")

    # 2. User Input: Source
    while True:
        print("\nSelect Source:")
        print("  (1) Webcam")
        print("  (2) Folder")
        source = input("Choose source (1 or 2): ").strip()

        if source == "1":
            # Webcam
            success = collect_from_webcam(person_id, num_images=30)
            break

        elif source == "2":
            # Folder
            folder_path = input("Path to image folder: ").strip()
            success = collect_from_folder(person_id, folder_path, num_images=30)
            break

        else:
            print("✗ Invalid selection, please enter 1 or 2")

    # 3. Summary
    print("\n" + "="*80)
    if success:
        person_dir = Path(config.PROJECT_ROOT) / "data" / "test_persons" / f"person_{person_id}"
        num_files = len(list(person_dir.glob("*.jpg")))
        print(f"✓ SUCCESSFULLY COMPLETED")
        print(f"  Person ID: {person_id}")
        print(f"  Images saved: {num_files}")
        print(f"  Storage Location: {person_dir}")
    else:
        print(f"✗ ERROR: Data collection unsuccessful")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()