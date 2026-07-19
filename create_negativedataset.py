from pathlib import Path
import tarfile
import shutil

# ============================================================================
# Paths
# ============================================================================

ARCHIVE_PATH = Path("lfw-funneled.tgz")
LFW_PATH = Path("lfw_funneled")
NEGATIVE_PATH = Path("data") / "negative"

def safe_extract(archive, destination):
    destination = Path(destination).resolve()

    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()

        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(
                f"Unsafe path detected in archive: {member.name}"
            )

    archive.extractall(path=destination)


# ============================================================================
# Extract archive
# ============================================================================

if not ARCHIVE_PATH.exists():
    raise FileNotFoundError(
        f"Archive file not found: {ARCHIVE_PATH.resolve()}"
    )

if not LFW_PATH.exists():
    print("Extracting LFW dataset...")

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        safe_extract(archive, ".")

    print("Archive extracted successfully.")
else:
    print(f"'{LFW_PATH}' already exists - skipping extraction.")


# ============================================================================
# 2. Create destination folder
# ============================================================================
NEGATIVE_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Move images to data/negative
# ============================================================================
image_extensions = {".jpg", ".jpeg", ".png"}
moved_images = 0

for person_dir in LFW_PATH.iterdir():
    if not person_dir.is_dir():
        continue

    for image_path in person_dir.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        target_name = f"{person_dir.name}_{image_path.name}"
        target_path = NEGATIVE_PATH / target_name

        # Do not overwrite images that already exist
        if target_path.exists():
            print(f"Skipped, file already exists: {target_path.name}")
            continue

        shutil.move(str(image_path), str(target_path))
        moved_images += 1

print(f"Done: {moved_images} images are now in '{NEGATIVE_PATH}'.")
