"""Poll until dataset is complete, then execute the notebook once."""
import subprocess
import time
from pathlib import Path

ROOT = Path(r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects")
SCRIPT = ROOT / "scripts" / "split_counts.py"
NOTEBOOK = ROOT / "notebooks" / "brain-tumor-detection.ipynb"


def total_images() -> int:
    import os

    base = ROOT / "data" / "brain-tumor-mri-dataset"
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    total = 0
    for split in ("Training", "Testing"):
        for c in classes:
            p = base / split / c
            if not p.is_dir():
                return -1
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                total += len(list(p.glob(ext)))
    return total


while True:
    n = total_images()
    print(f"Dataset ready check: total={n} (need 7200 with all 8 class folders)")
    if n >= 7150:
        break
    time.sleep(30)

print("Starting notebook execution...")
subprocess.run(
    [
        "conda",
        "run",
        "-n",
        "crowd-gpu",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output",
        "brain-tumor-detection.ipynb",
        "--ExecutePreprocessor.timeout=14400",
    ],
    cwd=ROOT / "notebooks",
    check=True,
)
print("Training notebook finished.")
