"""Wait until the MRI dataset is complete, then execute the training notebook."""
import os
import subprocess
import sys
import time

ROOT = r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects"
DATA = os.path.join(ROOT, "data", "brain-tumor-mri-dataset")
NOTEBOOK = os.path.join(ROOT, "notebooks", "brain-tumor-detection.ipynb")
MIN_IMAGES = 7150


def count_images() -> int:
    n = 0
    for dp, _, fns in os.walk(DATA):
        if ".cache" in dp.replace("\\", "/"):
            continue
        for f in fns:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                n += 1
    return n


def has_training_tree() -> bool:
    return os.path.isdir(os.path.join(DATA, "Training")) and os.path.isdir(
        os.path.join(DATA, "Testing")
    )


print("Waiting for dataset (need Training/, Testing/, and >=", MIN_IMAGES, "images)...")
while True:
    n = count_images()
    ok = has_training_tree() and n >= MIN_IMAGES
    print(f"  images={n}, training_tree={has_training_tree()}")
    if ok:
        break
    time.sleep(60)

print("Executing notebook (this will take a while)...")
cmd = [
    "conda",
    "run",
    "-n",
    "crowd-gpu",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    NOTEBOOK,
    "--output",
    "brain-tumor-detection.ipynb",
    "--ExecutePreprocessor.timeout=7200",
]
subprocess.run(cmd, cwd=os.path.join(ROOT, "notebooks"), check=True)
print("Notebook execution finished.")
