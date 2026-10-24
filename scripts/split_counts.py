import os
from pathlib import Path

base = Path(r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects\data\brain-tumor-mri-dataset")
classes = ["glioma", "meningioma", "notumor", "pituitary"]
total = 0
for split in ("Training", "Testing"):
    for c in classes:
        p = base / split / c
        n = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.jpeg"))) + len(list(p.glob("*.png"))) if p.is_dir() else -1
        print(f"{split}/{c}: {n}")
        if n > 0:
            total += n
print("TOTAL:", total)
