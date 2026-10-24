"""Smoke-test FastAPI /health and /predict endpoints."""
import json
import sys
from pathlib import Path

import requests

ROOT = Path(r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects")
BASE = "http://127.0.0.1:8000"


def find_sample_images(n=2):
    data = ROOT / "data" / "brain-tumor-mri-dataset"
    found = []
    for split in ("Testing", "Training"):
        split_dir = data / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img in class_dir.glob("*.jpg"):
                found.append((img, class_dir.name))
                if len(found) >= n:
                    return found
            for img in class_dir.glob("*.jpeg"):
                found.append((img, class_dir.name))
                if len(found) >= n:
                    return found
    return found


def main():
    health = requests.get(f"{BASE}/health", timeout=30)
    health.raise_for_status()
    print("GET /health:", json.dumps(health.json(), indent=2))

    samples = find_sample_images(2)
    if len(samples) < 2:
        print("Warning: fewer than 2 sample images found for predict tests", file=sys.stderr)

    for path, _ in samples:
        with path.open("rb") as f:
            resp = requests.post(
                f"{BASE}/predict",
                files={"file": (path.name, f, "image/jpeg")},
                timeout=120,
            )
        resp.raise_for_status()
        print(f"\nPOST /predict ({path.name}):", json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
