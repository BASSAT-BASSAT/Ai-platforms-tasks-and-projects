"""Download Brain Tumor MRI dataset from Hugging Face (resumable, low concurrency)."""
import os
import shutil
import subprocess
import time

LOCAL = r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects\data\brain-tumor-mri-dataset"
REPO = "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset"
MIN_IMAGES = 7000


def count_images() -> int:
    n = 0
    if not os.path.isdir(LOCAL):
        return 0
    for dp, _, fns in os.walk(LOCAL):
        if ".cache" in dp.replace("\\", "/") or ".git" in dp.replace("\\", "/"):
            continue
        for f in fns:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                n += 1
    return n


def try_snapshot():
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    return snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=LOCAL,
        max_workers=1,
        resume_download=True,
    )


def try_git_lfs():
    parent = os.path.dirname(LOCAL)
    os.makedirs(parent, exist_ok=True)
    tmp = LOCAL + "_git"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", f"https://huggingface.co/datasets/{REPO}", tmp],
        check=True,
    )
    if os.path.isdir(LOCAL):
        shutil.rmtree(LOCAL, ignore_errors=True)
    os.rename(tmp, LOCAL)


if count_images() >= MIN_IMAGES and os.path.isdir(os.path.join(LOCAL, "Training")):
    print("Dataset already complete:", count_images(), "images")
    raise SystemExit(0)

for attempt, fn in enumerate([try_snapshot, try_git_lfs, try_snapshot], start=1):
    try:
        print(f"Download attempt {attempt} via {fn.__name__}...")
        fn()
        n = count_images()
        print("Images after attempt:", n)
        if n >= MIN_IMAGES and os.path.isdir(os.path.join(LOCAL, "Training")):
            print("Done:", LOCAL)
            break
    except Exception as exc:
        print(f"Attempt {attempt} failed: {exc}")
        time.sleep(60 * attempt)
else:
    raise SystemExit(f"Download incomplete ({count_images()} images). Retry later.")
