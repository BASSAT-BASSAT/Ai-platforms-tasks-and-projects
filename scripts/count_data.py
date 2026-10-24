import os

root = r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects\data\brain-tumor-mri-dataset"
n = 0
n_cache = 0
tops = os.listdir(root) if os.path.isdir(root) else []
for dp, dns, fns in os.walk(root):
    in_cache = ".cache" in dp.replace("\\", "/")
    for f in fns:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            if in_cache:
                n_cache += 1
            else:
                n += 1
print("top:", tops)
print("images (materialized):", n)
print("images (in .cache):", n_cache)
print("Training exists:", os.path.isdir(os.path.join(root, "Training")))
