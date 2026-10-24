"""Create T08/ submission folder layout (without zipping)."""
import shutil
from pathlib import Path

ROOT = Path(r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects")
TEAM = ROOT / "T08"


def copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    TEAM.mkdir(exist_ok=True)
    for name in ("app", "model", "data"):
        src = ROOT / name
        if src.is_dir():
            copytree(src, TEAM / name)
    shutil.copy2(ROOT / "requirements.txt", TEAM / "requirements.txt")
    nb = ROOT / "notebooks" / "brain-tumor-detection.ipynb"
    nb_dir = TEAM / "notebooks"
    nb_dir.mkdir(exist_ok=True)
    if nb.is_file():
        shutil.copy2(nb, nb_dir / "pytorch_T08.ipynb")
        shutil.copy2(nb, nb_dir / "tensorflow_T08.ipynb")
    report_src = ROOT / "report_T08.pdf"
    if report_src.is_file():
        shutil.copy2(report_src, TEAM / "report_T08.pdf")
    print("Submission folder ready at", TEAM)


if __name__ == "__main__":
    main()
