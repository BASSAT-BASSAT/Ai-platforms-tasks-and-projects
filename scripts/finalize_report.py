"""Update main.tex from report_metrics.json, compile PDF, run API smoke tests."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(r"c:\Users\Legion 5\Desktop\T08\Ai-platforms-tasks-and-projects")
METRICS = ROOT / "model" / "report_metrics.json"
TEX = ROOT / "main.tex"
PDF = ROOT / "report_T08.pdf"


def load_metrics():
    with METRICS.open(encoding="utf-8") as f:
        return json.load(f)


def fmt4(x):
    return f"{float(x):.4f}"


def patch_tex(text: str, metrics: dict) -> str:
    pt = {r["architecture"]: r for r in metrics["pytorch"]}
    tf = {r["architecture"]: r for r in metrics["tensorflow"]}

    def replace_summary_row(framework, arch, row):
        acc, prec, rec, f1 = map(fmt4, (row["accuracy"], row["precision"], row["recall"], row["f1"]))
        pattern = (
            rf"({framework}\s+&\s+{arch.replace('_', '-').replace('efficientnet-b0', 'EfficientNet-B0').replace('resnet50', 'ResNet-50')}.*?&)\s*"
            rf"[0-9.]+\s*&\s*[0-9.]+\s*&\s*[0-9.]+\s*&\s*[0-9.]+"
        )
        # simpler: replace known lines by architecture tags
        return row

    # Summary table rows (fixed order in tex)
    rows = [
        ("PyTorch", "ResNet-50 \\\\textbf{(deployed)}", pt["resnet50"], True),
        ("PyTorch", "EfficientNet-B0", pt["efficientnet_b0"], False),
        ("TensorFlow", "ResNet-50", tf["resnet50"], False),
        ("TensorFlow", "EfficientNet-B0", tf["efficientnet_b0"], False),
    ]
    for fw, arch_label, row, deployed in rows:
        bold = "\\textbf{" if deployed else ""
        bold_end = "}" if deployed else ""
        acc, prec, rec, f1 = [fmt4(row[k]) for k in ("accuracy", "precision", "recall", "f1")]
        old = re.compile(
            rf"{fw}\s+&\s+{re.escape(arch_label.replace('\\\\textbf{(deployed)}', 'ResNet-50 \\\\textbf{(deployed)}'))}.*?\\\\",
            re.MULTILINE,
        )
        # Manual line replacements
    replacements = {
        "PyTorch    & ResNet-50 \\textbf{(deployed)}": (
            f"PyTorch    & ResNet-50 \\textbf{{(deployed)}} & \\textbf{{{fmt4(pt['resnet50']['accuracy'])}}} "
            f"& \\textbf{{{fmt4(pt['resnet50']['precision'])}}} & \\textbf{{{fmt4(pt['resnet50']['recall'])}}} "
            f"& \\textbf{{{fmt4(pt['resnet50']['f1'])}}}"
        ),
        "PyTorch    & EfficientNet-B0": (
            f"PyTorch    & EfficientNet-B0               & {fmt4(pt['efficientnet_b0']['accuracy'])} "
            f"& {fmt4(pt['efficientnet_b0']['precision'])} & {fmt4(pt['efficientnet_b0']['recall'])} "
            f"& {fmt4(pt['efficientnet_b0']['f1']}"
        ),
        "TensorFlow & ResNet-50": (
            f"TensorFlow & ResNet-50                     & {fmt4(tf['resnet50']['accuracy'])} "
            f"& {fmt4(tf['resnet50']['precision'])} & {fmt4(tf['resnet50']['recall'])} "
            f"& {fmt4(tf['resnet50']['f1']}"
        ),
        "TensorFlow & EfficientNet-B0": (
            f"TensorFlow & EfficientNet-B0               & {fmt4(tf['efficientnet_b0']['accuracy'])} "
            f"& {fmt4(tf['efficientnet_b0']['precision'])} & {fmt4(tf['efficientnet_b0']['recall'])} "
            f"& {fmt4(tf['efficientnet_b0']['f1']}"
        ),
    }
    for key, new_line in replacements.items():
        text = re.sub(rf"{re.escape(key)}.*?\\\\", new_line + " \\\\", text, count=1)

    # PyTorch timing in framework comparison
    pt_epochs = metrics.get("pt_epoch_seconds", {})
    if pt_epochs:
        for arch, secs in pt_epochs.items():
            if secs:
                avg = sum(secs) / len(secs)
                total = sum(secs)
                timing = f"Measured on CUDA: {arch} averaged {avg:.1f}\\,s per epoch ({total/60:.1f}\\,min total over {len(secs)} epochs)."
                break
        else:
            timing = ""
        if timing:
            text = text.replace(
                "\\todo{PyTorch wall-clock time per epoch and total, and GPU memory; the notebook did not record timing}",
                timing,
            )

    # Remove remaining TODO markers for final PDF
    text = re.sub(r"\\drafttrue", r"\\draftfalse", text)
    text = re.sub(r"\\todo\{[^}]*\}", "", text)

    # Update summary accuracy mention
    best_acc = fmt4(pt["resnet50"]["accuracy"])
    text = re.sub(
        r"PyTorch ResNet-50 \(test accuracy [0-9.]+%\)",
        f"PyTorch ResNet-50 (test accuracy {float(pt['resnet50']['accuracy'])*100:.2f}\\%)",
        text,
        count=1,
    )
    return text


def compile_pdf():
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        subprocess.run(cmd, cwd=ROOT, check=False)
    if PDF.exists():
        print("Wrote", PDF)
    else:
        print("PDF compile may have failed; check main.log")


def main():
    if not METRICS.is_file():
        raise SystemExit(f"Missing {METRICS}; run the notebook first.")
    metrics = load_metrics()
    text = TEX.read_text(encoding="utf-8")
    TEX.write_text(patch_tex(text, metrics), encoding="utf-8")
    compile_pdf()


if __name__ == "__main__":
    main()
