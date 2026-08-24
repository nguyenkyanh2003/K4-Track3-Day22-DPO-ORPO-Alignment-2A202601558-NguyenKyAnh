#!/usr/bin/env python3
"""Build the stitched Colab notebooks from the six Jupytext percent sources.

Run from the repository root after changing any file in ``notebooks/``:

    python scripts/build_colab_notebooks.py

The generated notebooks are intentionally output-free. Students execute one of
them in Colab and save the executed copy for submission.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CORE_SOURCES = [
    REPO / "notebooks" / "01_sft_mini.py",
    REPO / "notebooks" / "02_preference_data.py",
    REPO / "notebooks" / "03_dpo_train.py",
    REPO / "notebooks" / "04_compare_and_eval.py",
]
OPTIONAL_SOURCES = [
    REPO / "notebooks" / "05_merge_deploy_gguf.py",
    REPO / "notebooks" / "06_benchmark.py",
]


def source_lines(text: str) -> list[str]:
    """Return nbformat-compatible source lines with newline terminators."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    return lines


def cell(cell_type: str, text: str) -> dict:
    result = {"cell_type": cell_type, "metadata": {}, "source": source_lines(text)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def parse_percent(path: Path) -> list[dict]:
    """Parse the small subset of Jupytext percent syntax used by this lab."""
    cells: list[dict] = []
    kind: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if kind is None:
            body = []
            return
        text = "".join(body)
        if kind == "markdown":
            converted = []
            for line in text.splitlines(keepends=True):
                if line.startswith("# "):
                    converted.append(line[2:])
                elif line in {"#\n", "#\r\n", "#"}:
                    converted.append("\n")
                else:
                    converted.append(line)
            text = "".join(converted)
        cells.append(cell(kind, text))
        body = []

    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        elif kind is not None:
            body.append(line)
    flush()
    return cells


def setup_cells(tier: str, *, core_only: bool = False) -> list[dict]:
    model = "Qwen2.5-3B" if tier == "T4" else "Qwen2.5-7B"
    scope = "CORE NB1-NB4" if core_only else "NB1-NB6"
    pipeline = (
        "SFT → preference data → DPO → evaluation → verify → ZIP"
        if core_only
        else "SFT → preference data → DPO → evaluation → GGUF → benchmark"
    )
    return [
        cell(
            "markdown",
            f"# Lab 22 — DPO/ORPO Alignment ({tier} tier)\n\n"
            f"One-click pipeline: {pipeline}.\n\n"
            "**Sinh viên:** Nguyễn Kỳ Anh · **MSSV:** 2A202601558\n\n"
            f"**Tier:** {tier} · **Model:** {model} · **Run-all scope:** {scope}.\n",
        ),
        cell("markdown", "## A. Clean Colab setup\n"),
        cell(
            "code",
            "import os\n"
            f"os.environ['COMPUTE_TIER'] = '{tier}'\n"
            "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
            "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
            "print('COMPUTE_TIER =', os.environ['COMPUTE_TIER'])\n",
        ),
        cell(
            "code",
            "# Clean, mutually compatible Colab stack. transformers 5.x breaks\n"
            "# Unsloth merged export; pin 4.57.6 before anything imports it.\n"
            "# Gradio is unused here and conflicts with the transformers 4.x hub pin.\n"
            "!pip uninstall -y -q gradio gradio_client 2>/dev/null || true\n"
            "!pip install -q --upgrade-strategy only-if-needed \\\n"
            "  \"unsloth==2026.4.4\" \"transformers==4.57.6\" \"trl==0.19.1\" \\\n"
            "  \"peft>=0.18,<1.0\" \"bitsandbytes>=0.45.5,<1.0\" \\\n"
            "  \"datasets>=3.4.1,<4.0\" \"accelerate>=1.1,<2.0\" \\\n"
            "  \"matplotlib>=3.9,<4.0\" \"pandas>=2.2,<3.0\" \"pyarrow>=17,<22\"\n",
        ),
        cell(
            "code",
            "# Fail fast before training if Colab ever changes its base image.\n"
            "import transformers, trl, peft\n"
            "print('transformers:', transformers.__version__)\n"
            "print('trl:', trl.__version__)\n"
            "print('peft:', peft.__version__)\n"
            "assert transformers.__version__ == '4.57.6', transformers.__version__\n"
            "assert trl.__version__ == '0.19.1', trl.__version__\n",
        ),
        cell(
            "code",
            "import torch\n"
            "assert torch.cuda.is_available(), 'Runtime → Change runtime type → T4 GPU'\n"
            "gpu = torch.cuda.get_device_properties(0)\n"
            "print(f'GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)')\n",
        ),
        cell(
            "code",
            "# Re-clone on every Run-all so an older failed run cannot leak stale\n"
            "# artifacts into the new submission.\n"
            "import os, shutil, subprocess\n"
            "from pathlib import Path\n"
            "REPO_URL = os.environ.get(\n"
            "    'LAB_REPO_URL',\n"
            "    'https://github.com/nguyenkyanh2003/K4-Track3-Day22-DPO-ORPO-Alignment-2A202601558-NguyenKyAnh.git',\n"
            ")\n"
            "WORK = Path('/content/lab22-run-fixed')\n"
            "os.chdir('/content')\n"
            "if WORK.exists():\n"
            "    shutil.rmtree(WORK)\n"
            "subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(WORK)], check=True)\n"
            "os.chdir(WORK)\n"
            "print('Working directory:', Path.cwd())\n",
        ),
    ]


def final_cells() -> list[dict]:
    return [
        cell("markdown", "# Final — artifact summary and lightweight results bundle\n"),
        cell(
            "code",
            "from pathlib import Path\n"
            "import subprocess, sys, zipfile\n"
            "root = Path.cwd()\n"
            "expected = [\n"
            "    root/'adapters/sft-mini/adapter_config.json',\n"
            "    root/'adapters/dpo/adapter_config.json',\n"
            "    root/'adapters/dpo/dpo_metrics.json',\n"
            "    root/'data/pref/train.parquet',\n"
            "    root/'data/eval/side_by_side.jsonl',\n"
            "    root/'data/eval/judge_results.json',\n"
            "]\n"
            "for p in expected:\n"
            "    print(('OK      ' if p.exists() else 'MISSING '), p.relative_to(root))\n"
            "bundle = Path('/content/lab22-results.zip')\n"
            "include = [root/'data/pref', root/'data/eval', root/'submission/screenshots']\n"
            "with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
            "    for folder in include:\n"
            "        if folder.exists():\n"
            "            for p in folder.rglob('*'):\n"
            "                if p.is_file(): zf.write(p, p.relative_to(root))\n"
            "    for p in [root/'adapters/sft-mini/adapter_config.json',\n"
            "              root/'adapters/dpo/adapter_config.json',\n"
            "              root/'adapters/dpo/dpo_metrics.json']:\n"
            "        if p.exists(): zf.write(p, p.relative_to(root))\n"
            "print('Results bundle:', bundle, f'({bundle.stat().st_size/1024**2:.1f} MiB)')\n"
            "verify = subprocess.run([sys.executable, 'scripts/verify.py'], text=True, capture_output=True)\n"
            "print('\\nVERIFY OUTPUT\\n' + verify.stdout)\n"
            "if verify.stderr: print(verify.stderr)\n"
            "assert verify.returncode == 0, 'Submission verification failed; inspect VERIFY OUTPUT above.'\n"
            "print('READY TO SUBMIT:', bundle)\n",
        ),
        cell(
            "code",
            "# Colab starts the browser download after a successful verified run.\n"
            "from google.colab import files\n"
            "files.download('/content/lab22-results.zip')\n",
        ),
    ]


def build(tier: str, *, core_only: bool = False) -> dict:
    cells = setup_cells(tier, core_only=core_only)
    sources = CORE_SOURCES if core_only else CORE_SOURCES + OPTIONAL_SOURCES
    for source in sources:
        cells.extend(parse_percent(source))
    cells.extend(final_cells())
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4" if tier == "T4" else "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    targets = {
        ("T4", False): REPO / "colab" / "Lab22_DPO_T4.ipynb",
        ("T4", True): REPO / "colab" / "Lab22_DPO_T4_SAFE.ipynb",
        ("BIGGPU", False): REPO / "colab" / "Lab22_DPO_BigGPU.ipynb",
    }
    for (tier, core_only), target in targets.items():
        target.write_text(
            json.dumps(build(tier, core_only=core_only), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"Wrote {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
