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
SOURCES = [
    REPO / "notebooks" / "01_sft_mini.py",
    REPO / "notebooks" / "02_preference_data.py",
    REPO / "notebooks" / "03_dpo_train.py",
    REPO / "notebooks" / "04_compare_and_eval.py",
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


def setup_cells(tier: str) -> list[dict]:
    model = "Qwen2.5-3B" if tier == "T4" else "Qwen2.5-7B"
    return [
        cell(
            "markdown",
            f"# Lab 22 — DPO/ORPO Alignment ({tier} tier)\n\n"
            f"One-click pipeline: SFT → preference data → DPO → evaluation → GGUF → benchmark.\n\n"
            "**Sinh viên:** Nguyễn Kỳ Anh · **MSSV:** 2A202601558\n\n"
            f"**Tier:** {tier} · **Model:** {model} · generated from the reviewed Jupytext sources.\n",
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
            "# Install one mutually compatible stack. The final pin fixes the Colab\n"
            "# omegaconf/antlr resolver conflict seen with lm-eval.\n"
            "!pip install -q \\\n"
            "  \"unsloth>=2025.10,<2026.5\" \"trl>=0.12,<0.20\" \"peft>=0.13,<1.0\" \\\n"
            "  \"bitsandbytes>=0.44,<1.0\" \"datasets>=3.1,<4.0\" \"accelerate>=1.1,<2.0\" \\\n"
            "  \"llama-cpp-python>=0.3,<1.0\" \"lm-eval[ifeval,math]>=0.4.5,<1.0\" \\\n"
            "  \"matplotlib>=3.9,<4.0\" \"pandas>=2.2,<3.0\" \"pyarrow>=17,<22\" \\\n"
            "  \"openai>=1.55,<2.0\" \"anthropic>=0.40,<1.0\"\n"
            "!pip install -q --force-reinstall \"antlr4-python3-runtime==4.9.3\"\n",
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
            "# Work inside a clean clone so outputs follow the submission layout.\n"
            "import os, subprocess\n"
            "from pathlib import Path\n"
            "REPO_URL = os.environ.get(\n"
            "    'LAB_REPO_URL',\n"
            "    'https://github.com/nguyenkyanh2003/K4-Track3-Day22-DPO-ORPO-Alignment-2A202601558-NguyenKyAnh.git',\n"
            ")\n"
            "WORK = Path('/content/lab22-run')\n"
            "if not (WORK / '.git').exists():\n"
            "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(WORK)], check=True)\n"
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
            "import zipfile\n"
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
            "print('Results bundle:', bundle, f'({bundle.stat().st_size/1024**2:.1f} MiB)')\n",
        ),
    ]


def build(tier: str) -> dict:
    cells = setup_cells(tier)
    for source in SOURCES:
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
        "T4": REPO / "colab" / "Lab22_DPO_T4.ipynb",
        "BIGGPU": REPO / "colab" / "Lab22_DPO_BigGPU.ipynb",
    }
    for tier, target in targets.items():
        target.write_text(json.dumps(build(tier), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
