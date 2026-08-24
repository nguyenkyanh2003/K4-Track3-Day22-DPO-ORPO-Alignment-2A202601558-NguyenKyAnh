"""CPU-only smoke tests — run without a GPU (no torch/unsloth/trl import).

These guard the lab source against the most common breakages so `make test`
is a real gate, not a no-op:
- every notebook/script file exists and is valid Python (catches syntax errors)
- the TRL trainer calls use `processing_class=` (TRL >= 0.13), NOT the removed
  `tokenizer=` arg — the regression that broke NB1/NB3 on the resolved trl 0.19.x

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "01_sft_mini", "02_preference_data", "03_dpo_train",
    "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
]


def test_notebooks_exist_and_parse():
    for nb in NOTEBOOKS:
        p = REPO / "notebooks" / f"{nb}.py"
        assert p.exists(), f"missing notebook {p}"
        ast.parse(p.read_text(encoding="utf-8"))  # SyntaxError if broken


def test_scripts_parse():
    for p in (REPO / "scripts").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))


def test_colab_notebooks_are_valid_json():
    for p in (REPO / "colab").glob("*.ipynb"):
        notebook = json.loads(p.read_text(encoding="utf-8"))  # ValueError if corrupt
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            # IPython shell/magic cells are valid in Colab but not Python AST.
            if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
                continue
            ast.parse(source, filename=f"{p.name}:cell-{index}")


def test_colab_contains_reviewed_runtime_fixes():
    common_required = [
        '"unsloth==2026.4.4"',
        '"transformers==4.57.6"',
        '"trl==0.19.1"',
        "lab22-run-fixed",
        "bkai-foundation-models/vi-alpaca",
        'get_chat_template(tokenizer, chat_template="qwen-2.5")',
        "Preference-label audit:",
        "force_use_ref_model=True",
        "ref_model=ref_model",
        "Released NB3 policy/reference models.",
        "lab22-results.zip",
        "READY TO SUBMIT:",
    ]
    for p in (REPO / "colab").glob("*.ipynb"):
        notebook = json.loads(p.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        missing = [marker for marker in common_required if marker not in source]
        assert not missing, f"{p.name} is stale; missing {missing}. Rebuild it."

        if "SAFE" not in p.name:
            optional_required = [
                "Loaded combined SFT+DPO adapter",
                "maximum_memory_usage=0.60",
                "Installing optional lm-eval benchmark dependency",
            ]
            missing = [marker for marker in optional_required if marker not in source]
            assert not missing, f"{p.name} is stale; missing optional fixes {missing}."


def test_safe_colab_contains_only_core_pipeline():
    p = REPO / "colab" / "Lab22_DPO_T4_SAFE.ipynb"
    assert p.exists(), "missing deadline-safe T4 notebook"
    source = p.read_text(encoding="utf-8")
    assert "NB4 — Compare and Eval" in source
    assert "NB5 — Merge + Deploy" not in source
    assert "NB6 — LLM Benchmark" not in source


def test_trainer_uses_processing_class_not_tokenizer():
    # TRL >= 0.13 removed the `tokenizer=` arg in favour of `processing_class=`.
    # With the requirements pin `trl>=0.12,<0.20` a fresh install resolves to
    # 0.19.x, where `DPOTrainer/SFTTrainer(tokenizer=...)` raises TypeError.
    targets = [
        "notebooks/01_sft_mini.py",
        "notebooks/03_dpo_train.py",
        "scripts/train_dpo.py",
        "colab/Lab22_DPO_T4.ipynb",
        "colab/Lab22_DPO_BigGPU.ipynb",
    ]
    offenders = [t for t in targets if "tokenizer=tokenizer" in (REPO / t).read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} still pass tokenizer=tokenizer to a TRL trainer; "
        f"use processing_class=tokenizer (tokenizer= removed in trl>=0.13)."
    )
