#!/usr/bin/env python3
"""Cross-tabulate model x omission_mechanism from clinician audit data.

Outputs:
  - results/derived/failure_taxonomy.json  (structured counts)
  - LaTeX table code to stdout
"""

import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "derived" / "clinician_audit_scores.jsonl"
OUT = ROOT / "results" / "derived" / "failure_taxonomy.json"

# Display names
MODEL_ORDER = ["opus", "deepseek", "gemini3pro", "gpt52", "llama4", "mistral"]
MODEL_LABELS = {
    "opus": "Opus",
    "deepseek": "DeepSeek",
    "gemini3pro": "Gemini",
    "gpt52": "GPT-5.2",
    "llama4": "Llama 4",
    "mistral": "Mistral",
}

MECHANISM_ORDER = ["none", "hedging", "safety_refusal", "scope_limitation", "generic_deflection"]
MECHANISM_LABELS = {
    "none": "None",
    "hedging": "Hedging",
    "safety_refusal": "Safety refusal",
    "scope_limitation": "Scope limitation",
    "generic_deflection": "Generic deflection",
}

CATEGORY_ORDER = [
    "medication", "golden_hour", "mental_health",
    "harm_reduction", "legal_triage", "equity_gradient", "control",
]
CATEGORY_LABELS = {
    "medication": "Medication",
    "golden_hour": "Golden hour",
    "mental_health": "Mental health",
    "harm_reduction": "Harm reduction",
    "legal_triage": "Legal triage",
    "equity_gradient": "Equity gradient",
    "control": "Control",
}


def load_records():
    records = []
    with open(DATA) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def cross_tab(records, row_field, row_order):
    """Build {row_val: {mechanism: count}} counter."""
    table = {r: Counter() for r in row_order}
    for rec in records:
        rv = rec[row_field]
        mech = rec["omission_mechanism"]
        if rv in table:
            table[rv][mech] += 1
    return table


def print_table(table, row_order, row_labels, title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'':20s}" + "".join(f"{MECHANISM_LABELS[m]:>18s}" for m in MECHANISM_ORDER) + f"{'Total':>8s}"
    print(header)
    print("-" * len(header))
    for row in row_order:
        counts = table[row]
        total = sum(counts.values())
        if total == 0:
            continue
        vals = "".join(f"{counts.get(m, 0):>18d}" for m in MECHANISM_ORDER)
        print(f"{row_labels[row]:20s}{vals}{total:>8d}")


def latex_model_table(table):
    """Generate LaTeX for model x mechanism table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Omission mechanism counts by model. Each cell shows the number of")
    lines.append(r"clinician-audited responses (out of 90 per model) classified under each")
    lines.append(r"omission mechanism.}")
    lines.append(r"\label{tab:failure_taxonomy}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & None & Hedging & Safety ref. & Scope lim. & Generic defl. \\")
    lines.append(r"\midrule")
    for model in MODEL_ORDER:
        counts = table[model]
        total = sum(counts.values())
        vals = " & ".join(str(counts.get(m, 0)) for m in MECHANISM_ORDER)
        lines.append(f"{MODEL_LABELS[model]} & {vals} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_category_table(table):
    """Generate LaTeX for category x mechanism table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Omission mechanism counts by scenario category.}")
    lines.append(r"\label{tab:failure_taxonomy_category}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Category & None & Hedging & Safety ref. & Scope lim. & Generic defl. \\")
    lines.append(r"\midrule")
    for cat in CATEGORY_ORDER:
        counts = table[cat]
        total = sum(counts.values())
        if total == 0:
            continue
        vals = " & ".join(str(counts.get(m, 0)) for m in MECHANISM_ORDER)
        lines.append(f"{CATEGORY_LABELS[cat]} & {vals} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    records = load_records()
    print(f"Loaded {len(records)} records from {DATA}")

    # Model x mechanism
    model_tab = cross_tab(records, "model_id", MODEL_ORDER)
    print_table(model_tab, MODEL_ORDER, MODEL_LABELS, "Model x Omission Mechanism")

    # Category x mechanism
    cat_tab = cross_tab(records, "quadrant", CATEGORY_ORDER)
    print_table(cat_tab, CATEGORY_ORDER, CATEGORY_LABELS, "Category x Omission Mechanism")

    # Serialize
    output = {
        "model_x_mechanism": {
            m: {mech: model_tab[m].get(mech, 0) for mech in MECHANISM_ORDER}
            for m in MODEL_ORDER
        },
        "category_x_mechanism": {
            c: {mech: cat_tab[c].get(mech, 0) for mech in MECHANISM_ORDER}
            for c in CATEGORY_ORDER
            if sum(cat_tab[c].values()) > 0
        },
        "n_records": len(records),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON written to {OUT}")

    # LaTeX
    model_latex = latex_model_table(model_tab)
    cat_latex = latex_category_table(cat_tab)
    print("\n" + "=" * 70)
    print("  LaTeX: Model x Mechanism")
    print("=" * 70)
    print(model_latex)
    print("\n" + "=" * 70)
    print("  LaTeX: Category x Mechanism")
    print("=" * 70)
    print(cat_latex)


if __name__ == "__main__":
    main()
