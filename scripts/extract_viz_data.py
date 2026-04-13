#!/usr/bin/env python3
"""Extract and aggregate all data needed for GitHub Pages visualizations.
Outputs a single JSON file: docs/viz_data.json
"""

import json
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "scenarios"
RESULTS = ROOT / "results"

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_scenarios():
    """Load all scenario JSON files."""
    scenarios = []
    for subdir in sorted(DATA.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.glob("*.json")):
            with open(f) as fh:
                s = json.load(fh)
                scenarios.append(s)
    return scenarios

def category_label(quadrant):
    labels = {
        "golden_hour": "Golden Hour",
        "diagnostic_red_flags": "Diagnostic Red Flags",
        "harm_reduction": "Harm Reduction",
        "alarm_fatigue": "Alarm Fatigue",
        "control": "Control",
        "mental_health": "Mental Health",
        "medication": "Medication",
        "legal_triage": "Legal Triage",
        "paediatric": "Paediatric",
        "equity_gradient": "Equity Gradient",
    }
    return labels.get(quadrant, quadrant)

def main():
    # Load scenarios
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    # Load audit scores
    audit_path = RESULTS / "derived" / "clinician_audit_scores.jsonl"
    audit_scores = load_jsonl(audit_path)
    print(f"Loaded {len(audit_scores)} audit scores")

    # Load judge scores
    judge_path = RESULTS / "raw" / "judge_responses.jsonl"
    judge_scores = load_jsonl(judge_path)
    print(f"Loaded {len(judge_scores)} judge scores")

    # Load paper numbers
    with open(RESULTS / "derived" / "paper_numbers.json") as f:
        paper_numbers = json.load(f)

    # Build scenario metadata
    scenario_meta = {}
    for s in scenarios:
        sid = s["id"]
        scenario_meta[sid] = {
            "id": sid,
            "category": category_label(s.get("quadrant", "")),
            "quadrant": s.get("quadrant", ""),
            "prompt": s.get("prompt", ""),
            "gold_standard": s.get("gold_standard_response", "")[:300],  # truncated for page weight
            "critical_actions": s.get("critical_actions", []),
            "acuity_weight": s.get("acuity_weight", 1.0),
            "decoupling": s.get("decoupling_variant"),
        }

    # Compute per-scenario per-model audit scores
    # Group: scenario_id -> model_id -> list of OH/CH scores
    audit_by_sm = defaultdict(lambda: defaultdict(list))
    for r in audit_scores:
        sid = r.get("scenario_id")
        mid = r.get("model_id")
        if sid and mid:
            audit_by_sm[sid][mid].append({
                "oh": r.get("omission_harm", 0),
                "ch": r.get("commission_harm", 0),
                "response_class": r.get("response_class", ""),
                "omission_mechanism": r.get("omission_mechanism", ""),
            })

    # Also group judge scores
    judge_by_sm = defaultdict(lambda: defaultdict(list))
    for r in judge_scores:
        sid = r.get("scenario_id")
        mid = r.get("model_id")
        if sid and mid:
            judge_by_sm[sid][mid].append({
                "oh": r.get("omission_harm", 0),
                "ch": r.get("commission_harm", 0),
            })

    # Build per-scenario summary
    model_ids = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
    model_labels = {
        "opus": "Claude Opus 4.6",
        "gpt52": "GPT-5.2",
        "gemini3pro": "Gemini 3 Pro",
        "llama4": "Llama 4 Maverick",
        "deepseek": "DeepSeek V3.2",
        "mistral": "Mistral Large",
    }

    scenario_scores = {}
    for sid, meta in scenario_meta.items():
        scores = {}
        for mid in model_ids:
            audit_list = audit_by_sm.get(sid, {}).get(mid, [])
            judge_list = judge_by_sm.get(sid, {}).get(mid, [])

            if audit_list:
                mean_oh = max(0, sum(r["oh"] for r in audit_list) / len(audit_list))
                mean_ch = max(0, sum(r["ch"] for r in audit_list) / len(audit_list))
                scores[mid] = {"oh": round(mean_oh, 2), "ch": round(mean_ch, 2), "n": len(audit_list)}
            elif judge_list:
                mean_oh = max(0, sum(r["oh"] for r in judge_list) / len(judge_list))
                mean_ch = max(0, sum(r["ch"] for r in judge_list) / len(judge_list))
                scores[mid] = {"oh": round(mean_oh, 2), "ch": round(mean_ch, 2), "n": len(judge_list), "source": "judge"}

        scenario_scores[sid] = scores

    # Build individual OH scores for distribution plots
    # Use judge scores (full 60-scenario coverage) as primary
    model_oh_scores = defaultdict(list)
    model_ch_scores = defaultdict(list)
    for r in judge_scores:
        mid = r.get("model_id")
        if mid:
            model_oh_scores[mid].append(max(0, r.get("omission_harm", 0)))
            model_ch_scores[mid].append(max(0, r.get("commission_harm", 0)))

    # Build per-category per-model means (for heatmap)
    # Use judge scores for full coverage
    cat_model_oh = defaultdict(lambda: defaultdict(list))
    for sid, meta in scenario_meta.items():
        cat = meta["category"]
        for mid in model_ids:
            if sid in judge_by_sm and mid in judge_by_sm[sid]:
                for r in judge_by_sm[sid][mid]:
                    cat_model_oh[cat][mid].append(r["oh"])

    category_heatmap = {}
    for cat in sorted(cat_model_oh.keys()):
        category_heatmap[cat] = {}
        for mid in model_ids:
            vals = cat_model_oh[cat][mid]
            if vals:
                category_heatmap[cat][mid] = round(sum(vals) / len(vals), 2)
            else:
                category_heatmap[cat][mid] = None

    # Build decoupling pair data
    # Find paired scenarios
    pairs = {}
    for sid, meta in scenario_meta.items():
        dv = meta.get("decoupling")
        if dv and dv.get("pair_id"):
            pair_id = dv["pair_id"]
            framing = dv["framing"]
            if pair_id not in pairs:
                pairs[pair_id] = {}
            pairs[pair_id][framing] = sid

    decoupling_data = []
    for pair_id, framings in sorted(pairs.items()):
        if "layperson" not in framings or "physician" not in framings:
            continue
        lay_sid = framings["layperson"]
        phys_sid = framings["physician"]

        pair_entry = {
            "pair_id": pair_id,
            "lay_scenario": lay_sid,
            "phys_scenario": phys_sid,
            "category": scenario_meta.get(lay_sid, {}).get("category", ""),
            "models": {}
        }

        for mid in model_ids:
            # Prefer audit scores, fall back to judge scores
            lay_scores = audit_by_sm.get(lay_sid, {}).get(mid, [])
            phys_scores = audit_by_sm.get(phys_sid, {}).get(mid, [])
            source = "audit"
            if not lay_scores or not phys_scores:
                lay_scores = judge_by_sm.get(lay_sid, {}).get(mid, [])
                phys_scores = judge_by_sm.get(phys_sid, {}).get(mid, [])
                source = "judge"

            if lay_scores and phys_scores:
                lay_mean = sum(r["oh"] for r in lay_scores) / len(lay_scores)
                phys_mean = sum(r["oh"] for r in phys_scores) / len(phys_scores)
                pair_entry["models"][mid] = {
                    "lay_oh": round(lay_mean, 2),
                    "phys_oh": round(phys_mean, 2),
                    "gap": round(lay_mean - phys_mean, 2),
                    "source": source,
                }

        decoupling_data.append(pair_entry)

    # Assemble output
    output = {
        "scenarios": [
            {
                **{k: v for k, v in scenario_meta[sid].items() if k != "decoupling"},
                "has_decoupling": scenario_meta[sid].get("decoupling") is not None,
                "pair_id": (scenario_meta[sid].get("decoupling") or {}).get("pair_id"),
                "framing": (scenario_meta[sid].get("decoupling") or {}).get("framing"),
                "scores": scenario_scores.get(sid, {}),
            }
            for sid in sorted(scenario_meta.keys())
        ],
        "model_ids": model_ids,
        "model_labels": model_labels,
        "model_oh_distributions": {mid: model_oh_scores[mid] for mid in model_ids},
        "model_ch_distributions": {mid: model_ch_scores[mid] for mid in model_ids},
        "h1": {mid: {
            "mean_oh": paper_numbers["h1_systemic_oh"][mid]["mean_oh"],
            "median_oh": paper_numbers["h1_systemic_oh"][mid]["median_oh"],
            "mean_ch": paper_numbers["h1_systemic_oh"][mid]["mean_ch"],
        } for mid in model_ids},
        "h2": {
            "overall_gap": paper_numbers["h2_decoupling"]["overall_mean_gap_excl_gpt52"],
            "per_model": paper_numbers["h2_decoupling"]["per_model"],
        },
        "category_heatmap": category_heatmap,
        "decoupling_pairs": decoupling_data,
        "categories": sorted(set(m["category"] for m in scenario_meta.values())),
    }

    out_path = ROOT / "docs" / "viz_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
