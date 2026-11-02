"""Multi-Model Benchmark Comparison Table Generator for Scene Change Detection (SCD).

Scans an evaluation results directory (e.g., './eval_results_real') for per-model
'metrics.json' artifacts and compiles unified multi-model comparison tables
(Markdown + ASCII text) grouped by dataset ('pcd', 'changesim', 'vlcmu').
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple


MODEL_DISPLAY_NAMES = {
    "paradis_scd": "ParaDis-SCD (Ours)",
    "paradis_scd_no_dbaia": "w/o D-BAIA (Illum Align)",
    "paradis_scd_no_desg": "w/o D-ESG (Epipolar Gauge)",
    "paradis_scd_no_ecdca": "w/o EC-DCA (Deform Attn)",
    "paradis_scd_no_pdag": "w/o P-DAG (Disocc Gate)",
    "c3po": "C-3PO (CVPR 2023)",
    "cgnet": "CGNet (J-STARS 2023)",
    "ussfcnet": "USSFC-Net (TGRS 2023)",
    "changer": "Changer (ICCV 2023)",
    "csnet": "CSNet (ICRA 2021)",
    "changeformer": "ChangeFormer (J-STARS 2022)",
    "dr_tanet": "DR-TANet (TIP 2020)",
}

MODEL_ORDER = [
    "paradis_scd",
    "paradis_scd_no_dbaia",
    "paradis_scd_no_desg",
    "paradis_scd_no_ecdca",
    "paradis_scd_no_pdag",
    "c3po",
    "cgnet",
    "ussfcnet",
    "changer",
    "csnet",
    "changeformer",
    "dr_tanet",
]

DATASET_DISPLAY_NAMES = {
    "pcd": "PCD (TSUNAMI Panoramic)",
    "changesim": "ChangeSim (Indoor Warehouse)",
    "vlcmu": "VL-CMU-CD (Autonomous Driving Street-View)",
}


def generate_comparison_tables(eval_root: str, target_dataset: Optional[str] = None) -> Tuple[str, str]:
    """Aggregates all <dataset>_<model>/metrics.json files under eval_root into comparison tables.

    Returns:
        (txt_report, md_report) strings and saves them to eval_root.
    """
    if not os.path.isdir(eval_root):
        return "", ""

    json_files = sorted(glob.glob(os.path.join(eval_root, "*", "metrics.json")))
    if not json_files:
        return "", ""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            ds = str(data.get("dataset", "unknown")).lower()
            if "metrics" in data and isinstance(data["metrics"], dict):
                grouped.setdefault(ds, []).append(data)
        except Exception:
            continue

    if not grouped:
        return "", ""

    def _model_rank(entry: Dict[str, Any]) -> Tuple[int, str]:
        m = str(entry.get("model", "")).lower()
        idx = MODEL_ORDER.index(m) if m in MODEL_ORDER else 999
        return (idx, m)

    txt_lines: List[str] = []
    md_lines: List[str] = [
        "# Scene Change Detection (SCD) Multi-Model Benchmark Comparison",
        "",
    ]

    datasets_to_show = ["pcd", "changesim", "vlcmu"]
    for ds_key in grouped:
        if ds_key not in datasets_to_show:
            datasets_to_show.append(ds_key)

    for ds_key in datasets_to_show:
        if ds_key not in grouped:
            continue
        entries = sorted(grouped[ds_key], key=_model_rank)
        ds_title = DATASET_DISPLAY_NAMES.get(ds_key, ds_key.upper())
        res = entries[0].get("resolution", ["?", "?"])
        res_str = f"{res[0]}x{res[1]}" if isinstance(res, list) and len(res) == 2 else str(res)

        # ASCII Table
        sep = "=" * 106
        dash = "-" * 106
        txt_lines.append(sep)
        txt_lines.append(f"BENCHMARK COMPARISON TABLE: {ds_title} (Resolution: {res_str})")
        txt_lines.append(sep)
        header = (
            f"{'Model':<28} | {'F1 (%)':>8} | {'mIoU (%)':>8} | {'ChgIoU':>8} | "
            f"{'StcIoU':>8} | {'OA (%)':>8} | {'Prec (%)':>8} | {'Rec (%)':>8}"
        )
        txt_lines.append(header)
        txt_lines.append(dash)

        # Markdown Table
        md_lines.append(f"## {ds_title} (`{res_str}`)")
        md_lines.append("")
        md_lines.append(
            "| Model | F1 (%) | mIoU (%) | Change IoU (%) | Static IoU (%) | OA (%) | Precision (%) | Recall (%) |"
        )
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for item in entries:
            m_raw = str(item.get("model", "unknown"))
            m_name = MODEL_DISPLAY_NAMES.get(m_raw, m_raw)
            met = item["metrics"]
            f1 = met.get("f1", 0.0) * 100.0
            miou = met.get("miou", 0.0) * 100.0
            ciou = met.get("change_iou", 0.0) * 100.0
            siou = met.get("static_iou", 0.0) * 100.0
            oa = met.get("oa", 0.0) * 100.0
            prec = met.get("precision", 0.0) * 100.0
            rec = met.get("recall", 0.0) * 100.0

            txt_lines.append(
                f"{m_name:<28} | {f1:>8.2f} | {miou:>8.2f} | {ciou:>8.2f} | "
                f"{siou:>8.2f} | {oa:>8.2f} | {prec:>8.2f} | {rec:>8.2f}"
            )
            md_lines.append(
                f"| **{m_name}** | {f1:.2f} | {miou:.2f} | {ciou:.2f} | "
                f"{siou:.2f} | {oa:.2f} | {prec:.2f} | {rec:.2f} |"
            )

        txt_lines.append(sep)
        txt_lines.append("")
        md_lines.append("")

    txt_report = "\n".join(txt_lines)
    md_report = "\n".join(md_lines)

    txt_out = os.path.join(eval_root, "benchmark_comparison_table.txt")
    md_out = os.path.join(eval_root, "benchmark_comparison_table.md")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(txt_report + "\n")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_report + "\n")

    return txt_report, md_report


def main():
    parser = argparse.ArgumentParser(description="Compile multi-model SCD comparison tables.")
    parser.add_argument(
        "--eval_root",
        type=str,
        default="./eval_results_real",
        help="Root directory containing <dataset>_<model>/metrics.json subdirectories",
    )
    args = parser.parse_args()
    txt_report, _ = generate_comparison_tables(args.eval_root)
    if txt_report:
        print(txt_report)
        print(f"Saved comparison tables to {args.eval_root}/benchmark_comparison_table.{{txt,md}}")
    else:
        print(f"No metrics.json files found under {args.eval_root}")


if __name__ == "__main__":
    main()
