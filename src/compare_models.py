#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare stress prediction models (LSTM/PINN vs GBR baseline).
Optionally records weak PD-ML gamma/H metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CONFIG = {
    "lstm_metrics": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\metrics\test_metrics.json",
    "gbr_metrics": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\baseline_gbr\metrics.json",
    "pdml_weak_metrics": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\pdml_weak\metrics.json",
    "output_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\model_comparison.csv",
    "output_fig": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\model_comparison.png",
    "output_pdml_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\pdml_weak_comparison.csv",
}


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []

    lstm = _load_json(CONFIG["lstm_metrics"])
    if lstm:
        rows.append({
            "model": "LSTM_PINN",
            "r2": lstm.get("r2"),
            "rmse": lstm.get("rmse"),
            "mae": lstm.get("mae"),
            "samples": lstm.get("samples"),
        })

    gbr = _load_json(CONFIG["gbr_metrics"])
    if gbr and "overall" in gbr and "test" in gbr["overall"]:
        t = gbr["overall"]["test"]
        rows.append({
            "model": "GBR_baseline",
            "r2": t.get("r2"),
            "rmse": t.get("rmse"),
            "mae": t.get("mae"),
            "samples": t.get("n"),
        })

    out_path = Path(CONFIG["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    if not df.empty:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        df_plot = df.set_index("model")
        df_plot["r2"].plot(kind="bar", ax=ax1, color="#3B82F6", alpha=0.85)
        ax1.set_ylabel("R2")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax1.set_title("Stress Prediction Comparison")
        plt.tight_layout()
        plt.savefig(CONFIG["output_fig"], dpi=300, bbox_inches="tight")
        plt.close()

    # Optional weak PD-ML metrics
    pdml = _load_json(CONFIG["pdml_weak_metrics"])
    if pdml:
        rows_pdml = []
        for key in ("gamma", "H"):
            if key in pdml and "test" in pdml[key]:
                t = pdml[key]["test"]
                rows_pdml.append({
                    "target": key,
                    "r2": t.get("r2"),
                    "rmse": t.get("rmse"),
                    "mae": t.get("mae"),
                    "samples": t.get("n"),
                })
        if rows_pdml:
            pd.DataFrame(rows_pdml).to_csv(CONFIG["output_pdml_csv"], index=False)

    print("comparison saved:", str(out_path))
    print("figure saved:", CONFIG["output_fig"])
    if Path(CONFIG["output_pdml_csv"]).exists():
        print("pdml metrics saved:", CONFIG["output_pdml_csv"])


if __name__ == "__main__":
    main()
