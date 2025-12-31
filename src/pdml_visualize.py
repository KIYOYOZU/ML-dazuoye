#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization for PD-ML weak predictions (gamma_dot, H).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = {
    "predictions_csv": str(BASE_DIR / "results" / "pdml_weak" / "test_predictions.csv"),
    "output_dir": str(BASE_DIR / "results" / "figures"),
}


def _scatter_plot(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, edgecolors="none")
    if len(y_true) > 0:
        min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
        max_v = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.grid(True, alpha=0.3, linestyle="--")


def main():
    df = pd.read_csv(CONFIG["predictions_csv"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gamma dot scatter
    if "gamma_pred" in df.columns and "gamma_dot_smooth" in df.columns:
        mask = df["gamma_pred"].notna()
        y_true = df.loc[mask, "gamma_dot_smooth"].to_numpy(dtype=float)
        y_pred = df.loc[mask, "gamma_pred"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        _scatter_plot(ax, y_true, y_pred, "Gamma dot (true vs pred)")
        out_path = output_dir / "pdml_gamma_scatter.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path.name}")

    # H scatter (raw)
    if "H_pred" in df.columns and "hardening_modulus_clip" in df.columns:
        mask = df["H_pred"].notna()
        y_true = df.loc[mask, "hardening_modulus_clip"].to_numpy(dtype=float)
        y_pred = df.loc[mask, "H_pred"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        _scatter_plot(ax, y_true, y_pred, "Hardening modulus (true vs pred)")
        out_path = output_dir / "pdml_h_scatter.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path.name}")

    # H scatter (log)
    if "H_pred_log" in df.columns and "hardening_modulus_log" in df.columns:
        mask = df["H_pred_log"].notna()
        y_true = df.loc[mask, "hardening_modulus_log"].to_numpy(dtype=float)
        y_pred = df.loc[mask, "H_pred_log"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        _scatter_plot(ax, y_true, y_pred, "Hardening modulus log10 (true vs pred)")
        out_path = output_dir / "pdml_h_log_scatter.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path.name}")


if __name__ == "__main__":
    main()
