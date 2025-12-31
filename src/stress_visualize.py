#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stress-strain visualization with LSTM and GBR predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
from matplotlib import colormaps
from matplotlib.colors import Normalize

import sys
sys.path.append(str(Path(__file__).parent))
from stress_train import ImprovedStressStrainLSTM, StressStrainDataset


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = {
    "model_path": str(BASE_DIR / "models" / "lstm_physics_best.pth"),
    "data_dir": str(BASE_DIR / "data" / "processed"),
    "gbr_predictions": str(BASE_DIR / "results" / "baseline_gbr" / "all_predictions.csv"),
    "gbr_fallback_predictions": str(BASE_DIR / "results" / "baseline_gbr" / "test_predictions.csv"),
    "output_dir": str(BASE_DIR / "results" / "figures"),
    "clear_output_dir": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_lstm_model(model_path: str, device: torch.device) -> ImprovedStressStrainLSTM:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = ImprovedStressStrainLSTM(
        seq_len=config["seq_len"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def prepare_output_dir(output_dir: str, clear_output_dir: bool) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_output_dir:
        removed = 0
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
                removed += 1
        print(f"  cleared: {output_dir} ({removed} items)")
    return output_dir


def predict_lstm_results(model, dataset, device):
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            strain_seq = sample["strain_seq"].unsqueeze(0).to(device)
            conditions = sample["conditions"].unsqueeze(0).to(device)
            pred = model(strain_seq, conditions)

            pred_np = pred[0].cpu().numpy()
            target_np = sample["stress_seq"].numpy()
            scaler = sample["stress_scaler"]
            pred_denorm = scaler.inverse_transform(pred_np).flatten()
            target_denorm = scaler.inverse_transform(target_np).flatten()

            results.append({
                "material": sample["material"],
                "temperature": sample["temperature"],
                "strain_rate": sample["strain_rate"],
                "strain": sample["strain_seq_raw"],
                "stress_true": target_denorm,
                "stress_pred": pred_denorm,
            })
    return results


def build_gbr_results(pred_path: str, fallback_path: str):
    path = Path(pred_path)
    if not path.exists():
        path = Path(fallback_path)
    df = pd.read_csv(path)
    results = []
    for source_file, g in df.groupby("source_file"):
        g = g.sort_values("strain")
        results.append({
            "material": g["material"].iloc[0],
            "temperature": float(g["temperature"].iloc[0]),
            "strain_rate": float(g["strain_rate"].iloc[0]),
            "strain": g["strain"].to_numpy(dtype=float),
            "stress_true": g["stress_MPa"].to_numpy(dtype=float),
            "stress_pred": g["stress_pred"].to_numpy(dtype=float),
        })
    return results, str(path)


def aggregate_by_temperature(results, material, strain_rate, pred_key="stress_pred", grid_points=200):
    filtered = [r for r in results if r["material"] == material and abs(r["strain_rate"] - strain_rate) < 0.01]
    if not filtered:
        return []

    grouped = {}
    for r in filtered:
        grouped.setdefault(r["temperature"], []).append(r)

    aggregated = []
    for temp, group in grouped.items():
        min_max_strain = min(np.max(g["strain"]) for g in group)
        max_min_strain = max(np.min(g["strain"]) for g in group)
        if min_max_strain <= max_min_strain:
            continue

        grid = np.linspace(max_min_strain, min_max_strain, grid_points)
        true_stack = []
        pred_stack = []
        for g in group:
            true_stack.append(np.interp(grid, g["strain"], g["stress_true"]))
            pred_stack.append(np.interp(grid, g["strain"], g[pred_key]))

        aggregated.append({
            "temperature": temp,
            "strain": grid,
            "stress_true": np.mean(true_stack, axis=0),
            "stress_pred": np.mean(pred_stack, axis=0),
            "count": len(group),
        })

    return sorted(aggregated, key=lambda x: x["temperature"])


def plot_multi_temperature_comparison(lstm_agg, gbr_agg, material, strain_rate, output_path):
    if not lstm_agg:
        return

    gbr_map = {item["temperature"]: item for item in gbr_agg}
    temps = [r["temperature"] for r in lstm_agg]
    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = colormaps.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(10, 7))

    for result in lstm_agg:
        temp = result["temperature"]
        color = cmap(norm(temp))

        ax.plot(result["strain"], result["stress_true"], color=color, linewidth=2.2, alpha=0.9)
        ax.plot(result["strain"], result["stress_pred"], color=color, linewidth=2.2, linestyle="--", alpha=0.9)

        if temp in gbr_map:
            gbr_curve = gbr_map[temp]
            gbr_pred = np.interp(result["strain"], gbr_curve["strain"], gbr_curve["stress_pred"])
            ax.plot(result["strain"], gbr_pred, color=color, linewidth=2.2, linestyle=":", alpha=0.9)

    ax.set_xlabel("Strain ε", fontsize=12, fontweight="bold")
    ax.set_ylabel("Stress σ (MPa)", fontsize=12, fontweight="bold")
    title = f"{material.upper()} - Strain Rate: {strain_rate} s⁻¹"
    ax.set_title(title, fontsize=14, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2, label="Exp. Data"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="LSTM Pred."),
        Line2D([0], [0], color="black", linewidth=2, linestyle=":", label="GBR Pred."),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  saved: {output_path.name}")


def plot_all_materials(lstm_results, gbr_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for r in lstm_results:
        key = (r["material"], r["strain_rate"])
        groups.setdefault(key, set()).add(r["temperature"])
    for r in gbr_results:
        key = (r["material"], r["strain_rate"])
        groups.setdefault(key, set()).add(r["temperature"])

    print("\nGenerating LSTM + GBR comparison plots...")
    print("=" * 60)

    for (material, strain_rate), temps in groups.items():
        if len(temps) < 2:
            continue
        lstm_agg = aggregate_by_temperature(lstm_results, material, strain_rate)
        if len(lstm_agg) < 2:
            continue

        gbr_agg = aggregate_by_temperature(gbr_results, material, strain_rate)
        rate_str = f"{strain_rate}".replace(".", "_")
        filename = f"{material}_rate{rate_str}_lstm_gbr_comparison.png"
        plot_multi_temperature_comparison(lstm_agg, gbr_agg, material, strain_rate, output_dir / filename)


def main():
    device = torch.device(CONFIG["device"])
    print("=" * 60)
    print("Stress-strain comparison: LSTM vs GBR")
    print("=" * 60)
    print(f"\nDevice: {device}")

    model = load_lstm_model(CONFIG["model_path"], device)
    data_dir = Path(CONFIG["data_dir"])

    train_dataset = StressStrainDataset(
        str(data_dir / "train.pkl"),
        fit_scaler=True,
        fit_condition_scaler=True,
    )
    shared_stress_scaler = train_dataset.stress_scaler
    shared_condition_scaler = train_dataset.condition_scaler
    test_dataset = StressStrainDataset(
        str(data_dir / "test.pkl"),
        stress_scaler=shared_stress_scaler,
        condition_scaler=shared_condition_scaler,
    )

    print(f"  train samples: {len(train_dataset)}")
    print(f"  test samples: {len(test_dataset)}")
    lstm_results = predict_lstm_results(model, train_dataset, device) + predict_lstm_results(model, test_dataset, device)
    gbr_results, gbr_path = build_gbr_results(CONFIG["gbr_predictions"], CONFIG["gbr_fallback_predictions"])
    print(f"  GBR predictions: {gbr_path}")

    output_dir = prepare_output_dir(CONFIG["output_dir"], CONFIG["clear_output_dir"])
    plot_all_materials(lstm_results, gbr_results, output_dir)

    print("\nOutput dir:", str(output_dir))


if __name__ == "__main__":
    main()
