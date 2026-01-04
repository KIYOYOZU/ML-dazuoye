#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PINN 模型可视化 - 加载 PINN 模型生成预测曲线
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = {
    "pinn_model": str(BASE_DIR / "results" / "stress_pinn" / "pinn_model.pt"),
    "merged_csv": str(BASE_DIR / "data" / "processed" / "筛选数据_merged.csv"),
    "output_dir": str(BASE_DIR / "results" / "figures_pinn"),
    "clear_output_dir": True,
}

# 定义每个材料要绘制的图
PLOT_CONFIG = {
    "al2024": {
        "rates": [0.1, 1.0, 10.0],
        "file_patterns": ["fig2a", "fig2b", "fig2c"],
        "fig_names": ["fig2a", "fig2b", "fig2c"],
    },
    "al2219": {
        "rates": [0.1, 1.0, 5.0],
        "file_patterns": ["fig3a", "fig3b", "fig3d"],
        "fig_names": ["fig3a", "fig3b", "fig3d"],
    },
    "al6061": {
        "rates": [0.001],
        "file_patterns": ["tensile"],
        "fig_names": ["tensile"],
    },
    "al7075": {
        "rates": [0.01, 0.1, 1.0, 10.0],
        "file_patterns": ["fig1a"],
        "fig_names": ["fig1a_0.01", "fig1a_0.1", "fig1a_1.0", "fig1a_10.0"],
    },
}

# 材料物理参数（与 stress_pinn.py 一致）
MATERIAL_FEATURES = {
    "al2024": {
        "material_id": 0, "E_GPa": 73.1,
        "c1": 325.0, "c2": 0.33, "c3": 22.78, "c4": 502.0, "c5": 2.29e-05,
        "Si_wt": 0.50, "Fe_wt": 0.50, "Cu_wt": 4.35, "Mn_wt": 0.60,
        "Mg_wt": 1.50, "Cr_wt": 0.10, "Zn_wt": 0.25, "Ti_wt": 2.00,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 90.20,
    },
    "al2219": {
        "material_id": 1, "E_GPa": 73.1,
        "c1": 350.0, "c2": 0.33, "c3": 22.84, "c4": 543.0, "c5": 2.23e-05,
        "Si_wt": 0.20, "Fe_wt": 0.30, "Cu_wt": 6.30, "Mn_wt": 0.30,
        "Mg_wt": 0.02, "Cr_wt": 0.00, "Zn_wt": 0.10, "Ti_wt": 0.06,
        "Zr_wt": 0.175, "V_wt": 0.10, "Al_wt": 92.445,
    },
    "al6061": {
        "material_id": 2, "E_GPa": 68.9,
        "c1": 276.0, "c2": 0.33, "c3": 22.70, "c4": 582.0, "c5": 2.36e-05,
        "Si_wt": 0.63, "Fe_wt": 0.2225, "Cu_wt": 0.195, "Mn_wt": 0.045,
        "Mg_wt": 0.8775, "Cr_wt": 0.0525, "Zn_wt": 0.02, "Ti_wt": 0.015,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 97.9425,
    },
    "al7075": {
        "material_id": 3, "E_GPa": 71.7,
        "c1": 503.0, "c2": 0.33, "c3": 22.81, "c4": 477.0, "c5": 2.32e-05,
        "Si_wt": 0.40, "Fe_wt": 0.50, "Cu_wt": 1.60, "Mn_wt": 0.30,
        "Mg_wt": 2.50, "Cr_wt": 0.23, "Zn_wt": 5.60, "Ti_wt": 0.20,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 88.67,
    },
}

COMPOSITION_COLS = ["Si_wt", "Fe_wt", "Cu_wt", "Mn_wt", "Mg_wt",
                   "Cr_wt", "Zn_wt", "Ti_wt", "Zr_wt", "V_wt", "Al_wt"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== PINN 模型定义 ====================
class StressPINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_pinn_model(model_path: str):
    """加载 PINN 模型和 scaler"""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    config = checkpoint["config"]
    # 从 checkpoint 获取特征维度
    feature_cols = checkpoint.get("feature_cols", ["strain", "temperature", "log_strain_rate", "material_id"])
    input_dim = len(feature_cols)

    model = StressPINN(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 重建 scaler
    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint["scaler_X_mean"]
    scaler_X.scale_ = checkpoint["scaler_X_scale"]
    scaler_X.var_ = scaler_X.scale_ ** 2
    scaler_X.n_features_in_ = len(scaler_X.mean_)

    scaler_y = StandardScaler()
    scaler_y.mean_ = checkpoint["scaler_y_mean"]
    scaler_y.scale_ = checkpoint["scaler_y_scale"]
    scaler_y.var_ = scaler_y.scale_ ** 2
    scaler_y.n_features_in_ = 1

    return model, scaler_X, scaler_y, config


def predict_stress(model, scaler_X, scaler_y, strain, temperature, strain_rate, material):
    """使用 PINN 模型预测应力（21维特征）"""
    mat_feat = MATERIAL_FEATURES[material]
    log_strain_rate = np.log10(max(strain_rate, 1e-12))

    # 构建特征 [strain, temperature, log_strain_rate, material_id, E_GPa, c1-c5, 元素成分]
    X = []
    for s in strain:
        row = [s, temperature, log_strain_rate, mat_feat["material_id"],
               mat_feat["E_GPa"], mat_feat["c1"], mat_feat["c2"],
               mat_feat["c3"], mat_feat["c4"], mat_feat["c5"]]
        for col in COMPOSITION_COLS:
            row.append(mat_feat[col])
        X.append(row)

    X = np.array(X, dtype=np.float32)
    X_scaled = scaler_X.transform(X)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(DEVICE)
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return y_pred


def _format_temp(temp: float) -> str:
    if abs(temp - round(temp)) < 1e-3:
        return f"{int(round(temp))}C"
    return f"{temp:.0f}C"


def _format_rate(rate: float) -> str:
    if rate == 0:
        return "0"
    if rate >= 1:
        return f"{int(rate)}" if rate == int(rate) else f"{rate:.1f}"
    return f"{rate}"


def prepare_output_dir(output_dir: str, clear: bool) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    return output_dir


def filter_by_file_pattern(df: pd.DataFrame, patterns: list) -> pd.DataFrame:
    if "source_file" not in df.columns:
        return df
    mask = df["source_file"].str.lower().apply(
        lambda x: any(p in x.lower() for p in patterns)
    )
    return df[mask].copy()


def build_temperature_midpoints(temps: list[float]) -> list[float]:
    """计算相邻温度的中值点"""
    uniq = sorted(set(temps))
    if len(uniq) < 2:
        return []
    return [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]


def plot_single_rate_pinn(
    model, scaler_X, scaler_y,
    raw_df, material, strain_rate, file_pattern,
    output_dir, fig_name
):
    """为单个材料的单个应变率绘制温度曲线图（PINN版）"""

    # 过滤原始数据
    df = raw_df[
        (raw_df["material"] == material) &
        (np.isclose(raw_df["strain_rate"], strain_rate, rtol=0.1))
    ]

    if file_pattern:
        df = filter_by_file_pattern(df, [file_pattern] if isinstance(file_pattern, str) else file_pattern)

    if df.empty:
        print(f"  {material} @ {strain_rate}: no data, skip")
        return False

    temps = sorted(df["temperature"].unique())
    if len(temps) < 2:
        print(f"  {material} @ {strain_rate}: not enough temps, skip")
        return False

    # 计算中值温度
    mid_temps = build_temperature_midpoints(temps)

    # 构建应变网格
    strain_min = df["strain"].min()
    strain_max = df["strain"].max()
    strain_grid = np.linspace(max(0, strain_min), strain_max, 200)

    # 创建图
    fig, ax = plt.subplots(figsize=(10, 7))

    # 颜色映射
    all_temps = sorted(set(temps) | set(mid_temps))
    cmap = colormaps.get_cmap("coolwarm")
    norm = Normalize(vmin=min(all_temps), vmax=max(all_temps))

    # 绘制实验数据（实线）
    for temp in temps:
        temp_df = df[np.isclose(df["temperature"], temp)].sort_values("strain")
        color = cmap(norm(temp))
        label = _format_temp(temp)
        ax.plot(temp_df["strain"], temp_df["stress_MPa"], "-",
                color=color, linewidth=2, label=label)

    # 绘制 PINN 中值预测（虚线）
    for mid_temp in mid_temps:
        stress_pred = predict_stress(model, scaler_X, scaler_y,
                                     strain_grid, mid_temp, strain_rate, material)
        color = cmap(norm(mid_temp))
        label = f"PINN {_format_temp(mid_temp)}"
        ax.plot(strain_grid, stress_pred, "--",
                color=color, linewidth=2, label=label)

    # 设置图表
    ax.set_xlabel("Strain", fontsize=12)
    ax.set_ylabel("Stress (MPa)", fontsize=12)
    ax.set_title(f"{material.upper()} - {fig_name} ({_format_rate(strain_rate)} s^-1) [PINN]",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # 保存
    output_path = output_dir / f"{material}_{fig_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  saved: {output_path.name}")
    return True


def main():
    print("=" * 60)
    print("PINN Stress-strain visualization")
    print("=" * 60)

    # 加载 PINN 模型
    model_path = Path(CONFIG["pinn_model"])
    if not model_path.exists():
        print(f"Error: {model_path} not found, run stress_pinn.py first")
        return

    print(f"\nLoading PINN model from {model_path}...")
    model, scaler_X, scaler_y, model_config = load_pinn_model(str(model_path))
    print(f"  Model config: {model_config['hidden_dims']}")

    # 加载原始数据
    raw_df = pd.read_csv(CONFIG["merged_csv"])
    # 过滤 tensile
    raw_df = raw_df[raw_df["test_type"].str.lower() == "tensile"].copy()
    print(f"  Raw data: {len(raw_df)} rows")

    output_dir = prepare_output_dir(CONFIG["output_dir"], CONFIG["clear_output_dir"])

    # 生成图
    print("\nGenerating PINN plots...")
    for material, config in PLOT_CONFIG.items():
        print(f"\n{material.upper()}:")
        rates = config["rates"]
        patterns = config["file_patterns"]
        fig_names = config["fig_names"]

        for i, rate in enumerate(rates):
            pattern = patterns[i] if len(patterns) == len(rates) else patterns[0]
            fig_name = fig_names[i]
            plot_single_rate_pinn(
                model, scaler_X, scaler_y,
                raw_df, material, rate, pattern,
                output_dir, fig_name
            )

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
