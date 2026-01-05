#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stress-strain visualization - 每个应变率一张图，只做温度中值预测
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import joblib
from matplotlib import colormaps
from matplotlib.colors import Normalize


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = {
    "gbr_predictions": str(BASE_DIR / "results" / "baseline_gbr" / "all_predictions.csv"),
    "midpoint_predictions": str(BASE_DIR / "results" / "baseline_gbr" / "midpoint_predictions.csv"),
    "gbr_model": str(BASE_DIR / "results" / "baseline_gbr" / "gbr_model.pkl"),
    "output_dir": str(BASE_DIR / "results" / "figures"),
    "clear_output_dir": True,
}

# 定义每个材料要绘制的图（应变率列表和文件模式）
PLOT_CONFIG = {
    "al2024": {
        "rates": [0.1, 1.0, 10.0],
        "file_patterns": ["fig2a", "fig2b", "fig2c"],  # 只用fig2系列
        "fig_names": ["fig2a", "fig2b", "fig2c"],
    },
    "al2219": {
        "rates": [0.1, 1.0, 5.0],
        "file_patterns": ["fig3a", "fig3b", "fig3d"],  # 只用fig3a, fig3b, fig3d
        "fig_names": ["fig3a", "fig3b", "fig3d"],
    },
    "al6061": {
        "rates": [0.001],
        "file_patterns": ["tensile"],  # tensile系列
        "fig_names": ["tensile"],
    },
    "al7075": {
        "rates": [0.01, 0.1, 1.0, 10.0],
        "file_patterns": ["fig1a"],  # fig1a的4个应变率
        "fig_names": ["fig1a_0.01", "fig1a_0.1", "fig1a_1.0", "fig1a_10.0"],
    },
}


def _format_temp(temp: float) -> str:
    if abs(temp - round(temp)) < 1e-3:
        return f"{int(round(temp))}°C"
    return f"{temp:.0f}°C"


def _format_rate(rate: float) -> str:
    if rate == 0:
        return "0"
    if rate >= 1:
        if rate == int(rate):
            return f"{int(rate)}"
        return f"{rate:.1f}"
    else:
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
    """根据文件名模式过滤数据"""
    if "source_file" not in df.columns:
        return df

    mask = df["source_file"].str.lower().apply(
        lambda x: any(p in x.lower() for p in patterns)
    )
    return df[mask].copy()


def plot_single_rate(pred_df, midpoint_df, material, strain_rate, file_pattern, output_dir, fig_name):
    """为单个材料的单个应变率绘制温度曲线图"""

    # 过滤数据：指定材料、应变率、文件模式
    df = pred_df[
        (pred_df["material"] == material) &
        (np.isclose(pred_df["strain_rate"], strain_rate, rtol=0.1))
    ]

    # 按文件模式过滤
    if file_pattern:
        df = filter_by_file_pattern(df, [file_pattern] if isinstance(file_pattern, str) else file_pattern)

    if df.empty:
        print(f"  {material} @ {strain_rate}: 无数据，跳过")
        return False

    # 获取所有温度
    temps = sorted(df["temperature"].unique())
    if len(temps) < 2:
        print(f"  {material} @ {strain_rate}: 温度不足，跳过")
        return False

    # 获取中值预测温度
    pred_temps = []
    if midpoint_df is not None:
        mid = midpoint_df[
            (midpoint_df["material"] == material) &
            (midpoint_df["pred_type"] == "temp_midpoint") &
            (np.isclose(midpoint_df["strain_rate"], strain_rate, rtol=0.1))
        ]
        if not mid.empty:
            pred_temps = sorted(mid["temperature"].unique().tolist())

    # 创建图
    fig, ax = plt.subplots(figsize=(10, 7))

    # 颜色映射
    all_temps = sorted(set(temps) | set(pred_temps))
    cmap = colormaps.get_cmap("coolwarm")
    norm = Normalize(vmin=min(all_temps), vmax=max(all_temps))

    # 绘制每个温度的原始数据曲线（实线）
    for temp in temps:
        temp_df = df[np.isclose(df["temperature"], temp)].sort_values("strain")
        color = cmap(norm(temp))
        label = _format_temp(temp)
        ax.plot(temp_df["strain"], temp_df["stress_MPa"], "-",
                color=color, linewidth=2, label=label)

    # 绘制中值温度预测（虚线）
    if midpoint_df is not None and pred_temps:
        mid = midpoint_df[
            (midpoint_df["material"] == material) &
            (midpoint_df["pred_type"] == "temp_midpoint") &
            (np.isclose(midpoint_df["strain_rate"], strain_rate, rtol=0.1))
        ]
        for temp in pred_temps:
            temp_mid = mid[np.isclose(mid["temperature"], temp)].sort_values("strain")
            if not temp_mid.empty:
                color = cmap(norm(temp))
                label = f"Pred {_format_temp(temp)}"
                ax.plot(temp_mid["strain"], temp_mid["stress_pred"], "--",
                        color=color, linewidth=2, label=label)

    # 设置图表
    ax.set_xlabel("Strain", fontsize=12)
    ax.set_ylabel("Stress (MPa)", fontsize=12)
    ax.set_title(f"{material.upper()} - {fig_name} ({_format_rate(strain_rate)} s^-1)", fontsize=14, fontweight="bold")
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
    print("Stress-strain visualization")
    print("=" * 60)

    # 读取数据
    pred_path = Path(CONFIG["gbr_predictions"])
    if not pred_path.exists():
        print(f"错误: {pred_path} 不存在，请先运行 stress_train.py")
        return

    pred_df = pd.read_csv(pred_path)
    print(f"  Predictions: {len(pred_df)} rows")

    midpoint_path = Path(CONFIG["midpoint_predictions"])
    midpoint_df = pd.read_csv(midpoint_path) if midpoint_path.exists() else None
    if midpoint_df is not None:
        print(f"  Midpoint predictions: {len(midpoint_df)} rows")

    output_dir = prepare_output_dir(CONFIG["output_dir"], CONFIG["clear_output_dir"])

    # 为每个材料的每个应变率生成图
    print("\nGenerating plots...")
    for material, config in PLOT_CONFIG.items():
        print(f"\n{material.upper()}:")
        rates = config["rates"]
        patterns = config["file_patterns"]
        fig_names = config["fig_names"]

        for i, rate in enumerate(rates):
            # 确定文件模式
            if len(patterns) == len(rates):
                pattern = patterns[i]
            else:
                pattern = patterns[0]  # AL7075: 所有应变率都用fig1a

            fig_name = fig_names[i]
            plot_single_rate(pred_df, midpoint_df, material, rate, pattern, output_dir, fig_name)

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
