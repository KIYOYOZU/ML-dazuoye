#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare a smoothed, plastic-segment dataset for simplified PD-ML training.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


CONFIG = {
    "input_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged_with_plastic_metrics.csv",
    "output_all_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_pdml_with_mask.csv",
    "output_ready_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_pdml_ready.csv",
    "summary_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\pdml_ready_summary.csv",
    "smooth_window": 7,
    "smooth_method": "ma",  # ma | median | savgol
    "h_smooth_window": 5,
    "h_smooth_method": "median",  # ma | median
    "h_clip_percentile": (1.0, 99.0),
    "h_clip_by_material": True,
    "clip_negative_h": False,
    "min_plastic_strain": 1e-4,
    "min_d_eps_p_d_eps": 5e-4,
    "clip_negative_gamma_dot": True,
}


MATERIAL_FEATURES = {
    "al2024": {"E_GPa": 73.1},
    "al2219": {"E_GPa": 73.1},
    "al6061": {"E_GPa": 68.9},
    "al7075": {"E_GPa": 71.7},
}


def _smooth(values: np.ndarray, window: int, method: str) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if method == "median":
        return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
            poly = min(3, window - 1)
            return savgol_filter(values, window_length=window, polyorder=poly, mode="interp")
        except Exception:
            pass
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_nan(values: np.ndarray, window: int, method: str) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    if method == "median":
        return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _prepare_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("strain").copy()
    if g["strain"].duplicated().any():
        keep_cols = ["material", "test_type", "series", "temperature", "strain_rate", "is_converted", "source_file"]
        first_meta = g.iloc[0][keep_cols].to_dict()
        g = g.groupby("strain", as_index=False)["stress_MPa"].mean()
        for k, v in first_meta.items():
            g[k] = v
        g = g.sort_values("strain").reset_index(drop=True)
    return g


def _derive_smoothed_metrics(g: pd.DataFrame) -> pd.DataFrame:
    material = g["material"].iloc[0]
    if material not in MATERIAL_FEATURES:
        raise ValueError(f"Unknown material: {material}")

    E = MATERIAL_FEATURES[material]["E_GPa"] * 1000.0  # MPa
    strain = g["strain"].to_numpy(dtype=float)
    stress = g["stress_MPa"].to_numpy(dtype=float)
    strain_rate = float(g["strain_rate"].iloc[0])

    stress_s = _smooth(stress, CONFIG["smooth_window"], CONFIG["smooth_method"])
    plastic_strain = strain - stress_s / max(E, 1e-6)
    plastic_strain = np.clip(plastic_strain, 0.0, None)

    if len(strain) >= 2:
        d_eps_p_d_eps = np.gradient(plastic_strain, strain, edge_order=1)
        d_sigma_d_eps = np.gradient(stress_s, strain, edge_order=1)
    else:
        d_eps_p_d_eps = np.zeros_like(strain)
        d_sigma_d_eps = np.zeros_like(strain)

    gamma_dot = d_eps_p_d_eps * strain_rate
    if CONFIG["clip_negative_gamma_dot"]:
        gamma_dot = np.clip(gamma_dot, 0.0, None)

    H = np.full_like(stress_s, np.nan)
    mask = d_eps_p_d_eps > CONFIG["min_d_eps_p_d_eps"]
    with np.errstate(divide="ignore", invalid="ignore"):
        H[mask] = d_sigma_d_eps[mask] / d_eps_p_d_eps[mask]
    H_s = _smooth_nan(H, CONFIG["h_smooth_window"], CONFIG["h_smooth_method"])

    g["stress_MPa_smooth"] = stress_s
    g["plastic_strain_smooth"] = plastic_strain
    g["gamma_dot_smooth"] = gamma_dot
    g["d_eps_p_d_eps"] = d_eps_p_d_eps
    g["d_sigma_d_eps"] = d_sigma_d_eps
    g["hardening_modulus_raw"] = H
    g["hardening_modulus_smooth"] = H_s
    g["plastic_mask"] = (plastic_strain >= CONFIG["min_plastic_strain"]) & mask & np.isfinite(H_s)
    return g


def _clip_h(out_df: pd.DataFrame) -> pd.DataFrame:
    out_df = out_df.copy()
    out_df["hardening_modulus_clip"] = out_df["hardening_modulus_smooth"]

    if CONFIG["h_clip_by_material"]:
        for mat, block in out_df.groupby("material"):
            vals = block["hardening_modulus_smooth"].to_numpy()
            finite = np.isfinite(vals)
            if finite.sum() < 5:
                continue
            low_p, high_p = CONFIG["h_clip_percentile"]
            low = np.percentile(vals[finite], low_p)
            high = np.percentile(vals[finite], high_p)
            clipped = np.clip(vals, low, high)
            if CONFIG["clip_negative_h"]:
                clipped = np.clip(clipped, 0.0, None)
            out_df.loc[block.index, "hardening_modulus_clip"] = clipped
    else:
        vals = out_df["hardening_modulus_smooth"].to_numpy()
        finite = np.isfinite(vals)
        if finite.sum() >= 5:
            low_p, high_p = CONFIG["h_clip_percentile"]
            low = np.percentile(vals[finite], low_p)
            high = np.percentile(vals[finite], high_p)
            clipped = np.clip(vals, low, high)
            if CONFIG["clip_negative_h"]:
                clipped = np.clip(clipped, 0.0, None)
            out_df.loc[:, "hardening_modulus_clip"] = clipped

    out_df["plastic_mask"] = out_df["plastic_mask"] & np.isfinite(out_df["hardening_modulus_clip"])
    return out_df


def main() -> None:
    df = pd.read_csv(CONFIG["input_csv"])
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    all_groups = []
    summary = []
    for source_file, g in df.groupby("source_file"):
        g2 = _prepare_group(g)
        g2 = _derive_smoothed_metrics(g2)
        all_groups.append(g2)
        summary.append({
            "source_file": source_file,
            "material": g2["material"].iloc[0],
            "points_total": int(len(g2)),
            "points_plastic": int(g2["plastic_mask"].sum()),
        })

    out_all = pd.concat(all_groups, ignore_index=True)
    out_all = _clip_h(out_all)
    out_ready = out_all[out_all["plastic_mask"]].copy()

    out_all_path = Path(CONFIG["output_all_csv"])
    out_ready_path = Path(CONFIG["output_ready_csv"])
    out_all_path.parent.mkdir(parents=True, exist_ok=True)
    out_ready_path.parent.mkdir(parents=True, exist_ok=True)
    out_all.to_csv(out_all_path, index=False)
    out_ready.to_csv(out_ready_path, index=False)

    summary_path = Path(CONFIG["summary_csv"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("all rows:", len(out_all))
    print("plastic rows:", len(out_ready))
    print("output all:", str(out_all_path))
    print("output ready:", str(out_ready_path))
    print("summary:", str(summary_path))


if __name__ == "__main__":
    main()
