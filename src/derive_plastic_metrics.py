#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Derive approximate plastic metrics from stress-strain curves.
Outputs: plastic_strain, gamma_dot (equiv. plastic strain rate), hardening_modulus.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


CONFIG = {
    "merged_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged.csv",
    "output_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged_with_plastic_metrics.csv",
    "clip_negative_strain": True,
    "clip_negative_stress": False,
    "clip_negative_gamma_dot": True,
    "min_d_eps_p_d_eps": 1e-5,
}


MATERIAL_FEATURES = {
    "al2024": {"E_GPa": 73.1},
    "al2219": {"E_GPa": 73.1},
    "al6061": {"E_GPa": 68.9},
    "al7075": {"E_GPa": 71.7},
}


def _prepare_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("strain").copy()
    if CONFIG["clip_negative_strain"]:
        g["strain"] = g["strain"].clip(lower=0.0)
    if CONFIG["clip_negative_stress"]:
        g["stress_MPa"] = g["stress_MPa"].clip(lower=0.0)

    # Average duplicate strain points to avoid gradient issues
    if g["strain"].duplicated().any():
        keep_cols = ["material", "test_type", "series", "temperature", "strain_rate", "is_converted", "source_file"]
        first_meta = g.iloc[0][keep_cols].to_dict()
        g = g.groupby("strain", as_index=False)["stress_MPa"].mean()
        for k, v in first_meta.items():
            g[k] = v
        g = g.sort_values("strain").reset_index(drop=True)

    return g


def _derive_metrics(g: pd.DataFrame) -> pd.DataFrame:
    material = g["material"].iloc[0]
    if material not in MATERIAL_FEATURES:
        raise ValueError(f"Unknown material: {material}")

    E = MATERIAL_FEATURES[material]["E_GPa"] * 1000.0  # MPa
    strain = g["strain"].to_numpy(dtype=float)
    stress = g["stress_MPa"].to_numpy(dtype=float)
    strain_rate = float(g["strain_rate"].iloc[0])

    # Plastic strain approximation
    plastic_strain = strain - stress / max(E, 1e-6)
    plastic_strain = np.clip(plastic_strain, 0.0, None)

    # d_eps_p / d_eps
    if len(strain) >= 2:
        d_eps_p_d_eps = np.gradient(plastic_strain, strain, edge_order=1)
        d_sigma_d_eps = np.gradient(stress, strain, edge_order=1)
    else:
        d_eps_p_d_eps = np.zeros_like(strain)
        d_sigma_d_eps = np.zeros_like(strain)

    # Equivalent plastic strain rate (gamma_dot)
    gamma_dot = d_eps_p_d_eps * strain_rate
    if CONFIG["clip_negative_gamma_dot"]:
        gamma_dot = np.clip(gamma_dot, 0.0, None)

    # Hardening modulus: dσ/dεp = (dσ/dε) / (dεp/dε)
    H = np.full_like(stress, np.nan)
    mask = d_eps_p_d_eps > CONFIG["min_d_eps_p_d_eps"]
    with np.errstate(divide="ignore", invalid="ignore"):
        H[mask] = d_sigma_d_eps[mask] / d_eps_p_d_eps[mask]

    g["plastic_strain"] = plastic_strain
    g["gamma_dot"] = gamma_dot
    g["hardening_modulus"] = H
    return g


def main() -> None:
    df = pd.read_csv(CONFIG["merged_csv"])
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out_groups = []
    for _, g in df.groupby("source_file"):
        g2 = _prepare_group(g)
        g2 = _derive_metrics(g2)
        out_groups.append(g2)

    out_df = pd.concat(out_groups, ignore_index=True)
    out_path = Path(CONFIG["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Quick summary
    nan_gamma = int(np.isnan(out_df["gamma_dot"]).sum())
    nan_H = int(np.isnan(out_df["hardening_modulus"]).sum())
    neg_gamma = int((out_df["gamma_dot"] < 0).sum())
    print("derived rows:", len(out_df))
    print("gamma_dot NaN:", nan_gamma, "negative:", neg_gamma)
    print("hardening_modulus NaN:", nan_H)
    print("output:", str(out_path))


if __name__ == "__main__":
    main()
