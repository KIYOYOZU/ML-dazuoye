#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weak PD-ML training:
Inputs: (stress, strain, temperature, strain_rate, material props)
Outputs: gamma_dot_smooth, hardening_modulus_smooth
Per-material models (two regressors per material).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib


CONFIG = {
    "input_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_pdml_ready.csv",
    "output_dir": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\weak_pdml",
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
    "use_log_strain_rate": True,
    "model_params": {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "subsample": 1.0,
        "random_state": 42,
    },
}


MATERIAL_FEATURES = {
    "al2024": {"material_id": 0, "E_GPa": 73.1, "c1": 325.0, "c2": 0.33, "c3": 2.78, "c4": 502.0, "c5": 2.29e-05},
    "al2219": {"material_id": 1, "E_GPa": 73.1, "c1": 350.0, "c2": 0.33, "c3": 2.84, "c4": 543.0, "c5": 2.23e-05},
    "al6061": {"material_id": 2, "E_GPa": 68.9, "c1": 276.0, "c2": 0.33, "c3": 2.70, "c4": 582.0, "c5": 2.36e-05},
    "al7075": {"material_id": 3, "E_GPa": 71.7, "c1": 503.0, "c2": 0.33, "c3": 2.81, "c4": 477.0, "c5": 2.32e-05},
}


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def split_by_curve(df: pd.DataFrame, seed: int) -> Tuple[List[str], List[str], List[str]]:
    curve_meta = df.groupby("source_file").first()[["material"]]
    rng = np.random.default_rng(seed)
    train_files, val_files, test_files = [], [], []

    for material in sorted(curve_meta["material"].unique()):
        files = sorted(curve_meta[curve_meta["material"] == material].index.tolist())
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * CONFIG["split_ratio"]["train"]))
        n_val = int(round(n * CONFIG["split_ratio"]["val"]))
        if n_train + n_val > n:
            n_train = max(0, n - n_val)
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    return train_files, val_files, test_files


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if CONFIG["use_log_strain_rate"]:
        df["log_strain_rate"] = np.log10(df["strain_rate"].clip(lower=1e-12))
    else:
        df["log_strain_rate"] = df["strain_rate"]

    feature_cols = [
        "stress_MPa_smooth",
        "strain",
        "temperature",
        "log_strain_rate",
        "material_id",
        "E_GPa",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
    ]
    return df[feature_cols].astype(float)


def add_material_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mats = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mats.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mats)
    return pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)


def train_for_material(df: pd.DataFrame, output_dir: Path, material: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    mat_df = df[df["material"] == material].copy()
    if mat_df.empty:
        return {}

    train_files, val_files, test_files = split_by_curve(mat_df, CONFIG["split_seed"])
    train_df = mat_df[mat_df["source_file"].isin(train_files)].copy()
    val_df = mat_df[mat_df["source_file"].isin(val_files)].copy()
    test_df = mat_df[mat_df["source_file"].isin(test_files)].copy()

    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)

    y_train_gamma = train_df["gamma_dot_smooth"].to_numpy(dtype=float)
    y_val_gamma = val_df["gamma_dot_smooth"].to_numpy(dtype=float)
    y_test_gamma = test_df["gamma_dot_smooth"].to_numpy(dtype=float)

    y_train_H = train_df["hardening_modulus_smooth"].to_numpy(dtype=float)
    y_val_H = val_df["hardening_modulus_smooth"].to_numpy(dtype=float)
    y_test_H = test_df["hardening_modulus_smooth"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model_gamma = GradientBoostingRegressor(**CONFIG["model_params"])
    model_H = GradientBoostingRegressor(**CONFIG["model_params"])

    model_gamma.fit(X_train_s, y_train_gamma)
    model_H.fit(X_train_s, y_train_H)

    pred_train_gamma = model_gamma.predict(X_train_s)
    pred_val_gamma = model_gamma.predict(X_val_s)
    pred_test_gamma = model_gamma.predict(X_test_s)

    pred_train_H = model_H.predict(X_train_s)
    pred_val_H = model_H.predict(X_val_s)
    pred_test_H = model_H.predict(X_test_s)

    metrics = {
        "gamma_dot": {
            "train": eval_metrics(y_train_gamma, pred_train_gamma),
            "val": eval_metrics(y_val_gamma, pred_val_gamma),
            "test": eval_metrics(y_test_gamma, pred_test_gamma),
        },
        "hardening_modulus": {
            "train": eval_metrics(y_train_H, pred_train_H),
            "val": eval_metrics(y_val_H, pred_val_H),
            "test": eval_metrics(y_test_H, pred_test_H),
        },
    }

    # Save predictions for test
    test_out = test_df.copy()
    test_out["gamma_dot_pred"] = pred_test_gamma
    test_out["hardening_modulus_pred"] = pred_test_H
    test_out.to_csv(output_dir / f"{material}_test_predictions.csv", index=False)

    # Save models and scaler
    joblib.dump(model_gamma, output_dir / f"{material}_gbr_gamma.pkl")
    joblib.dump(model_H, output_dir / f"{material}_gbr_H.pkl")
    joblib.dump(scaler, output_dir / f"{material}_scaler.pkl")

    return metrics


def main() -> None:
    df = pd.read_csv(CONFIG["input_csv"])
    required = {"source_file", "material", "temperature", "strain_rate", "strain",
                "stress_MPa_smooth", "gamma_dot_smooth", "hardening_modulus_smooth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = add_material_features(df)
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for material in sorted(df["material"].unique()):
        all_metrics[material] = train_for_material(df, output_dir, material)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("weak PD-ML training done")
    print("output:", str(output_dir))


if __name__ == "__main__":
    main()
