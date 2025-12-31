#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train simplified PD-ML regressors on plastic-segment data.
Predict gamma_dot and hardening modulus from stress/strain/conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "input_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_pdml_ready.csv",
    "output_dir": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\pdml_weak",
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
    "use_log_strain_rate": True,
    "target_gamma": "gamma_dot_smooth",
    "target_h": "hardening_modulus_clip",
    "model_gamma": {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "subsample": 1.0,
        "random_state": 42,
    },
    "model_h": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "subsample": 1.0,
        "random_state": 42,
    },
}


MATERIAL_FEATURES = {
    "al2024": {"material_id": 0, "E_GPa": 73.1},
    "al2219": {"material_id": 1, "E_GPa": 73.1},
    "al6061": {"material_id": 2, "E_GPa": 68.9},
    "al7075": {"material_id": 3, "E_GPa": 71.7},
}


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def split_by_curve(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    curve_meta = df.groupby("source_file").first()[["material"]]
    rng = np.random.default_rng(CONFIG["split_seed"])
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

    mats = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mats.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mats)
    df = pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)

    feature_cols = [
        "stress_MPa_smooth",
        "strain",
        "plastic_strain_smooth",
        "temperature",
        "log_strain_rate",
        "material_id",
        "E_GPa",
    ]
    return df[feature_cols].astype(float)


def main() -> None:
    df = pd.read_csv(CONFIG["input_csv"])
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa_smooth",
                "plastic_strain_smooth", CONFIG["target_gamma"], CONFIG["target_h"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Drop any invalid rows for targets
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[CONFIG["target_gamma"], CONFIG["target_h"]]).copy()

    train_files, val_files, test_files = split_by_curve(df)
    train_df = df[df["source_file"].isin(train_files)].copy()
    val_df = df[df["source_file"].isin(val_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    y_gamma_train = train_df[CONFIG["target_gamma"]].to_numpy(dtype=float)
    y_gamma_val = val_df[CONFIG["target_gamma"]].to_numpy(dtype=float)
    y_gamma_test = test_df[CONFIG["target_gamma"]].to_numpy(dtype=float)

    y_h_train = train_df[CONFIG["target_h"]].to_numpy(dtype=float)
    y_h_val = val_df[CONFIG["target_h"]].to_numpy(dtype=float)
    y_h_test = test_df[CONFIG["target_h"]].to_numpy(dtype=float)

    model_gamma = GradientBoostingRegressor(**CONFIG["model_gamma"])
    model_h = GradientBoostingRegressor(**CONFIG["model_h"])

    model_gamma.fit(X_train_s, y_gamma_train)
    model_h.fit(X_train_s, y_h_train)

    pred_gamma_train = model_gamma.predict(X_train_s)
    pred_gamma_val = model_gamma.predict(X_val_s)
    pred_gamma_test = model_gamma.predict(X_test_s)

    pred_h_train = model_h.predict(X_train_s)
    pred_h_val = model_h.predict(X_val_s)
    pred_h_test = model_h.predict(X_test_s)

    metrics = {
        "gamma": {
            "train": eval_metrics(y_gamma_train, pred_gamma_train),
            "val": eval_metrics(y_gamma_val, pred_gamma_val),
            "test": eval_metrics(y_gamma_test, pred_gamma_test),
        },
        "H": {
            "train": eval_metrics(y_h_train, pred_h_train),
            "val": eval_metrics(y_h_val, pred_h_val),
            "test": eval_metrics(y_h_test, pred_h_test),
        },
    }

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    test_out = test_df.copy()
    test_out["gamma_pred"] = pred_gamma_test
    test_out["H_pred"] = pred_h_test
    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

    print("weak PD-ML training done")
    print("gamma test:", metrics["gamma"]["test"])
    print("H test:", metrics["H"]["test"])
    print("outputs:", str(output_dir))


if __name__ == "__main__":
    main()
