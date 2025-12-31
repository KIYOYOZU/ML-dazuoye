#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified PD-ML baseline using Gradient Boosting Regressor.
Point-wise mapping: (strain, temperature, strain_rate, material props) -> stress.
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
    "merged_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged.csv",
    "output_dir": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\baseline_gbr",
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
    "clip_negative_strain": True,
    "clip_negative_stress": False,
    "drop_negative_stress": False,
    "use_log_strain_rate": True,
    "model_params": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in merged csv: {sorted(missing)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if CONFIG["clip_negative_strain"]:
        df["strain"] = df["strain"].clip(lower=0.0)
    if CONFIG["drop_negative_stress"]:
        df = df[df["stress_MPa"] >= 0].copy()
    if CONFIG["clip_negative_stress"]:
        df["stress_MPa"] = df["stress_MPa"].clip(lower=0.0)
    return df.dropna(subset=["strain", "stress_MPa", "temperature", "strain_rate", "material"])


def add_material_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mats = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mats.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mats)
    return pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)


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

    feature_cols = [
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


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def main() -> None:
    df = load_data(CONFIG["merged_csv"])
    df = clean_data(df)
    df = add_material_features(df)

    train_files, val_files, test_files = split_by_curve(df)
    train_df = df[df["source_file"].isin(train_files)].copy()
    val_df = df[df["source_file"].isin(val_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)

    y_train = train_df["stress_MPa"].to_numpy(dtype=float)
    y_val = val_df["stress_MPa"].to_numpy(dtype=float)
    y_test = test_df["stress_MPa"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(**CONFIG["model_params"])
    model.fit(X_train_s, y_train)

    pred_train = model.predict(X_train_s)
    pred_val = model.predict(X_val_s)
    pred_test = model.predict(X_test_s)

    metrics = {
        "train": eval_metrics(y_train, pred_train),
        "val": eval_metrics(y_val, pred_val),
        "test": eval_metrics(y_test, pred_test),
    }

    # Per-material metrics on test set
    per_material = {}
    for mat in sorted(test_df["material"].unique()):
        idx = test_df["material"] == mat
        per_material[mat] = eval_metrics(y_test[idx.values], pred_test[idx.values])

    # Curve-level RMSE on test set
    test_out = test_df.copy()
    test_out["stress_pred"] = pred_test
    curve_rmse = []
    for source_file, g in test_out.groupby("source_file"):
        rmse = float(np.sqrt(mean_squared_error(g["stress_MPa"], g["stress_pred"])))
        curve_rmse.append({"source_file": source_file, "material": g["material"].iloc[0], "rmse": rmse, "n": int(len(g))})

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"overall": metrics, "per_material": per_material}, f, indent=2)

    pd.DataFrame(curve_rmse).sort_values("rmse", ascending=False).to_csv(
        output_dir / "curve_rmse.csv", index=False
    )
    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

    print("GBR baseline done")
    print("overall test:", metrics["test"])
    print("per_material:", per_material)
    print("outputs:", str(output_dir))


if __name__ == "__main__":
    main()
