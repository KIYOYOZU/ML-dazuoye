#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBR应力应变训练入口（仅保留GBR基线）
包含: 数据加载、特征构建、训练、评估、结果保存
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==================== 配置 ====================
CONFIG = {
    "merged_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged.csv",
    "output_dir": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业",
    "filter_test_type": "tensile",
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
}

# 材料基础参数（来自既有训练数据的特征分布）
MATERIAL_FEATURES = {
    "al2024": {"material_id": 0, "E_GPa": 73.1, "c1": 325.0, "c2": 0.33, "c3": 22.78, "c4": 502.0, "c5": 2.29e-05},
    "al2219": {"material_id": 1, "E_GPa": 73.1, "c1": 350.0, "c2": 0.33, "c3": 22.84, "c4": 543.0, "c5": 2.23e-05},
    "al6061": {"material_id": 2, "E_GPa": 68.9, "c1": 276.0, "c2": 0.33, "c3": 22.70, "c4": 582.0, "c5": 2.36e-05},
    "al7075": {"material_id": 3, "E_GPa": 71.7, "c1": 503.0, "c2": 0.33, "c3": 22.81, "c4": 477.0, "c5": 2.32e-05},
}


def _filter_by_test_type(df: pd.DataFrame, test_type: str | None):
    if not test_type:
        return df
    if "test_type" not in df.columns:
        print("[GBR] 未找到 test_type 列，跳过过滤")
        return df
    mask = df["test_type"].astype(str).str.lower() == str(test_type).lower()
    filtered = df.loc[mask].copy()
    print(f"[GBR] 仅保留 test_type={test_type}: {len(filtered)} 行")
    return filtered


def gbr_clean_data(df: pd.DataFrame, test_type: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df = _filter_by_test_type(df, test_type)
    df["strain"] = df["strain"].clip(lower=0.0)
    return df.dropna(subset=["strain", "stress_MPa", "temperature", "strain_rate", "material"])


def gbr_add_material_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mats = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mats.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mats)
    return pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)


def gbr_split_by_curve(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
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


def gbr_build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_strain_rate"] = np.log10(df["strain_rate"].clip(lower=1e-12))
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


def gbr_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def run_gbr_baseline(base_dir: Path) -> None:
    merged_csv = Path(CONFIG["merged_csv"])
    df = pd.read_csv(merged_csv)
    df = gbr_clean_data(df, CONFIG.get("filter_test_type"))
    df = gbr_add_material_features(df)

    train_files, val_files, test_files = gbr_split_by_curve(df)
    train_df = df[df["source_file"].isin(train_files)].copy()
    val_df = df[df["source_file"].isin(val_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    X_train = gbr_build_features(train_df)
    X_val = gbr_build_features(val_df)
    X_test = gbr_build_features(test_df)

    y_train = train_df["stress_MPa"].to_numpy(dtype=float)
    y_val = val_df["stress_MPa"].to_numpy(dtype=float)
    y_test = test_df["stress_MPa"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=2,
        subsample=1.0,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    pred_train = model.predict(X_train_s)
    pred_val = model.predict(X_val_s)
    pred_test = model.predict(X_test_s)

    metrics = {
        "train": gbr_eval_metrics(y_train, pred_train),
        "val": gbr_eval_metrics(y_val, pred_val),
        "test": gbr_eval_metrics(y_test, pred_test),
    }

    per_material = {}
    for mat in sorted(test_df["material"].unique()):
        idx = test_df["material"] == mat
        per_material[mat] = gbr_eval_metrics(y_test[idx.values], pred_test[idx.values])

    test_out = test_df.copy()
    test_out["stress_pred"] = pred_test
    curve_rmse = []
    for source_file, g in test_out.groupby("source_file"):
        rmse = float(np.sqrt(mean_squared_error(g["stress_MPa"], g["stress_pred"])))
        curve_rmse.append({"source_file": source_file, "material": g["material"].iloc[0], "rmse": rmse, "n": int(len(g))})

    output_dir = base_dir / "results" / "baseline_gbr"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "gbr_model.pkl")
    joblib.dump(scaler, output_dir / "gbr_scaler.pkl")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"overall": metrics, "per_material": per_material}, f, indent=2)

    pd.DataFrame(curve_rmse).sort_values("rmse", ascending=False).to_csv(
        output_dir / "curve_rmse.csv", index=False
    )
    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

    X_all = gbr_build_features(df)
    X_all_s = scaler.transform(X_all)
    pred_all = model.predict(X_all_s)
    all_out = df.copy()
    all_out["stress_pred"] = pred_all
    all_out.to_csv(output_dir / "all_predictions.csv", index=False)

    print("GBR baseline done")
    print("overall test:", metrics["test"])
    print("per_material:", per_material)
    print("outputs:", str(output_dir))


def main():
    print("=" * 60)
    print("铝合金应力应变曲线预测 - GBR训练程序")
    print("=" * 60)

    base_dir = Path(CONFIG["output_dir"])
    run_gbr_baseline(base_dir)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
