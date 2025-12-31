#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PD-ML 训练入口：派生塑性指标 -> 准备数据集 -> 弱 PD-ML 训练（可选 per-material）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

PATHS = {
    "merged_csv": BASE_DIR / "data" / "processed" / "筛选数据_merged.csv",
    "metrics_csv": BASE_DIR / "data" / "processed" / "筛选数据_merged_with_plastic_metrics.csv",
    "pdml_with_mask_csv": BASE_DIR / "data" / "processed" / "筛选数据_pdml_with_mask.csv",
    "pdml_ready_csv": BASE_DIR / "data" / "processed" / "筛选数据_pdml_ready.csv",
    "summary_csv": BASE_DIR / "results" / "pdml_ready_summary.csv",
    "weak_output_dir": BASE_DIR / "results" / "pdml_weak",
    "per_material_output_dir": BASE_DIR / "results" / "pdml_weak_material",
}

RUN_STEPS = {
    "derive_metrics": True,
    "prepare_dataset": True,
    "weak_pdml": True,
    "per_material": False,
}

DERIVE_CONFIG = {
    "clip_negative_strain": True,
    "clip_negative_stress": False,
    "clip_negative_gamma_dot": True,
    "min_d_eps_p_d_eps": 1e-5,
}

PREP_CONFIG = {
    "smooth_window": 7,
    "smooth_method": "ma",  # ma | median | savgol
    "h_smooth_window": 5,
    "h_smooth_method": "median",  # ma | median
    "h_clip_percentile": (1.0, 99.0),
    "h_clip_by_material": True,
    "clip_negative_h": False,
    "derive_h_log": True,
    "h_log_offset": 1e-6,
    "stable_plastic_quantile": (0.10, 0.90),
    "stable_min_points": 8,
    "ready_mask": "plastic_mask",  # plastic_mask | plastic_stable_mask
    "min_plastic_strain": 1e-4,
    "min_d_eps_p_d_eps": 5e-4,
    "clip_negative_gamma_dot": True,
}

WEAK_CONFIG = {
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
    "use_log_strain_rate": True,
    "target_gamma": "gamma_dot_smooth",
    "target_h": "hardening_modulus_clip",
    "h_use_log": True,
    "h_log_column": "hardening_modulus_log",
    "h_log_offset": 1e-6,
    "h_use_stable_mask": True,
    "h_stable_mask_col": "plastic_stable_mask",
    "h_positive_only": True,
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

PER_MATERIAL_CONFIG = {
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
    "al2024": {"material_id": 0, "E_GPa": 73.1, "c1": 325.0, "c2": 0.33, "c3": 22.78, "c4": 502.0, "c5": 2.29e-05},
    "al2219": {"material_id": 1, "E_GPa": 73.1, "c1": 350.0, "c2": 0.33, "c3": 22.84, "c4": 543.0, "c5": 2.23e-05},
    "al6061": {"material_id": 2, "E_GPa": 68.9, "c1": 276.0, "c2": 0.33, "c3": 22.70, "c4": 582.0, "c5": 2.36e-05},
    "al7075": {"material_id": 3, "E_GPa": 71.7, "c1": 503.0, "c2": 0.33, "c3": 22.81, "c4": 477.0, "c5": 2.32e-05},
}


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在: {path}")


def _prepare_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("strain").copy()
    if DERIVE_CONFIG["clip_negative_strain"]:
        g["strain"] = g["strain"].clip(lower=0.0)
    if DERIVE_CONFIG["clip_negative_stress"]:
        g["stress_MPa"] = g["stress_MPa"].clip(lower=0.0)

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

    E = MATERIAL_FEATURES[material]["E_GPa"] * 1000.0
    strain = g["strain"].to_numpy(dtype=float)
    stress = g["stress_MPa"].to_numpy(dtype=float)
    strain_rate = float(g["strain_rate"].iloc[0])

    plastic_strain = strain - stress / max(E, 1e-6)
    plastic_strain = np.clip(plastic_strain, 0.0, None)

    if len(strain) >= 2:
        d_eps_p_d_eps = np.gradient(plastic_strain, strain, edge_order=1)
        d_sigma_d_eps = np.gradient(stress, strain, edge_order=1)
    else:
        d_eps_p_d_eps = np.zeros_like(strain)
        d_sigma_d_eps = np.zeros_like(strain)

    gamma_dot = d_eps_p_d_eps * strain_rate
    if DERIVE_CONFIG["clip_negative_gamma_dot"]:
        gamma_dot = np.clip(gamma_dot, 0.0, None)

    H = np.full_like(stress, np.nan)
    mask = d_eps_p_d_eps > DERIVE_CONFIG["min_d_eps_p_d_eps"]
    with np.errstate(divide="ignore", invalid="ignore"):
        H[mask] = d_sigma_d_eps[mask] / d_eps_p_d_eps[mask]

    g["plastic_strain"] = plastic_strain
    g["gamma_dot"] = gamma_dot
    g["hardening_modulus"] = H
    return g


def derive_plastic_metrics(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    nan_gamma = int(np.isnan(out_df["gamma_dot"]).sum())
    nan_H = int(np.isnan(out_df["hardening_modulus"]).sum())
    neg_gamma = int((out_df["gamma_dot"] < 0).sum())
    print("derived rows:", len(out_df))
    print("gamma_dot NaN:", nan_gamma, "negative:", neg_gamma)
    print("hardening_modulus NaN:", nan_H)
    print("output:", str(output_csv))


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


def _prepare_group_pdml(g: pd.DataFrame) -> pd.DataFrame:
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

    E = MATERIAL_FEATURES[material]["E_GPa"] * 1000.0
    strain = g["strain"].to_numpy(dtype=float)
    stress = g["stress_MPa"].to_numpy(dtype=float)
    strain_rate = float(g["strain_rate"].iloc[0])

    stress_s = _smooth(stress, PREP_CONFIG["smooth_window"], PREP_CONFIG["smooth_method"])
    plastic_strain = strain - stress_s / max(E, 1e-6)
    plastic_strain = np.clip(plastic_strain, 0.0, None)

    if len(strain) >= 2:
        d_eps_p_d_eps = np.gradient(plastic_strain, strain, edge_order=1)
        d_sigma_d_eps = np.gradient(stress_s, strain, edge_order=1)
    else:
        d_eps_p_d_eps = np.zeros_like(strain)
        d_sigma_d_eps = np.zeros_like(strain)

    gamma_dot = d_eps_p_d_eps * strain_rate
    if PREP_CONFIG["clip_negative_gamma_dot"]:
        gamma_dot = np.clip(gamma_dot, 0.0, None)

    H = np.full_like(stress_s, np.nan)
    mask = d_eps_p_d_eps > PREP_CONFIG["min_d_eps_p_d_eps"]
    with np.errstate(divide="ignore", invalid="ignore"):
        H[mask] = d_sigma_d_eps[mask] / d_eps_p_d_eps[mask]
    H_s = _smooth_nan(H, PREP_CONFIG["h_smooth_window"], PREP_CONFIG["h_smooth_method"])

    g["stress_MPa_smooth"] = stress_s
    g["plastic_strain_smooth"] = plastic_strain
    g["gamma_dot_smooth"] = gamma_dot
    g["d_eps_p_d_eps"] = d_eps_p_d_eps
    g["d_sigma_d_eps"] = d_sigma_d_eps
    g["hardening_modulus_raw"] = H
    g["hardening_modulus_smooth"] = H_s
    g["plastic_mask"] = (plastic_strain >= PREP_CONFIG["min_plastic_strain"]) & mask & np.isfinite(H_s)

    stable_mask = g["plastic_mask"].to_numpy()
    if stable_mask.sum() >= PREP_CONFIG["stable_min_points"]:
        ps = plastic_strain[stable_mask]
        q_low, q_high = PREP_CONFIG["stable_plastic_quantile"]
        low = np.quantile(ps, q_low)
        high = np.quantile(ps, q_high)
        stable_mask = stable_mask & (plastic_strain >= low) & (plastic_strain <= high)
    g["plastic_stable_mask"] = stable_mask
    return g


def _clip_h(out_df: pd.DataFrame) -> pd.DataFrame:
    out_df = out_df.copy()
    out_df["hardening_modulus_clip"] = out_df["hardening_modulus_smooth"]

    if PREP_CONFIG["h_clip_by_material"]:
        for mat, block in out_df.groupby("material"):
            vals = block["hardening_modulus_smooth"].to_numpy()
            finite = np.isfinite(vals)
            if finite.sum() < 5:
                continue
            low_p, high_p = PREP_CONFIG["h_clip_percentile"]
            low = np.percentile(vals[finite], low_p)
            high = np.percentile(vals[finite], high_p)
            clipped = np.clip(vals, low, high)
            if PREP_CONFIG["clip_negative_h"]:
                clipped = np.clip(clipped, 0.0, None)
            out_df.loc[block.index, "hardening_modulus_clip"] = clipped
    else:
        vals = out_df["hardening_modulus_smooth"].to_numpy()
        finite = np.isfinite(vals)
        if finite.sum() >= 5:
            low_p, high_p = PREP_CONFIG["h_clip_percentile"]
            low = np.percentile(vals[finite], low_p)
            high = np.percentile(vals[finite], high_p)
            clipped = np.clip(vals, low, high)
            if PREP_CONFIG["clip_negative_h"]:
                clipped = np.clip(clipped, 0.0, None)
            out_df.loc[:, "hardening_modulus_clip"] = clipped

    out_df["plastic_mask"] = out_df["plastic_mask"] & np.isfinite(out_df["hardening_modulus_clip"])
    if "plastic_stable_mask" in out_df.columns:
        out_df["plastic_stable_mask"] = out_df["plastic_stable_mask"] & np.isfinite(out_df["hardening_modulus_clip"])

    if PREP_CONFIG["derive_h_log"]:
        h_vals = out_df["hardening_modulus_clip"].to_numpy(dtype=float)
        offset = float(PREP_CONFIG["h_log_offset"])
        h_log = np.full_like(h_vals, np.nan, dtype=float)
        positive = h_vals > 0
        if positive.any():
            h_log[positive] = np.log10(h_vals[positive] + offset)
        out_df["hardening_modulus_log"] = h_log
    return out_df


def prepare_pdml_dataset(input_csv: Path, output_all_csv: Path, output_ready_csv: Path, summary_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    all_groups = []
    summary = []
    for source_file, g in df.groupby("source_file"):
        g2 = _prepare_group_pdml(g)
        g2 = _derive_smoothed_metrics(g2)
        all_groups.append(g2)
        summary.append({
            "source_file": source_file,
            "material": g2["material"].iloc[0],
            "points_total": int(len(g2)),
            "points_plastic": int(g2["plastic_mask"].sum()),
            "points_stable": int(g2["plastic_stable_mask"].sum()),
        })

    out_all = pd.concat(all_groups, ignore_index=True)
    out_all = _clip_h(out_all)
    ready_col = PREP_CONFIG.get("ready_mask", "plastic_mask")
    if ready_col not in out_all.columns:
        raise ValueError(f"Unknown ready_mask: {ready_col}")
    out_ready = out_all[out_all[ready_col]].copy()

    output_all_csv.parent.mkdir(parents=True, exist_ok=True)
    output_ready_csv.parent.mkdir(parents=True, exist_ok=True)
    out_all.to_csv(output_all_csv, index=False)
    out_ready.to_csv(output_ready_csv, index=False)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(summary_csv, index=False)

    print("all rows:", len(out_all))
    print("plastic rows:", len(out_ready))
    print("output all:", str(output_all_csv))
    print("output ready:", str(output_ready_csv))
    print("summary:", str(summary_csv))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def split_by_curve(df: pd.DataFrame, split_ratio: Dict[str, float], seed: int) -> Tuple[List[str], List[str], List[str]]:
    curve_meta = df.groupby("source_file").first()[["material"]]
    rng = np.random.default_rng(seed)
    train_files, val_files, test_files = [], [], []

    for material in sorted(curve_meta["material"].unique()):
        files = sorted(curve_meta[curve_meta["material"] == material].index.tolist())
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * split_ratio["train"]))
        n_val = int(round(n * split_ratio["val"]))
        if n_train + n_val > n:
            n_train = max(0, n - n_val)
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    return train_files, val_files, test_files


def build_features(df: pd.DataFrame, use_log_strain_rate: bool) -> pd.DataFrame:
    df = df.copy()
    if use_log_strain_rate:
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


def filter_h_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    info: Dict[str, int] = {"input": int(len(df))}
    out = df.copy()

    if WEAK_CONFIG["h_use_stable_mask"]:
        mask_col = WEAK_CONFIG["h_stable_mask_col"]
        if mask_col not in out.columns:
            raise ValueError(f"Missing stable mask column: {mask_col}")
        out = out[out[mask_col].fillna(False)]
        info["after_stable_mask"] = int(len(out))

    if WEAK_CONFIG["h_positive_only"]:
        out = out[out[WEAK_CONFIG["target_h"]] > 0]
        info["after_positive"] = int(len(out))

    if WEAK_CONFIG["h_use_log"] and WEAK_CONFIG["h_log_column"] in out.columns:
        out = out.dropna(subset=[WEAK_CONFIG["h_log_column"]])
        info["after_log"] = int(len(out))
    else:
        out = out.dropna(subset=[WEAK_CONFIG["target_h"]])
        info["after_target"] = int(len(out))

    return out, info


def get_h_target_values(df: pd.DataFrame) -> np.ndarray:
    if WEAK_CONFIG["h_use_log"]:
        if WEAK_CONFIG["h_log_column"] in df.columns:
            return df[WEAK_CONFIG["h_log_column"]].to_numpy(dtype=float)
        h_vals = df[WEAK_CONFIG["target_h"]].to_numpy(dtype=float)
        offset = float(WEAK_CONFIG["h_log_offset"])
        return np.log10(h_vals + offset)
    return df[WEAK_CONFIG["target_h"]].to_numpy(dtype=float)


def run_weak_pdml(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    required = {"source_file", "material", "temperature", "strain_rate", "strain", "stress_MPa_smooth",
                "plastic_strain_smooth", WEAK_CONFIG["target_gamma"], WEAK_CONFIG["target_h"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan)

    train_files, val_files, test_files = split_by_curve(df, WEAK_CONFIG["split_ratio"], WEAK_CONFIG["split_seed"])
    train_df = df[df["source_file"].isin(train_files)].copy()
    val_df = df[df["source_file"].isin(val_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    X_train_all = build_features(train_df, WEAK_CONFIG["use_log_strain_rate"])
    scaler = StandardScaler()
    scaler.fit(X_train_all)

    def transform(df_sub: pd.DataFrame) -> np.ndarray:
        return scaler.transform(build_features(df_sub, WEAK_CONFIG["use_log_strain_rate"]))

    train_gamma = train_df.dropna(subset=[WEAK_CONFIG["target_gamma"]]).copy()
    val_gamma = val_df.dropna(subset=[WEAK_CONFIG["target_gamma"]]).copy()
    test_gamma = test_df.dropna(subset=[WEAK_CONFIG["target_gamma"]]).copy()

    X_train_gamma = transform(train_gamma)
    X_val_gamma = transform(val_gamma)
    X_test_gamma = transform(test_gamma)

    y_gamma_train = train_gamma[WEAK_CONFIG["target_gamma"]].to_numpy(dtype=float)
    y_gamma_val = val_gamma[WEAK_CONFIG["target_gamma"]].to_numpy(dtype=float)
    y_gamma_test = test_gamma[WEAK_CONFIG["target_gamma"]].to_numpy(dtype=float)

    train_h, info_train_h = filter_h_df(train_df)
    val_h, info_val_h = filter_h_df(val_df)
    test_h, info_test_h = filter_h_df(test_df)

    if train_h.empty:
        raise ValueError("H training set is empty; check stable/positive filters")

    X_train_h = transform(train_h)
    X_val_h = transform(val_h)
    X_test_h = transform(test_h)

    y_h_train = get_h_target_values(train_h)
    y_h_val = get_h_target_values(val_h)
    y_h_test = get_h_target_values(test_h)

    model_gamma = GradientBoostingRegressor(**WEAK_CONFIG["model_gamma"])
    model_h = GradientBoostingRegressor(**WEAK_CONFIG["model_h"])

    model_gamma.fit(X_train_gamma, y_gamma_train)
    model_h.fit(X_train_h, y_h_train)

    pred_gamma_train = model_gamma.predict(X_train_gamma)
    pred_gamma_val = model_gamma.predict(X_val_gamma)
    pred_gamma_test = model_gamma.predict(X_test_gamma)

    pred_h_train = model_h.predict(X_train_h)
    pred_h_val = model_h.predict(X_val_h)
    pred_h_test = model_h.predict(X_test_h)

    metrics = {
        "gamma": {
            "train": eval_metrics(y_gamma_train, pred_gamma_train),
            "val": eval_metrics(y_gamma_val, pred_gamma_val),
            "test": eval_metrics(y_gamma_test, pred_gamma_test),
            "samples": {
                "train": int(len(train_gamma)),
                "val": int(len(val_gamma)),
                "test": int(len(test_gamma)),
            },
        },
        "H": {
            "train": eval_metrics(y_h_train, pred_h_train),
            "val": eval_metrics(y_h_val, pred_h_val),
            "test": eval_metrics(y_h_test, pred_h_test),
            "space": "log10" if WEAK_CONFIG["h_use_log"] else "linear",
            "filter": {
                "train": info_train_h,
                "val": info_val_h,
                "test": info_test_h,
            },
        },
    }

    if WEAK_CONFIG["h_use_log"]:
        offset = float(WEAK_CONFIG["h_log_offset"])

        def to_raw(pred_log: np.ndarray) -> np.ndarray:
            return np.clip(np.power(10.0, pred_log) - offset, 0.0, None)

        y_h_train_raw = train_h[WEAK_CONFIG["target_h"]].to_numpy(dtype=float)
        y_h_val_raw = val_h[WEAK_CONFIG["target_h"]].to_numpy(dtype=float)
        y_h_test_raw = test_h[WEAK_CONFIG["target_h"]].to_numpy(dtype=float)

        metrics["H_raw"] = {
            "train": eval_metrics(y_h_train_raw, to_raw(pred_h_train)),
            "val": eval_metrics(y_h_val_raw, to_raw(pred_h_val)),
            "test": eval_metrics(y_h_test_raw, to_raw(pred_h_test)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_out = test_df.copy()
    test_out["gamma_pred"] = np.nan
    if len(test_gamma) > 0:
        test_out.loc[test_gamma.index, "gamma_pred"] = pred_gamma_test

    test_out["H_pred"] = np.nan
    if WEAK_CONFIG["h_use_log"]:
        test_out["H_pred_log"] = np.nan
        if len(test_h) > 0:
            test_out.loc[test_h.index, "H_pred_log"] = pred_h_test
            offset = float(WEAK_CONFIG["h_log_offset"])
            test_out.loc[test_h.index, "H_pred"] = np.clip(np.power(10.0, pred_h_test) - offset, 0.0, None)
    else:
        if len(test_h) > 0:
            test_out.loc[test_h.index, "H_pred"] = pred_h_test

    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

    print("weak PD-ML training done")
    print("gamma test:", metrics["gamma"]["test"])
    print("H test:", metrics["H"]["test"])
    print("outputs:", str(output_dir))


def add_material_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mats = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mats.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mats)
    return pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)


def build_features_per_material(df: pd.DataFrame, use_log_strain_rate: bool) -> pd.DataFrame:
    df = df.copy()
    if use_log_strain_rate:
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


def train_for_material(df: pd.DataFrame, output_dir: Path, material: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    mat_df = df[df["material"] == material].copy()
    if mat_df.empty:
        return {}

    train_files, val_files, test_files = split_by_curve(mat_df, PER_MATERIAL_CONFIG["split_ratio"], PER_MATERIAL_CONFIG["split_seed"])
    train_df = mat_df[mat_df["source_file"].isin(train_files)].copy()
    val_df = mat_df[mat_df["source_file"].isin(val_files)].copy()
    test_df = mat_df[mat_df["source_file"].isin(test_files)].copy()

    X_train = build_features_per_material(train_df, PER_MATERIAL_CONFIG["use_log_strain_rate"])
    X_val = build_features_per_material(val_df, PER_MATERIAL_CONFIG["use_log_strain_rate"])
    X_test = build_features_per_material(test_df, PER_MATERIAL_CONFIG["use_log_strain_rate"])

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

    model_gamma = GradientBoostingRegressor(**PER_MATERIAL_CONFIG["model_params"])
    model_H = GradientBoostingRegressor(**PER_MATERIAL_CONFIG["model_params"])

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

    test_out = test_df.copy()
    test_out["gamma_dot_pred"] = pred_test_gamma
    test_out["hardening_modulus_pred"] = pred_test_H
    test_out.to_csv(output_dir / f"{material}_test_predictions.csv", index=False)

    joblib.dump(model_gamma, output_dir / f"{material}_gbr_gamma.pkl")
    joblib.dump(model_H, output_dir / f"{material}_gbr_H.pkl")
    joblib.dump(scaler, output_dir / f"{material}_scaler.pkl")

    return metrics


def run_per_material_pdml(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    required = {"source_file", "material", "temperature", "strain_rate", "strain",
                "stress_MPa_smooth", "gamma_dot_smooth", "hardening_modulus_smooth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = add_material_features(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for material in sorted(df["material"].unique()):
        all_metrics[material] = train_for_material(df, output_dir, material)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("per-material PD-ML training done")
    print("output:", str(output_dir))


def main() -> None:
    print("=" * 60)
    print("PD-ML 训练流水线")
    print("=" * 60)

    if RUN_STEPS["derive_metrics"]:
        _ensure_exists(PATHS["merged_csv"], "合并数据")
        print("\n[1/3] 派生塑性指标...")
        derive_plastic_metrics(PATHS["merged_csv"], PATHS["metrics_csv"])
    else:
        _ensure_exists(PATHS["metrics_csv"], "塑性指标数据")

    if RUN_STEPS["prepare_dataset"]:
        _ensure_exists(PATHS["metrics_csv"], "塑性指标数据")
        print("\n[2/3] 准备 PD-ML 数据集...")
        prepare_pdml_dataset(PATHS["metrics_csv"], PATHS["pdml_with_mask_csv"], PATHS["pdml_ready_csv"], PATHS["summary_csv"])
    else:
        _ensure_exists(PATHS["pdml_ready_csv"], "PD-ML 就绪数据")

    if RUN_STEPS["weak_pdml"]:
        _ensure_exists(PATHS["pdml_ready_csv"], "PD-ML 就绪数据")
        print("\n[3/3] 弱 PD-ML 训练...")
        run_weak_pdml(PATHS["pdml_ready_csv"], PATHS["weak_output_dir"])

    if RUN_STEPS["per_material"]:
        _ensure_exists(PATHS["pdml_ready_csv"], "PD-ML 就绪数据")
        print("\n[可选] 分材料 PD-ML 训练...")
        run_per_material_pdml(PATHS["pdml_ready_csv"], PATHS["per_material_output_dir"])

    print("\n" + "=" * 60)
    print("PD-ML 流水线完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
