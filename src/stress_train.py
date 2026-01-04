#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBR应力应变训练入口（仅保留GBR基线）
包含: 数据加载、特征构建、训练、评估、结果保存
新增: 温度/应变率中值预测（用于可视化外推能力）
新增: 预测平滑处理（Savitzky-Golay滤波）
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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
# c1~c5: Johnson-Cook 模型 5 个常数（A, B, n, C, m）作为固定材料特征使用
# 元素含量：wt%，区间取中值；“≤”取上限；Al=100-其它元素之和；Al6061 为 Lot A/D/G/I 平均
MATERIAL_FEATURES = {
    "al2024": {
        "material_id": 0,
        "E_GPa": 73.1,
        "c1": 325.0,
        "c2": 0.33,
        "c3": 22.78,
        "c4": 502.0,
        "c5": 2.29e-05,
        "Si_wt": 0.50,
        "Fe_wt": 0.50,
        "Cu_wt": 4.35,
        "Mn_wt": 0.60,
        "Mg_wt": 1.50,
        "Cr_wt": 0.10,
        "Zn_wt": 0.25,
        "Ti_wt": 2.00,
        "Zr_wt": 0.00,
        "V_wt": 0.00,
        "Al_wt": 90.20,
    },
    "al2219": {
        "material_id": 1,
        "E_GPa": 73.1,
        "c1": 350.0,
        "c2": 0.33,
        "c3": 22.84,
        "c4": 543.0,
        "c5": 2.23e-05,
        "Si_wt": 0.20,
        "Fe_wt": 0.30,
        "Cu_wt": 6.30,
        "Mn_wt": 0.30,
        "Mg_wt": 0.02,
        "Cr_wt": 0.00,
        "Zn_wt": 0.10,
        "Ti_wt": 0.06,
        "Zr_wt": 0.175,
        "V_wt": 0.10,
        "Al_wt": 92.445,
    },
    "al6061": {
        "material_id": 2,
        "E_GPa": 68.9,
        "c1": 276.0,
        "c2": 0.33,
        "c3": 22.70,
        "c4": 582.0,
        "c5": 2.36e-05,
        "Si_wt": 0.63,
        "Fe_wt": 0.2225,
        "Cu_wt": 0.195,
        "Mn_wt": 0.045,
        "Mg_wt": 0.8775,
        "Cr_wt": 0.0525,
        "Zn_wt": 0.02,
        "Ti_wt": 0.015,
        "Zr_wt": 0.00,
        "V_wt": 0.00,
        "Al_wt": 97.9425,
    },
    "al7075": {
        "material_id": 3,
        "E_GPa": 71.7,
        "c1": 503.0,
        "c2": 0.33,
        "c3": 22.81,
        "c4": 477.0,
        "c5": 2.32e-05,
        "Si_wt": 0.40,
        "Fe_wt": 0.50,
        "Cu_wt": 1.60,
        "Mn_wt": 0.30,
        "Mg_wt": 2.50,
        "Cr_wt": 0.23,
        "Zn_wt": 5.60,
        "Ti_wt": 0.20,
        "Zr_wt": 0.00,
        "V_wt": 0.00,
        "Al_wt": 88.67,
    },
}

COMPOSITION_FEATURE_COLS = [
    "Si_wt",
    "Fe_wt",
    "Cu_wt",
    "Mn_wt",
    "Mg_wt",
    "Cr_wt",
    "Zn_wt",
    "Ti_wt",
    "Zr_wt",
    "V_wt",
    "Al_wt",
]


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
    ] + COMPOSITION_FEATURE_COLS
    return df[feature_cols].astype(float)


def gbr_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def build_temperature_midpoints(temps: list[float]) -> list[float]:
    """计算相邻温度的中值点"""
    uniq = sorted(set(temps))
    if len(uniq) < 2:
        return []
    midpoints = []
    for i in range(len(uniq) - 1):
        midpoints.append((uniq[i] + uniq[i + 1]) / 2.0)
    return midpoints


def build_rate_midpoints(rates: list[float]) -> list[float]:
    """计算相邻应变率的中值点"""
    uniq = sorted(set(rates))
    if len(uniq) < 2:
        return []
    midpoints = []
    for i in range(len(uniq) - 1):
        midpoints.append((uniq[i] + uniq[i + 1]) / 2.0)
    return midpoints


def build_strain_grid(df_group: pd.DataFrame, n_points: int = 200) -> np.ndarray | None:
    """构建应变网格（取所有曲线的公共应变范围）"""
    per_curve_max = df_group.groupby("source_file")["strain"].max().to_numpy()
    per_curve_min = df_group.groupby("source_file")["strain"].min().to_numpy()
    if len(per_curve_max) == 0:
        return None
    max_strain = float(np.min(per_curve_max))
    min_strain = float(np.max(per_curve_min))
    if max_strain <= min_strain:
        return None
    return np.linspace(min_strain, max_strain, n_points)


def smooth_prediction(y: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    """使用Savitzky-Golay滤波器平滑预测曲线

    Args:
        y: 预测值数组
        window: 滤波窗口大小（必须为奇数）
        polyorder: 多项式阶数

    Returns:
        平滑后的预测值
    """
    if len(y) < window:
        window = len(y) // 2 * 2 + 1  # 确保是奇数且小于数据长度
        if window < 5:
            return y  # 数据太少，不平滑
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def interpolate_stress_by_condition(
    model, scaler, material: str, strain_grid: np.ndarray,
    fixed_param: str, fixed_value: float,
    vary_param: str, observed_values: list[float], target_value: float
) -> np.ndarray:
    """通过线性插值计算中间条件的应力预测

    Args:
        model: 训练好的GBR模型
        scaler: 特征归一化器
        material: 材料名称
        strain_grid: 应变网格
        fixed_param: 固定参数名 ('temperature' or 'strain_rate')
        fixed_value: 固定参数值
        vary_param: 变化参数名 ('temperature' or 'strain_rate')
        observed_values: 已观测的变化参数值列表
        target_value: 目标插值点

    Returns:
        插值得到的应力预测
    """
    # 找到target_value两侧的观测值
    sorted_vals = sorted(observed_values)
    lower_val = None
    upper_val = None
    for i, v in enumerate(sorted_vals):
        if v <= target_value:
            lower_val = v
        if v >= target_value and upper_val is None:
            upper_val = v
            break

    if lower_val is None or upper_val is None or lower_val == upper_val:
        # 无法插值，退回直接预测
        return None

    # 计算两个边界条件的应力预测
    def predict_single(vary_value):
        if fixed_param == "strain_rate":
            df = pd.DataFrame({
                "material": material,
                "temperature": vary_value,
                "strain_rate": fixed_value,
                "strain": strain_grid,
            })
        else:
            df = pd.DataFrame({
                "material": material,
                "temperature": fixed_value,
                "strain_rate": vary_value,
                "strain": strain_grid,
            })
        df = gbr_add_material_features(df)
        X = gbr_build_features(df)
        X_s = scaler.transform(X)
        return model.predict(X_s)

    stress_lower = predict_single(lower_val)
    stress_upper = predict_single(upper_val)

    # 线性插值
    t = (target_value - lower_val) / (upper_val - lower_val)
    stress_interp = stress_lower * (1 - t) + stress_upper * t

    return stress_interp


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

    # ==================== 中值预测（用于外推可视化）====================
    # 使用线性插值而非直接预测，解决GBR阶梯状预测问题
    print("\n生成温度/应变率中值预测（线性插值）...")
    midpoint_predictions = []

    # 各材料用于中值预测的文件模式（只使用完整曲线，排除塑性段强化数据）
    midpoint_file_patterns = {
        "al2024": ["fig2"],  # 只用fig2系列，fig8是塑性段强化数据
        "al2219": ["fig3"],
        "al7075": ["fig1"],
        "al6061": None,  # 使用全部数据
    }

    for material in sorted(df["material"].unique()):
        df_mat = df[df["material"] == material]

        # 过滤文件模式（用于中值预测的应变网格计算）
        patterns = midpoint_file_patterns.get(material)
        if patterns:
            mask = df_mat["source_file"].str.lower().apply(
                lambda x: any(p in x.lower() for p in patterns)
            )
            df_mat_filtered = df_mat[mask].copy()
            print(f"  {material}: 中值预测使用 {patterns} 模式 ({len(df_mat_filtered)}/{len(df_mat)} 行)")
        else:
            df_mat_filtered = df_mat

        # 对每个应变率都生成温度中值预测
        all_rates = sorted(df_mat_filtered["strain_rate"].unique())
        for rate in all_rates:
            df_rate_group = df_mat_filtered[np.isclose(df_mat_filtered["strain_rate"], rate)]
            obs_temps = sorted(df_rate_group["temperature"].unique().tolist())

            if len(obs_temps) < 2:
                continue  # 至少需要2个温度才能计算中值

            mid_temps = build_temperature_midpoints(obs_temps)

            if mid_temps:
                strain_grid = build_strain_grid(df_rate_group, n_points=200)
                if strain_grid is not None:
                    for mid_temp in mid_temps:
                        # 使用线性插值计算中值温度的应力
                        stress_pred = interpolate_stress_by_condition(
                            model, scaler, material, strain_grid,
                            fixed_param="strain_rate", fixed_value=rate,
                            vary_param="temperature", observed_values=obs_temps,
                            target_value=mid_temp
                        )
                        if stress_pred is None:
                            continue
                        stress_pred = smooth_prediction(stress_pred)  # 平滑处理

                        for i, (s, sp) in enumerate(zip(strain_grid, stress_pred)):
                            midpoint_predictions.append({
                                "material": material,
                                "temperature": mid_temp,
                                "strain_rate": rate,
                                "strain": float(s),
                                "stress_pred": float(sp),
                                "pred_type": "temp_midpoint",
                                "observed_temps": str(obs_temps),
                            })
                    print(f"  {material}: 温度中值 {mid_temps} @ rate={rate}")

        # 找出该材料覆盖应变率最多的温度（用于应变率中值预测）
        temp_rate_counts = df_mat_filtered.groupby("temperature")["strain_rate"].nunique()
        if not temp_rate_counts.empty:
            best_temp = float(temp_rate_counts.idxmax())
            df_temp_group = df_mat_filtered[np.isclose(df_mat_filtered["temperature"], best_temp)]
            obs_rates = sorted(df_temp_group["strain_rate"].unique().tolist())
            mid_rates = build_rate_midpoints(obs_rates)

            if mid_rates:
                strain_grid = build_strain_grid(df_temp_group, n_points=200)
                if strain_grid is not None:
                    for mid_rate in mid_rates:
                        # 使用线性插值计算中值应变率的应力
                        stress_pred = interpolate_stress_by_condition(
                            model, scaler, material, strain_grid,
                            fixed_param="temperature", fixed_value=best_temp,
                            vary_param="strain_rate", observed_values=obs_rates,
                            target_value=mid_rate
                        )
                        if stress_pred is None:
                            continue
                        stress_pred = smooth_prediction(stress_pred)  # 平滑处理

                        for i, (s, sp) in enumerate(zip(strain_grid, stress_pred)):
                            midpoint_predictions.append({
                                "material": material,
                                "temperature": best_temp,
                                "strain_rate": mid_rate,
                                "strain": float(s),
                                "stress_pred": float(sp),
                                "pred_type": "rate_midpoint",
                                "observed_rates": str(obs_rates),
                            })
                    print(f"  {material}: 应变率中值 {mid_rates} @ temp={best_temp}")

    if midpoint_predictions:
        midpoint_df = pd.DataFrame(midpoint_predictions)
        midpoint_df.to_csv(output_dir / "midpoint_predictions.csv", index=False)
        print(f"  保存中值预测: {len(midpoint_predictions)} 条记录")

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
