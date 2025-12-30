#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM应力应变曲线预测一体化脚本

功能:
1. 读取合并CSV并恢复test_type
2. 数据预处理（物理约束、序列化、归一化、分层划分）
3. LSTM模型训练（含加权损失、早停、学习率调度）
4. 评估与可视化输出
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

try:
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


DEFAULT_SEED = 42
SEQ_LEN = 50

MATERIAL_PROPERTIES = {
    "Al6061": {"E_GPa": 69, "yield_MPa": 275, "poisson": 0.33},
    "Al2024": {"E_GPa": 73, "yield_MPa": 325, "poisson": 0.33},
    "Al7075": {"E_GPa": 72, "yield_MPa": 505, "poisson": 0.33},
}


def set_seed(seed: int) -> None:
    """固定随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_test_type(source_file: str) -> str:
    """从source_file恢复test_type"""
    if source_file.startswith("T_"):
        return "tension"
    if source_file.startswith("P_"):
        return "plane_strain"
    if source_file.startswith("al2024_"):
        return "tension"
    if source_file.startswith("al7075_"):
        return "compression"
    return "unknown"


def _monotonicity_ratio(values: np.ndarray) -> float:
    """计算单调性违例比例"""
    if len(values) < 3:
        return 0.0
    direction = 1 if (values[-1] - values[0]) >= 0 else -1
    diffs = np.diff(values)
    violations = np.sum(diffs * direction < 0)
    return violations / max(len(diffs), 1)


def validate_curve_physics(strain: np.ndarray, stress: np.ndarray) -> bool:
    """物理约束验证（放宽标准以保留更多数据）"""
    # 基本范围检查
    if np.nanmax(np.abs(strain)) > 2.0:
        return False
    if np.nanmax(np.abs(stress)) > 1500:  # 放宽到1500 MPa
        return False

    # 检查是否有足够的数据点
    if len(strain) < 3 or len(stress) < 3:
        return False

    # 检查是否有NaN或Inf
    if np.any(np.isnan(strain)) or np.any(np.isnan(stress)):
        return False
    if np.any(np.isinf(strain)) or np.any(np.isinf(stress)):
        return False

    # ⭐ 新增：过滤负应力数据（除非超过90%点为负，则可能是压缩测试）
    negative_ratio = np.sum(stress < 0) / len(stress)
    if 0.1 < negative_ratio < 0.9:  # 有部分负应力（异常）
        return False

    # 如果全是负应力（压缩测试），取绝对值
    if negative_ratio >= 0.9:
        stress = np.abs(stress)

    # 弹性模量估算（放宽范围，接受更多数据）
    small_mask = np.abs(strain) <= 0.02
    if np.sum(small_mask) >= 2:
        try:
            slope, _ = np.polyfit(strain[small_mask], stress[small_mask], 1)
            # 极度放宽范围：1-300 GPa（允许异常数据）
            if not (1000 <= abs(slope) <= 300000):
                return False
        except:
            pass  # 拟合失败也接受

    # 单调性允许80%非单调点（极度放宽）
    if _monotonicity_ratio(stress) > 0.8:
        return False

    return True


def detect_and_convert_to_true_stress(
    strain_eng: np.ndarray,
    stress_eng: np.ndarray,
    threshold: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    检测工程应力并转换为真应力-应变

    Args:
        strain_eng: 工程应变
        stress_eng: 工程应力 (MPa)
        threshold: 下降比例阈值,超过此值判定为工程应力

    Returns:
        (真应变, 真应力, 是否进行了转换)
    """
    # 检查是否有明显的颈缩下降(工程应力特征)
    max_idx = np.argmax(stress_eng)
    if max_idx < len(stress_eng) - 3:  # 排除末尾噪声
        max_stress = stress_eng[max_idx]
        end_stress = stress_eng[-1]
        drop_ratio = (max_stress - end_stress) / max_stress if max_stress > 0 else 0

        is_engineering = drop_ratio > threshold
    else:
        is_engineering = False

    if is_engineering:
        # 转换为真应力-应变
        # 只保留应变>= -0.01的数据点(避免过大的负值)
        valid_mask = strain_eng >= -0.01
        if np.sum(valid_mask) < len(strain_eng) * 0.5:
            # 如果超过一半数据被过滤,保持原样
            return strain_eng, stress_eng, False

        strain_eng = strain_eng[valid_mask]
        stress_eng = stress_eng[valid_mask]

        # 确保应变非负
        strain_eng_abs = np.maximum(strain_eng, 0)

        # 真应变: ε_true = ln(1 + ε_eng)
        strain_true = np.log(1 + strain_eng_abs)

        # 真应力: σ_true = σ_eng × (1 + ε_eng)
        stress_true = stress_eng * (1 + strain_eng_abs)

        return strain_true, stress_true, True
    else:
        # 已经是真应力或无颈缩,无需转换
        return strain_eng, stress_eng, False


def interpolate_curve(
    strain: np.ndarray,
    stress: np.ndarray,
    num_points: int = SEQ_LEN,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """线性插值到固定长度"""
    if len(strain) < 2:
        return None

    # 去除重复的应变值(保留第一个)
    unique_indices = np.concatenate([[True], np.diff(strain) != 0])
    strain = strain[unique_indices]
    stress = stress[unique_indices]

    if len(strain) < 2:
        return None

    x_min, x_max = float(np.min(strain)), float(np.max(strain))
    if np.isclose(x_min, x_max):
        return None

    x_new = np.linspace(x_min, x_max, num_points)

    if _HAS_SCIPY:
        try:
            interp_fn = interp1d(strain, stress, kind="linear", fill_value="extrapolate")
            y_new = interp_fn(x_new)
        except Exception:
            # scipy插值失败,回退到numpy
            y_new = np.interp(x_new, strain, stress)
    else:
        y_new = np.interp(x_new, strain, stress)

    # 检查并处理NaN值
    if np.isnan(y_new).any():
        return None

    return x_new, y_new


def load_curves(csv_path: Path) -> List[Dict[str, Any]]:
    """加载CSV并构建曲线样本"""
    print("\n" + "=" * 70)
    print("[DATA] 读取合并数据并恢复test_type...")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    df["test_type"] = df["source_file"].astype(str).apply(extract_test_type)
    df = df[df["test_type"] != "unknown"].copy()

    curves = []
    skipped = 0
    converted_count = 0  # 统计转换数量

    for curve_id, group in df.groupby("curve_id"):
        group = group.dropna(subset=["strain", "stress_MPa", "temperature_C", "strain_rate"])
        if group.empty:
            skipped += 1
            continue

        material = group["material"].iloc[0]
        mat_props = MATERIAL_PROPERTIES.get(material)
        if mat_props is None:
            skipped += 1
            continue
        temperature = float(group["temperature_C"].iloc[0])
        strain_rate = float(group["strain_rate"].iloc[0])
        test_type = group["test_type"].iloc[0]

        # 去重与排序
        group = (
            group.groupby("strain", as_index=False)["stress_MPa"]
            .mean()
            .sort_values("strain")
        )
        strain = group["strain"].to_numpy(dtype=np.float64)
        stress = group["stress_MPa"].to_numpy(dtype=np.float64)

        # ⭐ 自动检测并转换工程应力到真应力
        strain, stress, was_converted = detect_and_convert_to_true_stress(strain, stress)
        if was_converted:
            converted_count += 1

        if not validate_curve_physics(strain, stress):
            skipped += 1
            continue

        interpolated = interpolate_curve(strain, stress, SEQ_LEN)
        if interpolated is None:
            skipped += 1
            continue

        strain_seq, stress_seq = interpolated
        curves.append(
            {
                "curve_id": curve_id,
                "material": material,
                "temperature_C": temperature,
                "strain_rate": strain_rate,
                "test_type": test_type,
                "strain_seq": strain_seq,
                "stress_seq": stress_seq,
                "E_GPa": mat_props["E_GPa"],
                "yield_MPa": mat_props["yield_MPa"],
                "poisson": mat_props["poisson"],
            }
        )

    print(f"  OK 有效曲线: {len(curves)} 条 | 过滤曲线: {skipped} 条")
    print(f"  >> 工程应力->真应力转换: {converted_count} 条")
    return curves


def stratified_split(
    curves: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """按材料类型分层划分"""
    rng = random.Random(seed)
    material_groups: Dict[str, List[int]] = {}
    for idx, curve in enumerate(curves):
        material_groups.setdefault(curve["material"], []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for indices in material_groups.values():
        rng.shuffle(indices)
        total = len(indices)
        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
        train_idx.extend(indices[:train_end])
        val_idx.extend(indices[train_end:val_end])
        test_idx.extend(indices[val_end:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def fit_scalers(
    curves: List[Dict[str, Any]],
    train_indices: List[int],
) -> Dict[str, Any]:
    """拟合归一化器与编码器"""
    train_curves = [curves[i] for i in train_indices]

    strain_all = np.concatenate([c["strain_seq"] for c in train_curves])[:, None]
    stress_all = np.concatenate([c["stress_seq"] for c in train_curves])[:, None]

    strain_scaler = MinMaxScaler(feature_range=(-1, 1))
    stress_scaler = MinMaxScaler(feature_range=(-1, 1))
    strain_scaler.fit(strain_all)
    stress_scaler.fit(stress_all)

    material_encoder = LabelEncoder()
    test_type_encoder = LabelEncoder()
    material_encoder.fit([c["material"] for c in curves])
    test_type_encoder.fit([c["test_type"] for c in curves])

    cond_scaler = StandardScaler()
    cond_values = np.array(
        [
            [
                c["temperature_C"],
                c["strain_rate"],
                float(material_encoder.transform([c["material"]])[0]),
                float(test_type_encoder.transform([c["test_type"]])[0]),
                c["E_GPa"],
                c["yield_MPa"],
                c["poisson"],
            ]
            for c in train_curves
        ],
        dtype=np.float64,
    )
    cond_scaler.fit(cond_values)

    return {
        "strain_scaler": strain_scaler,
        "stress_scaler": stress_scaler,
        "cond_scaler": cond_scaler,
        "material_encoder": material_encoder,
        "test_type_encoder": test_type_encoder,
    }


def build_dataset(curves: List[Dict[str, Any]], indices: List[int], scalers: Dict[str, Any]) -> Dict[str, Any]:
    """构建训练/验证/测试数据集字典"""
    strain_scaler = scalers["strain_scaler"]
    stress_scaler = scalers["stress_scaler"]
    cond_scaler = scalers["cond_scaler"]
    material_encoder = scalers["material_encoder"]
    test_type_encoder = scalers["test_type_encoder"]

    strain_seq_list = []
    stress_seq_list = []
    conditions_list = []
    material_ids = []
    test_type_ids = []
    curve_ids = []
    strain_raw_list = []
    stress_raw_list = []
    material_labels = []
    test_type_labels = []
    temperatures = []
    strain_rates = []
    e_gpa_list = []
    yield_mpa_list = []
    poisson_list = []

    for idx in indices:
        curve = curves[idx]
        strain_seq = curve["strain_seq"][:, None]
        stress_seq = curve["stress_seq"][:, None]

        strain_scaled = strain_scaler.transform(strain_seq)
        stress_scaled = stress_scaler.transform(stress_seq)

        material_id = material_encoder.transform([curve["material"]])[0]
        test_type_id = test_type_encoder.transform([curve["test_type"]])[0]

        cond_raw = np.array(
            [
                [
                    curve["temperature_C"],
                    curve["strain_rate"],
                    float(material_id),
                    float(test_type_id),
                    curve["E_GPa"],
                    curve["yield_MPa"],
                    curve["poisson"],
                ]
            ],
            dtype=np.float64,
        )
        cond_scaled = cond_scaler.transform(cond_raw)[0]
        conditions = np.array(
            [
                cond_scaled[0],
                cond_scaled[1],
                float(material_id),
                float(test_type_id),
                curve["E_GPa"],
                curve["yield_MPa"],
                curve["poisson"],
            ],
            dtype=np.float32,
        )

        strain_seq_list.append(strain_scaled.astype(np.float32))
        stress_seq_list.append(stress_scaled.astype(np.float32))
        conditions_list.append(conditions)
        material_ids.append(material_id)
        test_type_ids.append(test_type_id)
        curve_ids.append(curve["curve_id"])
        strain_raw_list.append(curve["strain_seq"].astype(np.float32))
        stress_raw_list.append(curve["stress_seq"].astype(np.float32))
        material_labels.append(curve["material"])
        test_type_labels.append(curve["test_type"])
        temperatures.append(curve["temperature_C"])
        strain_rates.append(curve["strain_rate"])
        e_gpa_list.append(curve["E_GPa"])
        yield_mpa_list.append(curve["yield_MPa"])
        poisson_list.append(curve["poisson"])

    return {
        "strain_seq": np.stack(strain_seq_list),
        "stress_seq": np.stack(stress_seq_list),
        "conditions": np.stack(conditions_list),
        "material_ids": np.array(material_ids, dtype=np.int64),
        "test_type_ids": np.array(test_type_ids, dtype=np.int64),
        "curve_ids": curve_ids,
        "strain_raw": np.stack(strain_raw_list),
        "stress_raw": np.stack(stress_raw_list),
        "material_labels": material_labels,
        "test_type_labels": test_type_labels,
        "temperature_C": np.array(temperatures, dtype=np.float32),
        "strain_rate": np.array(strain_rates, dtype=np.float32),
        "E_GPa": np.array(e_gpa_list, dtype=np.float32),
        "yield_MPa": np.array(yield_mpa_list, dtype=np.float32),
        "poisson": np.array(poisson_list, dtype=np.float32),
    }


class StressStrainDataset(Dataset):
    """应力应变数据集"""

    def __init__(self, data: dict, scalers: Optional[Dict[str, Any]] = None):
        self.strain_seq = torch.from_numpy(data["strain_seq"])
        self.stress_seq = torch.from_numpy(data["stress_seq"])
        self.conditions = torch.from_numpy(data["conditions"])
        self.material_ids = torch.from_numpy(data["material_ids"])
        self.strain_raw = torch.from_numpy(data["strain_raw"])
        self.stress_raw = torch.from_numpy(data["stress_raw"])
        self.temperature_raw = torch.from_numpy(data["temperature_C"])
        self.strain_rate_raw = torch.from_numpy(data["strain_rate"])
        self.e_gpa = torch.from_numpy(data["E_GPa"])
        self.yield_mpa = torch.from_numpy(data["yield_MPa"])
        self.poisson = torch.from_numpy(data["poisson"])
        self.stress_scaler = scalers["stress_scaler"] if scalers else None

    def __len__(self) -> int:
        return len(self.strain_seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "strain_seq": self.strain_seq[idx],
            "stress_seq": self.stress_seq[idx],
            "conditions": self.conditions[idx],
            "materials": self.material_ids[idx],
            "strain_seq_raw": self.strain_raw[idx],
            "stress_seq_raw": self.stress_raw[idx],
            "temperature_raw": self.temperature_raw[idx],
            "strain_rate_raw": self.strain_rate_raw[idx],
            "E_GPa": self.e_gpa[idx],
            "yield_MPa": self.yield_mpa[idx],
            "poisson": self.poisson[idx],
        }


class StressStrainLSTM(nn.Module):
    """应力应变预测LSTM"""

    def __init__(self, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.condition_embed = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, strain_seq: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(strain_seq)
        cond_embed = self.condition_embed(conditions)
        cond_expand = cond_embed.unsqueeze(1).repeat(1, self.seq_len, 1)
        fused = torch.cat([lstm_out, cond_expand], dim=-1)
        return self.fc(fused)


class PhysicsAwareMSELoss(nn.Module):
    """物理约束增强的加权MSE"""

    def __init__(
        self,
        material_weights: torch.Tensor,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("material_weights", material_weights.float())
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch_data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        material_ids = batch_data["material_ids"]
        mse_loss = self._weighted_mse(pred, target, material_ids)

        if self.beta <= 0:
            physics_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
            total_loss = self.alpha * mse_loss
        else:
            pred_raw = self._inverse_transform_stress(
                pred, batch_data.get("stress_scaler")
            ).squeeze(-1)
            physics_loss = (
                self._elastic_modulus_constraint(
                    batch_data["strain_seq_raw"],
                    pred_raw,
                    batch_data["E_GPa"],
                )
                + self._monotonicity_constraint(pred_raw)
                + self._temperature_softening_constraint(
                    pred_raw,
                    batch_data["temperature_raw"],
                    batch_data["strain_rate_raw"],
                    material_ids,
                )
                + self._strain_rate_sensitivity_constraint(
                    pred_raw,
                    batch_data["temperature_raw"],
                    batch_data["strain_rate_raw"],
                    material_ids,
                )
            )
            total_loss = self.alpha * mse_loss + self.beta * physics_loss

        metrics = {
            "mse_loss": float(mse_loss.detach().item()),
            "physics_loss": float(physics_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss, metrics

    def _weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        material_ids: torch.Tensor,
    ) -> torch.Tensor:
        mse = (pred - target) ** 2
        mse = mse.mean(dim=(1, 2))
        weights = self.material_weights[material_ids]
        return (mse * weights).mean()

    def _inverse_transform_stress(
        self,
        stress_norm: torch.Tensor,
        scaler: Optional[MinMaxScaler],
    ) -> torch.Tensor:
        """MinMaxScaler逆变换：[-1,1] → [MPa]"""
        if scaler is None:
            return stress_norm
        min_val = torch.tensor(
            scaler.data_min_, device=stress_norm.device, dtype=stress_norm.dtype
        )
        max_val = torch.tensor(
            scaler.data_max_, device=stress_norm.device, dtype=stress_norm.dtype
        )
        stress_01 = (stress_norm + 1) / 2
        return stress_01 * (max_val - min_val) + min_val

    def _elastic_modulus_constraint(
        self,
        strain_raw: torch.Tensor,
        stress_raw: torch.Tensor,
        e_gpa: torch.Tensor,
    ) -> torch.Tensor:
        """弹性模量约束：小应变区斜率≈真实E"""
        batch_size = strain_raw.shape[0]
        e_true = e_gpa.view(-1).to(stress_raw.device) * 1000

        losses: List[torch.Tensor] = []
        for i in range(batch_size):
            mask = strain_raw[i] < 0.02
            if int(mask.sum().item()) < 2:
                continue
            eps = strain_raw[i][mask]
            sig = stress_raw[i][mask]
            eps_mean = eps.mean()
            sig_mean = sig.mean()
            covariance = ((eps - eps_mean) * (sig - sig_mean)).sum()
            variance = ((eps - eps_mean) ** 2).sum()
            e_pred = covariance / (variance + 1e-8)
            error = torch.abs(e_pred - e_true[i]) / (e_true[i] + 1e-8)
            losses.append(error)

        if not losses:
            return torch.zeros((), device=stress_raw.device, dtype=stress_raw.dtype)
        return torch.stack(losses).mean()

    def _monotonicity_constraint(self, stress_raw: torch.Tensor) -> torch.Tensor:
        """单调性约束：惩罚应力下降"""
        delta_stress = stress_raw[:, 1:] - stress_raw[:, :-1]
        violations = torch.relu(-delta_stress)
        return violations.mean()

    def _temperature_softening_constraint(
        self,
        stress_raw: torch.Tensor,
        temperature: torch.Tensor,
        strain_rate: torch.Tensor,
        material_ids: torch.Tensor,
    ) -> torch.Tensor:
        """温度软化约束：高温应力 < 低温应力"""
        batch_size = stress_raw.shape[0]
        violations: List[torch.Tensor] = []

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if int(material_ids[i].item()) != int(material_ids[j].item()):
                    continue
                sr_i = float(strain_rate[i].item())
                sr_j = float(strain_rate[j].item())
                sr_ratio = max(sr_i, sr_j) / (min(sr_i, sr_j) + 1e-8)
                if sr_ratio > 1.2:
                    continue
                temp_i = float(temperature[i].item())
                temp_j = float(temperature[j].item())
                if abs(temp_i - temp_j) < 10:
                    continue

                if temp_i > temp_j:
                    high_t_stress = stress_raw[i]
                    low_t_stress = stress_raw[j]
                else:
                    high_t_stress = stress_raw[j]
                    low_t_stress = stress_raw[i]

                violation = torch.relu(high_t_stress - low_t_stress).mean()
                violations.append(violation)

        if not violations:
            return torch.zeros((), device=stress_raw.device, dtype=stress_raw.dtype)
        return torch.stack(violations).mean()

    def _strain_rate_sensitivity_constraint(
        self,
        stress_raw: torch.Tensor,
        temperature: torch.Tensor,
        strain_rate: torch.Tensor,
        material_ids: torch.Tensor,
    ) -> torch.Tensor:
        """应变率敏感性约束：高应变率应力 > 低应变率应力"""
        batch_size = stress_raw.shape[0]
        violations: List[torch.Tensor] = []

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if int(material_ids[i].item()) != int(material_ids[j].item()):
                    continue
                temp_i = float(temperature[i].item())
                temp_j = float(temperature[j].item())
                if abs(temp_i - temp_j) > 10:
                    continue
                sr_i = float(strain_rate[i].item())
                sr_j = float(strain_rate[j].item())
                sr_ratio = max(sr_i, sr_j) / (min(sr_i, sr_j) + 1e-8)
                if sr_ratio < 1.2:
                    continue

                if sr_i > sr_j:
                    high_sr_stress = stress_raw[i]
                    low_sr_stress = stress_raw[j]
                else:
                    high_sr_stress = stress_raw[j]
                    low_sr_stress = stress_raw[i]

                violation = torch.relu(low_sr_stress - high_sr_stress).mean()
                violations.append(violation)

        if not violations:
            return torch.zeros((), device=stress_raw.device, dtype=stress_raw.dtype)
        return torch.stack(violations).mean()


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: PhysicsAwareMSELoss,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = criterion.to(device)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5
        )

        self.best_val_loss = float("inf")
        self.patience = 10
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "train_physics": [],
            "val_mse": [],
            "val_physics": [],
            "lr": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        metrics_sum = {"total_loss": 0.0, "mse_loss": 0.0, "physics_loss": 0.0}
        for batch in self.train_loader:
            strain_seq = batch["strain_seq"].to(self.device)
            stress_seq = batch["stress_seq"].to(self.device)
            conditions = batch["conditions"].to(self.device)
            materials = batch["materials"].to(self.device)
            strain_seq_raw = batch["strain_seq_raw"].to(self.device)
            stress_seq_raw = batch["stress_seq_raw"].to(self.device)
            temperature_raw = batch["temperature_raw"].to(self.device)
            strain_rate_raw = batch["strain_rate_raw"].to(self.device)
            e_gpa = batch["E_GPa"].to(self.device)
            yield_mpa = batch["yield_MPa"].to(self.device)
            poisson = batch["poisson"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(strain_seq, conditions)
            batch_data = {
                "material_ids": materials,
                "strain_seq_raw": strain_seq_raw,
                "stress_seq_raw": stress_seq_raw,
                "temperature_raw": temperature_raw,
                "strain_rate_raw": strain_rate_raw,
                "E_GPa": e_gpa,
                "yield_MPa": yield_mpa,
                "poisson": poisson,
                "stress_scaler": self.train_loader.dataset.stress_scaler,
            }
            loss, metrics = self.criterion(pred, stress_seq, batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            metrics_sum["total_loss"] += metrics["total_loss"]
            metrics_sum["mse_loss"] += metrics["mse_loss"]
            metrics_sum["physics_loss"] += metrics["physics_loss"]

        divisor = max(len(self.train_loader), 1)
        return {k: v / divisor for k, v in metrics_sum.items()}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics_sum = {"total_loss": 0.0, "mse_loss": 0.0, "physics_loss": 0.0}
        with torch.no_grad():
            for batch in self.val_loader:
                strain_seq = batch["strain_seq"].to(self.device)
                stress_seq = batch["stress_seq"].to(self.device)
                conditions = batch["conditions"].to(self.device)
                materials = batch["materials"].to(self.device)
                strain_seq_raw = batch["strain_seq_raw"].to(self.device)
                stress_seq_raw = batch["stress_seq_raw"].to(self.device)
                temperature_raw = batch["temperature_raw"].to(self.device)
                strain_rate_raw = batch["strain_rate_raw"].to(self.device)
                e_gpa = batch["E_GPa"].to(self.device)
                yield_mpa = batch["yield_MPa"].to(self.device)
                poisson = batch["poisson"].to(self.device)
                pred = self.model(strain_seq, conditions)
                batch_data = {
                    "material_ids": materials,
                    "strain_seq_raw": strain_seq_raw,
                    "stress_seq_raw": stress_seq_raw,
                    "temperature_raw": temperature_raw,
                    "strain_rate_raw": strain_rate_raw,
                    "E_GPa": e_gpa,
                    "yield_MPa": yield_mpa,
                    "poisson": poisson,
                    "stress_scaler": self.val_loader.dataset.stress_scaler,
                }
                loss, metrics = self.criterion(pred, stress_seq, batch_data)
                metrics_sum["total_loss"] += metrics["total_loss"]
                metrics_sum["mse_loss"] += metrics["mse_loss"]
                metrics_sum["physics_loss"] += metrics["physics_loss"]
        divisor = max(len(self.val_loader), 1)
        return {k: v / divisor for k, v in metrics_sum.items()}

    def train(self, epochs: int, save_path: Path) -> dict:
        print("\n" + "=" * 60)
        print(f"开始训练 - 总Epoch数: {epochs}")
        print("=" * 60 + "\n")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics["total_loss"]
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["total_loss"])
            self.history["val_loss"].append(val_metrics["total_loss"])
            self.history["train_mse"].append(train_metrics["mse_loss"])
            self.history["train_physics"].append(train_metrics["physics_loss"])
            self.history["val_mse"].append(val_metrics["mse_loss"])
            self.history["val_physics"].append(val_metrics["physics_loss"])
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Total: {train_metrics['total_loss']:.6f} | "
                f"Train MSE: {train_metrics['mse_loss']:.6f} | "
                f"Train Physics: {train_metrics['physics_loss']:.6f} | "
                f"Val Total: {val_metrics['total_loss']:.6f} | "
                f"Val MSE: {val_metrics['mse_loss']:.6f} | "
                f"Val Physics: {val_metrics['physics_loss']:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "history": self.history,
                    },
                    save_path,
                )
                print(f"  → 保存最佳模型 (Val Loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping触发 (patience={self.patience})")
                    break

        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
        print("=" * 60 + "\n")
        return self.history


def prepare_datasets(data_path: Path, output_dir: Path, seed: int) -> Dict[str, Any]:
    """完整数据预处理流程"""
    curves = load_curves(data_path)
    train_idx, val_idx, test_idx = stratified_split(curves, seed=seed)

    scalers = fit_scalers(curves, train_idx)
    train_data = build_dataset(curves, train_idx, scalers)
    val_data = build_dataset(curves, val_idx, scalers)
    test_data = build_dataset(curves, test_idx, scalers)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_data, output_dir / "train_dataset.pkl")
    joblib.dump(val_data, output_dir / "val_dataset.pkl")
    joblib.dump(test_data, output_dir / "test_dataset.pkl")
    joblib.dump(scalers, output_dir / "feature_scaler.pkl")

    print("\n" + "=" * 60)
    print("[SAVE] 预处理数据已保存")
    print(f"  train_dataset.pkl: {len(train_idx)} 条曲线")
    print(f"  val_dataset.pkl: {len(val_idx)} 条曲线")
    print(f"  test_dataset.pkl: {len(test_idx)} 条曲线")
    print("=" * 60)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "scalers": scalers,
    }


def load_preprocessed(output_dir: Path) -> Dict[str, Any]:
    """加载预处理后的数据"""
    train_data = joblib.load(output_dir / "train_dataset.pkl")
    val_data = joblib.load(output_dir / "val_dataset.pkl")
    test_data = joblib.load(output_dir / "test_dataset.pkl")
    scalers = joblib.load(output_dir / "feature_scaler.pkl")
    return {"train": train_data, "val": val_data, "test": test_data, "scalers": scalers}


def get_device(device_str: str) -> torch.device:
    """获取可用设备"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "cuda":
        print("WARNING 未检测到CUDA，已切换到CPU")
    return torch.device("cpu")


def build_material_weights(material_encoder: LabelEncoder) -> torch.Tensor:
    """构建材料权重向量"""
    weight_map = {"Al6061": 1.0, "Al2024": 50.0, "Al7075": 50.0}
    weights = [weight_map.get(name, 1.0) for name in material_encoder.classes_]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(
    model: nn.Module,
    data: Dict[str, Any],
    scalers: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    max_plots: int = 10,
    run_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """模型评估与可视化"""
    model.eval()
    strain_seq = torch.from_numpy(data["strain_seq"]).to(device)
    conditions = torch.from_numpy(data["conditions"]).to(device)

    with torch.no_grad():
        pred_scaled = model(strain_seq, conditions).cpu().numpy()

    stress_scaler = scalers["stress_scaler"]
    pred_raw = stress_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(
        pred_scaled.shape
    )
    true_raw = data["stress_raw"].reshape(pred_raw.shape)

    y_true = true_raw.reshape(-1)
    y_pred = pred_raw.reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    # 分材料指标
    per_material = {}
    for material in sorted(set(data["material_labels"])):
        indices = [i for i, m in enumerate(data["material_labels"]) if m == material]
        if not indices:
            continue
        true_m = true_raw[indices].reshape(-1)
        pred_m = pred_raw[indices].reshape(-1)
        mse_m = mean_squared_error(true_m, pred_m)
        rmse_m = float(np.sqrt(mse_m))
        r2_m = float(r2_score(true_m, pred_m))
        mape_m = float(np.mean(np.abs((true_m - pred_m) / (np.abs(true_m) + eps))) * 100)
        per_material[material] = {
            "r2": r2_m,
            "rmse": rmse_m,
            "mape": mape_m,
            "samples": len(indices),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 随机曲线对比
    rng = np.random.default_rng(DEFAULT_SEED)
    indices = rng.choice(len(data["strain_raw"]), size=min(max_plots, len(data["strain_raw"])), replace=False)
    cols = 2
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(12, 4 * rows))
    for i, idx in enumerate(indices, start=1):
        plt.subplot(rows, cols, i)
        plt.plot(data["strain_raw"][idx], true_raw[idx].squeeze(), label="True", linewidth=2)
        plt.plot(data["strain_raw"][idx], pred_raw[idx].squeeze(), label="Pred", linestyle="--")
        plt.title(f"{data['material_labels'][idx]} | {data['curve_ids'][idx]}")
        plt.xlabel("Strain")
        plt.ylabel("Stress (MPa)")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "predictions_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 保存JSON指标
    metrics: Dict[str, Any] = {
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
        "mse": float(mse),
        "samples": len(data["strain_seq"]),
    }
    if run_info:
        metrics.update(run_info)
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(metrics_dir / "per_material_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_material, f, ensure_ascii=False, indent=2)

    return {"metrics": metrics, "per_material": per_material}


def plot_training_curves(history: Dict[str, Any], output_dir: Path) -> None:
    """绘制训练曲线"""
    if not history:
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["lr"], label="Learning Rate", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM应力应变曲线预测一体化脚本")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/all_materials_merged.csv",
        help="合并CSV路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["preprocess", "train", "evaluate", "all"],
        help="运行模式",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--alpha", type=float, default=0.7, help="MSE损失权重")
    parser.add_argument("--beta", type=float, default=0.3, help="物理约束损失权重")
    parser.add_argument(
        "--disable_physics", action="store_true", help="禁用物理约束（向后兼容）"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    data_path = Path(args.data_path)
    output_dir = Path("data/processed")
    models_dir = Path("models")
    results_dir = Path("results")

    device = get_device(args.device)

    datasets = None
    history = None
    run_info: Dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "seq_len": SEQ_LEN,
        "alpha": args.alpha,
        "beta": args.beta,
        "disable_physics": args.disable_physics,
    }

    if args.mode in ["preprocess", "all"]:
        datasets = prepare_datasets(data_path, output_dir, args.seed)

    if args.mode in ["train", "all"]:
        if datasets is None:
            datasets = load_preprocessed(output_dir)

        train_dataset = StressStrainDataset(datasets["train"], datasets["scalers"])
        val_dataset = StressStrainDataset(datasets["val"], datasets["scalers"])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = StressStrainLSTM()
        material_weights = build_material_weights(datasets["scalers"]["material_encoder"])
        if args.disable_physics:
            alpha, beta = 1.0, 0.0
        else:
            alpha, beta = args.alpha, args.beta
        criterion = PhysicsAwareMSELoss(material_weights, alpha=alpha, beta=beta)
        trainer = Trainer(model, train_loader, val_loader, criterion, device)
        train_start = time.time()
        history = trainer.train(args.epochs, models_dir / "lstm_best.pth")
        run_info["training_time_sec"] = round(time.time() - train_start, 2)
        run_info["best_val_loss"] = trainer.best_val_loss
        plot_training_curves(history, results_dir)

    if args.mode in ["evaluate", "all"]:
        if datasets is None:
            datasets = load_preprocessed(output_dir)

        model = StressStrainLSTM()
        model_path = models_dir / "lstm_best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        evaluate_model(
            model,
            datasets["test"],
            datasets["scalers"],
            device,
            results_dir,
            run_info=run_info,
        )

    print("\nDONE 所有流程执行完成！")


if __name__ == "__main__":
    main()
