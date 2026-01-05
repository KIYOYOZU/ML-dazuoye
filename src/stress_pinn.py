#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physics-Informed Neural Network (PINN) 应力应变预测

物理约束:
1. 单调性约束: dσ/dε ≥ 0 (硬化阶段)
2. 弹性模量约束: dσ/dε|ε→0 ≈ E
3. 温度软化约束: dσ/dT ≤ 0
4. 应变率强化约束: dσ/d(log_ε̇) ≥ 0

特征（共21维）:
- strain, temperature, log_strain_rate, material_id
- E_GPa, c1, c2, c3, c4, c5 (Johnson-Cook参数)
- Si, Fe, Cu, Mn, Mg, Cr, Zn, Ti, Zr, V, Al (元素成分)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==================== 配置 ====================
CONFIG = {
    "merged_csv": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged.csv",
    "output_dir": r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\stress_pinn",
    "filter_test_type": "tensile",
    "split_ratio": {"train": 0.75, "val": 0.125, "test": 0.125},
    "split_seed": 42,
    "min_test_curves": 1,  # 每种材料至少保留的测试曲线数
    # 模型参数
    "hidden_dims": [256, 256, 128, 64],
    "dropout": 0.05,
    # 训练参数
    "epochs": 800,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    # 物理约束权重
    "lambda_data": 1.0,
    "lambda_monotonic": 0.01,
    "lambda_elastic": 0.001,
    "lambda_temp_soft": 0.01,
    "lambda_rate_hard": 0.01,
    # 材料权重（数据少的材料给更高权重）
    "material_weights": {
        "al2024": 1.0,
        "al2219": 1.2,
        "al6061": 3.0,  # 数据最少，给最高权重
        "al7075": 1.5,
    },
    # 边界条件约束
    "lambda_boundary": 0.0,  # 禁用边界约束，使用collocation points代替
    "boundary_strain_threshold": 0.002,
    # 零应变collocation points配置
    "use_collocation_points": True,  # 启用collocation points
    "n_collocation_per_material": 50,  # 每种材料生成的零应变点数
}

# 材料物理参数（与 stress_train.py 一致）
MATERIAL_FEATURES = {
    "al2024": {
        "material_id": 0,
        "E_GPa": 73.1,
        "c1": 325.0, "c2": 0.33, "c3": 22.78, "c4": 502.0, "c5": 2.29e-05,
        "Si_wt": 0.50, "Fe_wt": 0.50, "Cu_wt": 4.35, "Mn_wt": 0.60,
        "Mg_wt": 1.50, "Cr_wt": 0.10, "Zn_wt": 0.25, "Ti_wt": 2.00,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 90.20,
    },
    "al2219": {
        "material_id": 1,
        "E_GPa": 73.1,
        "c1": 350.0, "c2": 0.33, "c3": 22.84, "c4": 543.0, "c5": 2.23e-05,
        "Si_wt": 0.20, "Fe_wt": 0.30, "Cu_wt": 6.30, "Mn_wt": 0.30,
        "Mg_wt": 0.02, "Cr_wt": 0.00, "Zn_wt": 0.10, "Ti_wt": 0.06,
        "Zr_wt": 0.175, "V_wt": 0.10, "Al_wt": 92.445,
    },
    "al6061": {
        "material_id": 2,
        "E_GPa": 68.9,
        "c1": 276.0, "c2": 0.33, "c3": 22.70, "c4": 582.0, "c5": 2.36e-05,
        "Si_wt": 0.63, "Fe_wt": 0.2225, "Cu_wt": 0.195, "Mn_wt": 0.045,
        "Mg_wt": 0.8775, "Cr_wt": 0.0525, "Zn_wt": 0.02, "Ti_wt": 0.015,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 97.9425,
    },
    "al7075": {
        "material_id": 3,
        "E_GPa": 71.7,
        "c1": 503.0, "c2": 0.33, "c3": 22.81, "c4": 477.0, "c5": 2.32e-05,
        "Si_wt": 0.40, "Fe_wt": 0.50, "Cu_wt": 1.60, "Mn_wt": 0.30,
        "Mg_wt": 2.50, "Cr_wt": 0.23, "Zn_wt": 5.60, "Ti_wt": 0.20,
        "Zr_wt": 0.00, "V_wt": 0.00, "Al_wt": 88.67,
    },
}

COMPOSITION_COLS = ["Si_wt", "Fe_wt", "Cu_wt", "Mn_wt", "Mg_wt",
                   "Cr_wt", "Zn_wt", "Ti_wt", "Zr_wt", "V_wt", "Al_wt"]

# 特征列（共21维，包含material_id）
FEATURE_COLS = ["strain", "temperature", "log_strain_rate", "material_id",
                "E_GPa", "c1", "c2", "c3", "c4", "c5"] + COMPOSITION_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 数据处理 ====================
def load_and_prepare_data():
    """加载并准备数据"""
    df = pd.read_csv(CONFIG["merged_csv"])

    if CONFIG["filter_test_type"]:
        mask = df["test_type"].astype(str).str.lower() == CONFIG["filter_test_type"].lower()
        df = df[mask].copy()
        print(f"[PINN] 过滤 test_type={CONFIG['filter_test_type']}: {len(df)} 行")

    df["strain"] = df["strain"].clip(lower=0.0)
    df = df.dropna(subset=["strain", "stress_MPa", "temperature", "strain_rate", "material"])

    # 添加材料特征
    df = add_material_features(df)
    df["log_strain_rate"] = np.log10(df["strain_rate"].clip(lower=1e-12))

    return df


def add_material_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加材料物理参数特征"""
    df = df.copy()
    mat_rows = []
    for m in df["material"]:
        if m not in MATERIAL_FEATURES:
            raise ValueError(f"Unknown material: {m}")
        mat_rows.append(MATERIAL_FEATURES[m])
    mat_df = pd.DataFrame(mat_rows)
    return pd.concat([df.reset_index(drop=True), mat_df.reset_index(drop=True)], axis=1)


def split_by_curve(df: pd.DataFrame):
    """按曲线划分数据集，确保每种材料至少有 min_test_curves 条测试曲线"""
    curve_meta = df.groupby("source_file").first()[["material"]]
    rng = np.random.default_rng(CONFIG["split_seed"])
    train_files, val_files, test_files = [], [], []
    min_test = CONFIG.get("min_test_curves", 1)

    for material in sorted(curve_meta["material"].unique()):
        files = sorted(curve_meta[curve_meta["material"] == material].index.tolist())
        rng.shuffle(files)
        n = len(files)

        # 先确保至少有 min_test 条测试数据
        n_test = max(min_test, int(round(n * CONFIG["split_ratio"]["test"])))
        n_val = max(1, int(round(n * CONFIG["split_ratio"]["val"])))

        # 确保不会超过总数
        if n_test + n_val >= n:
            n_test = min(min_test, n - 1) if n > 1 else 0
            n_val = min(1, n - n_test - 1) if n - n_test > 1 else 0

        n_train = n - n_val - n_test

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

        print(f"  {material}: {n}条 -> 训练{n_train}, 验证{n_val}, 测试{n_test}")

    return train_files, val_files, test_files


def build_features(df: pd.DataFrame) -> np.ndarray:
    """构建特征矩阵（21维）"""
    return df[FEATURE_COLS].values.astype(np.float32)


def generate_collocation_points(train_df: pd.DataFrame, n_per_material: int = 50) -> pd.DataFrame:
    """生成零应变collocation points（应变=0时应力=0）

    为每种材料生成覆盖其温度和应变率范围的零应变点
    """
    collocation_data = []

    for material in train_df["material"].unique():
        mat_df = train_df[train_df["material"] == material]
        mat_feat = MATERIAL_FEATURES[material]

        # 获取该材料的温度和应变率范围
        temps = mat_df["temperature"].unique()
        rates = mat_df["strain_rate"].unique()

        # 生成均匀分布的温度和应变率组合
        n_temps = min(len(temps), int(np.sqrt(n_per_material)) + 1)
        n_rates = min(len(rates), n_per_material // n_temps + 1)

        selected_temps = np.linspace(temps.min(), temps.max(), n_temps)
        selected_rates = np.geomspace(max(rates.min(), 1e-6), rates.max(), n_rates)

        for temp in selected_temps:
            for rate in selected_rates:
                row = {
                    "source_file": f"collocation_{material}",
                    "material": material,
                    "temperature": temp,
                    "strain_rate": rate,
                    "strain": 0.0,  # 零应变
                    "stress_MPa": 0.0,  # 零应力（边界条件）
                    "log_strain_rate": np.log10(max(rate, 1e-12)),
                    **mat_feat,
                }
                collocation_data.append(row)

    collocation_df = pd.DataFrame(collocation_data)
    return collocation_df


def create_dataloaders(df, train_files, val_files, test_files, scaler_X, scaler_y):
    """创建数据加载器（带材料权重和collocation points）"""
    train_df = df[df["source_file"].isin(train_files)].copy()
    val_df = df[df["source_file"].isin(val_files)].copy()
    test_df = df[df["source_file"].isin(test_files)].copy()

    # 添加零应变collocation points
    if CONFIG.get("use_collocation_points", False):
        n_per_mat = CONFIG.get("n_collocation_per_material", 50)
        collocation_df = generate_collocation_points(train_df, n_per_mat)
        print(f"  生成collocation points: {len(collocation_df)} 个零应变点")
        # 合并到训练集
        train_df = pd.concat([train_df, collocation_df], ignore_index=True)

    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)

    y_train = train_df["stress_MPa"].values.astype(np.float32).reshape(-1, 1)
    y_val = val_df["stress_MPa"].values.astype(np.float32).reshape(-1, 1)
    y_test = test_df["stress_MPa"].values.astype(np.float32).reshape(-1, 1)

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    # 弹性模量（用于物理约束）
    E_train = train_df["E_GPa"].values.astype(np.float32).reshape(-1, 1) * 1000  # GPa -> MPa
    E_val = val_df["E_GPa"].values.astype(np.float32).reshape(-1, 1) * 1000

    # 计算材料权重（collocation points给更高权重）
    material_weights = CONFIG.get("material_weights", {})
    collocation_weight = 2.0  # collocation points的权重倍数

    weights_train = []
    for idx, row in train_df.iterrows():
        base_weight = material_weights.get(row["material"], 1.0)
        if str(row.get("source_file", "")).startswith("collocation_"):
            weights_train.append(base_weight * collocation_weight)
        else:
            weights_train.append(base_weight)
    weights_train = np.array(weights_train, dtype=np.float32)

    # 显示权重统计
    print("  材料权重统计:")
    for mat in sorted(train_df["material"].unique()):
        mat_mask = train_df["material"] == mat
        colloc_mask = train_df["source_file"].str.startswith("collocation_")
        n_data = (~colloc_mask & mat_mask).sum()
        n_colloc = (colloc_mask & mat_mask).sum()
        w = material_weights.get(mat, 1.0)
        print(f"    {mat}: {n_data}数据 + {n_colloc}边界点, 权重={w}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train_s),
        torch.from_numpy(X_train),
        torch.from_numpy(E_train),
        torch.from_numpy(weights_train.reshape(-1, 1)),  # 添加权重
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_s),
        torch.from_numpy(y_val_s),
        torch.from_numpy(X_val),
        torch.from_numpy(E_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # 测试集原始特征和E值（用于硬约束模式评估）
    X_test = build_features(test_df)
    E_test = test_df["E_GPa"].values.astype(np.float32).reshape(-1, 1) * 1000

    return train_loader, val_loader, (X_test_s, y_test, test_df, X_test, E_test)


# ==================== PINN 模型 ====================
class StressPINN(nn.Module):
    """软约束PINN模型（标准版本）"""
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, x_raw=None, E_MPa=None):
        # 忽略x_raw和E_MPa，保持兼容性
        return self.network(x)


# ==================== 物理约束损失 ====================
class PhysicsLoss:
    def __init__(self, scaler_X: StandardScaler, scaler_y: StandardScaler):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.X_mean = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=DEVICE)
        self.X_std = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=DEVICE)
        self.y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=DEVICE)
        self.y_std = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=DEVICE)

    def compute_gradients(self, model, X_scaled, create_graph=True):
        X_scaled = X_scaled.requires_grad_(True)
        y_pred = model(X_scaled)

        grads = torch.autograd.grad(
            outputs=y_pred, inputs=X_scaled,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=create_graph, retain_graph=True,
        )[0]

        real_grads = grads * (self.y_std / self.X_std)
        # 特征顺序: strain(0), temperature(1), log_strain_rate(2), ...
        dsigma_depsilon = real_grads[:, 0:1]
        dsigma_dT = real_grads[:, 1:2]
        dsigma_dlograte = real_grads[:, 2:3]

        return y_pred, dsigma_depsilon, dsigma_dT, dsigma_dlograte

    def monotonicity_loss(self, dsigma_depsilon):
        violation = torch.relu(-dsigma_depsilon)
        return torch.mean(violation ** 2)

    def elastic_modulus_loss(self, dsigma_depsilon, X_raw, E_MPa):
        strain = X_raw[:, 0:1]
        mask = (strain < 0.02).float()
        if mask.sum() < 1:
            return torch.tensor(0.0, device=DEVICE)
        relative_error = (dsigma_depsilon - E_MPa) / (E_MPa + 1e-6)
        weighted_error = mask * (relative_error ** 2)
        return torch.sum(weighted_error) / (mask.sum() + 1e-6)

    def temperature_softening_loss(self, dsigma_dT):
        violation = torch.relu(dsigma_dT)
        return torch.mean(violation ** 2)

    def strain_rate_hardening_loss(self, dsigma_dlograte):
        violation = torch.relu(-dsigma_dlograte)
        return torch.mean(violation ** 2)

    def boundary_condition_loss(self, y_pred_scaled, X_raw):
        """边界条件损失: σ(ε≈0) = 0

        在应变接近0时，应力也应该接近0
        """
        strain = X_raw[:, 0:1]
        threshold = CONFIG.get("boundary_strain_threshold", 0.005)
        mask = (strain < threshold).float()

        if mask.sum() < 1:
            return torch.tensor(0.0, device=DEVICE)

        # 将预测值转换回真实尺度
        y_pred_real = y_pred_scaled * self.y_std + self.y_mean

        # 惩罚非零应力，使用相对误差避免量纲问题
        # 在低应变区域，应力应该接近0
        boundary_error = mask * (y_pred_real ** 2)
        return torch.sum(boundary_error) / (mask.sum() + 1e-6)


# ==================== 训练 ====================
def train_epoch(model, train_loader, optimizer, criterion, physics_loss, scaler_y):
    """训练一个epoch（软约束模式）"""
    model.train()
    total_loss = 0
    loss_components = {"data": 0, "mono": 0, "elastic": 0, "temp": 0, "rate": 0, "boundary": 0}

    for X_scaled, y_scaled, X_raw, E_MPa, weights in train_loader:
        X_scaled = X_scaled.to(DEVICE)
        y_scaled = y_scaled.to(DEVICE)
        X_raw = X_raw.to(DEVICE)
        E_MPa = E_MPa.to(DEVICE)
        weights = weights.to(DEVICE)

        optimizer.zero_grad()
        y_pred, dsigma_de, dsigma_dT, dsigma_dlograte = physics_loss.compute_gradients(model, X_scaled)

        # 加权MSE损失
        loss_data = torch.mean(weights * (y_pred - y_scaled) ** 2)
        loss_mono = physics_loss.monotonicity_loss(dsigma_de)
        loss_elastic = physics_loss.elastic_modulus_loss(dsigma_de, X_raw, E_MPa)
        loss_temp = physics_loss.temperature_softening_loss(dsigma_dT)
        loss_rate = physics_loss.strain_rate_hardening_loss(dsigma_dlograte)
        loss_boundary = physics_loss.boundary_condition_loss(y_pred, X_raw)

        loss = (CONFIG["lambda_data"] * loss_data +
                CONFIG["lambda_monotonic"] * loss_mono +
                CONFIG["lambda_elastic"] * loss_elastic +
                CONFIG["lambda_temp_soft"] * loss_temp +
                CONFIG["lambda_rate_hard"] * loss_rate +
                CONFIG["lambda_boundary"] * loss_boundary)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loss_components["data"] += loss_data.item()
        loss_components["mono"] += loss_mono.item()
        loss_components["elastic"] += loss_elastic.item()
        loss_components["temp"] += loss_temp.item()
        loss_components["rate"] += loss_rate.item()
        loss_components["boundary"] += loss_boundary.item()

    n_batches = len(train_loader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}


def validate(model, val_loader, criterion, physics_loss):
    """验证（软约束模式）"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_scaled, y_scaled, X_raw, E_MPa in val_loader:
            X_scaled = X_scaled.to(DEVICE)
            y_scaled = y_scaled.to(DEVICE)
            y_pred = model(X_scaled)
            loss = criterion(y_pred, y_scaled)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def evaluate(model, X_test_s, y_test, scaler_y, X_test_raw=None, E_test=None):
    """评估（软约束模式）"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test_s.astype(np.float32)).to(DEVICE)
        y_pred_s = model(X_tensor).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_s).flatten()
    y_true = y_test.flatten()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("PINN 应力应变预测训练 (软约束模式, 21维特征, 启用边界约束)")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] 加载数据...")
    df = load_and_prepare_data()
    train_files, val_files, test_files = split_by_curve(df)
    print(f"  训练曲线: {len(train_files)}, 验证曲线: {len(val_files)}, 测试曲线: {len(test_files)}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_loader, val_loader, (X_test_s, y_test, test_df, X_test_raw, E_test) = create_dataloaders(
        df, train_files, val_files, test_files, scaler_X, scaler_y
    )
    print(f"  训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    print(f"  特征维度: {len(FEATURE_COLS)}")

    print("\n[2] 创建模型...")
    input_dim = len(FEATURE_COLS)
    model = StressPINN(
        input_dim=input_dim,
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    physics_loss = PhysicsLoss(scaler_X, scaler_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    print("\n[3] 开始训练...")
    print(f"  物理约束权重: mono={CONFIG['lambda_monotonic']}, elastic={CONFIG['lambda_elastic']}, "
          f"temp={CONFIG['lambda_temp_soft']}, rate={CONFIG['lambda_rate_hard']}, boundary={CONFIG['lambda_boundary']}")

    best_val_loss = float("inf")
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "components": []}

    for epoch in range(CONFIG["epochs"]):
        train_loss, loss_comp = train_epoch(model, train_loader, optimizer, criterion, physics_loss, scaler_y)
        val_loss = validate(model, val_loader, criterion, physics_loss)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["components"].append(loss_comp)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"data={loss_comp['data']:.4f}, mono={loss_comp['mono']:.4f}, boundary={loss_comp['boundary']:.4f}")

    model.load_state_dict(best_model_state)

    print("\n[4] 评估测试集...")
    test_metrics = evaluate(model, X_test_s, y_test, scaler_y)
    print(f"  R2: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f} MPa")
    print(f"  MAE: {test_metrics['mae']:.2f} MPa")

    per_material = {}
    for mat in test_df["material"].unique():
        mask = test_df["material"] == mat
        idx = mask.values
        per_material[mat] = evaluate(model, X_test_s[idx], y_test[idx], scaler_y)

    print("\n  各材料性能:")
    for mat, m in per_material.items():
        print(f"    {mat}: R2={m['r2']:.4f}, RMSE={m['rmse']:.2f}")

    print("\n[5] 保存结果...")
    torch.save({
        "model_state": best_model_state,
        "config": CONFIG,
        "feature_cols": FEATURE_COLS,
        "scaler_X_mean": scaler_X.mean_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_mean": scaler_y.mean_,
        "scaler_y_scale": scaler_y.scale_,
    }, output_dir / "pinn_model.pt")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "per_material": per_material,
                   "config": {k: v for k, v in CONFIG.items() if not k.endswith("_dir") and not k.endswith("_csv")}}, f, indent=2)

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)

    print("  生成全量预测...")
    X_all = build_features(df).astype(np.float32)
    X_all_s = scaler_X.transform(X_all)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_all_s).to(DEVICE)
        y_pred_s = model(X_tensor).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_s).flatten()

    df_out = df.copy()
    df_out["stress_pred"] = y_pred
    df_out.to_csv(output_dir / "all_predictions.csv", index=False)

    print(f"\n输出目录: {output_dir}")
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
