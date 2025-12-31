#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的训练程序 - 铝合金应力应变曲线预测
包含: 数据加载、模型定义、物理约束损失、训练循环、评估、可视化
"""

import os
import json
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 配置 ====================
CONFIG = {
    'data_dir': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed",
    'merged_csv': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed\筛选数据_merged.csv",
    'output_dir': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业",
    'rebuild_dataset': True,
    'split_ratio': {'train': 0.75, 'val': 0.125, 'test': 0.125},
    'split_seed': 42,
    'seq_len': 100,
    'batch_size': 8,  # 减小batch size
    'hidden_size': 128,  # 减小隐藏层
    'num_layers': 2,  # 减少层数
    'dropout': 0.2,  # 减小dropout
    'epochs': 300,
    'lr': 0.002,  # 提高学习率
    'alpha': 1.00,  # MSE权重
    'beta': 0.20,   # 物理约束权重（启用）
    'early_stop_patience': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

# 材料基础参数（来自既有训练数据的特征分布）
MATERIAL_FEATURES = {
    'al2024': {'material_id': 0, 'E_GPa': 73.1, 'c1': 325.0, 'c2': 0.33, 'c3': 2.78, 'c4': 502.0, 'c5': 2.29e-05},
    'al2219': {'material_id': 1, 'E_GPa': 73.1, 'c1': 350.0, 'c2': 0.33, 'c3': 2.84, 'c4': 543.0, 'c5': 2.23e-05},
    'al6061': {'material_id': 2, 'E_GPa': 68.9, 'c1': 276.0, 'c2': 0.33, 'c3': 2.70, 'c4': 582.0, 'c5': 2.36e-05},
    'al7075': {'material_id': 3, 'E_GPa': 71.7, 'c1': 503.0, 'c2': 0.33, 'c3': 2.81, 'c4': 477.0, 'c5': 2.32e-05},
}


def rebuild_dataset_from_merged(config):
    """基于真实应力应变的合并数据重建 train/val/test.pkl"""
    data_dir = Path(config['data_dir'])
    merged_csv = Path(config['merged_csv'])

    if not merged_csv.exists():
        raise FileNotFoundError(f"未找到合并数据: {merged_csv}")

    df = pd.read_csv(merged_csv)
    required_cols = ['source_file', 'material', 'temperature', 'strain_rate', 'strain', 'stress_MPa']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"合并数据缺少列: {missing_cols}")

    # 曲线级元信息（避免被数据点数量加权）
    curve_meta = df.groupby('source_file').first()[['material', 'temperature', 'strain_rate']]
    temps = curve_meta['temperature'].astype(float).to_numpy()
    rates = curve_meta['strain_rate'].astype(float).to_numpy()

    if np.any(rates <= 0):
        raise ValueError("存在非正应变率，无法进行 log10 标准化")

    rate_logs = np.log10(rates)
    temp_mean = temps.mean()
    temp_std = temps.std(ddof=0)
    rate_mean = rate_logs.mean()
    rate_std = rate_logs.std(ddof=0)

    if temp_std <= 0 or rate_std <= 0:
        raise ValueError("温度或应变率标准差为 0，无法标准化")

    rng = np.random.default_rng(config['split_seed'])
    train_files, val_files, test_files = [], [], []

    for material in sorted(curve_meta['material'].unique()):
        files = sorted(curve_meta[curve_meta['material'] == material].index.tolist())
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * config['split_ratio']['train']))
        n_val = int(round(n * config['split_ratio']['val']))
        if n_train + n_val > n:
            n_train = max(0, n - n_val)
        n_test = n - n_train - n_val

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    groups = {name: g for name, g in df.groupby('source_file')}

    def build_sample(source_file):
        if source_file not in groups:
            return None
        g = groups[source_file].sort_values('strain')

        material = curve_meta.loc[source_file, 'material']
        if material not in MATERIAL_FEATURES:
            return None

        temperature = float(curve_meta.loc[source_file, 'temperature'])
        strain_rate = float(curve_meta.loc[source_file, 'strain_rate'])

        # 合理化应变（去负值并去重）
        temp_df = g[['strain', 'stress_MPa']].copy()
        temp_df['strain'] = temp_df['strain'].clip(lower=0.0)

        if temp_df['strain'].duplicated().any():
            temp_df = temp_df.groupby('strain', as_index=False)['stress_MPa'].mean()

        temp_df = temp_df.sort_values('strain').reset_index(drop=True)

        # 零点锚定：若起始应变>0则补(0,0)，若起始应变≈0但应力偏移则平移归零
        min_idx = temp_df['strain'].idxmin()
        min_strain = float(temp_df.loc[min_idx, 'strain'])
        min_stress = float(temp_df.loc[min_idx, 'stress_MPa'])
        if min_strain <= 1e-6:
            if abs(min_stress) > 1e-3:
                temp_df['stress_MPa'] = temp_df['stress_MPa'] - min_stress
        else:
            temp_df = pd.concat(
                [pd.DataFrame({'strain': [0.0], 'stress_MPa': [0.0]}), temp_df],
                ignore_index=True,
            )
            temp_df = temp_df.sort_values('strain').reset_index(drop=True)

        strain = temp_df['strain'].to_numpy(dtype=float)
        stress = temp_df['stress_MPa'].to_numpy(dtype=float)

        if len(strain) < 2 or np.allclose(strain.min(), strain.max()):
            return None

        order = np.argsort(strain)
        strain = strain[order]
        stress = stress[order]

        mask = ~np.isnan(strain) & ~np.isnan(stress)
        strain = strain[mask]
        stress = stress[mask]

        if len(strain) < 2:
            return None

        strain_seq = np.linspace(float(strain.min()), float(strain.max()), config['seq_len'])
        stress_seq = np.interp(strain_seq, strain, stress)
        # 物理合理化：张拉应力不应为负
        stress_seq = np.clip(stress_seq, 0.0, None)

        props = MATERIAL_FEATURES[material]
        temp_z = (temperature - temp_mean) / temp_std
        rate_z = (math.log10(strain_rate) - rate_mean) / rate_std

        conditions = np.array([
            props['E_GPa'], props['c1'], props['c2'], props['c3'], props['c4'], props['c5'],
            float(props['material_id']), temp_z, rate_z
        ], dtype=float)

        return {
            'strain_seq': strain_seq.astype(float),
            'stress_seq': stress_seq.astype(float),
            'conditions': conditions,
            'material': material,
            'material_id': np.int64(props['material_id']),
            'temperature': int(round(temperature)),
            'strain_rate': strain_rate,
            'E_GPa': float(props['E_GPa']),
            'source_file': source_file,
        }

    def build_split(file_list):
        samples, missing = [], []
        for f in file_list:
            sample = build_sample(f)
            if sample is None:
                missing.append(f)
            else:
                samples.append(sample)
        return samples, missing

    new_train, miss_train = build_split(train_files)
    new_val, miss_val = build_split(val_files)
    new_test, miss_test = build_split(test_files)

    with open(data_dir / "train.pkl", 'wb') as f:
        pickle.dump(new_train, f)
    with open(data_dir / "val.pkl", 'wb') as f:
        pickle.dump(new_val, f)
    with open(data_dir / "test.pkl", 'wb') as f:
        pickle.dump(new_test, f)

    print("\n[数据集重建] 完成")
    print(f"  训练集: {len(new_train)}")
    print(f"  验证集: {len(new_val)}")
    print(f"  测试集: {len(new_test)}")
    if miss_train or miss_val or miss_test:
        print("  ⚠️  以下文件生成样本失败:")
        print(f"    train: {miss_train}")
        print(f"    val: {miss_val}")
        print(f"    test: {miss_test}")

# ==================== 数据集 ====================
class StressStrainDataset(Dataset):
    def __init__(self, data_path, stress_scaler=None, fit_scaler=False,
                 condition_scaler=None, fit_condition_scaler=False):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        if fit_scaler:
            # 仅在训练集拟合归一化器，其他集复用同一scaler
            all_stress = []
            for sample in self.data:
                all_stress.append(sample['stress_seq'])

            all_stress = np.concatenate(all_stress).reshape(-1, 1)
            self.stress_scaler = MinMaxScaler()
            self.stress_scaler.fit(all_stress)

            print(f"  应力范围: {all_stress.min():.2f} ~ {all_stress.max():.2f} MPa")
        else:
            if stress_scaler is None:
                raise ValueError("stress_scaler 为空且未允许拟合，请传入训练集的scaler")
            self.stress_scaler = stress_scaler

        if fit_condition_scaler:
            all_conditions = np.stack([sample['conditions'] for sample in self.data])
            self.condition_scaler = StandardScaler()
            self.condition_scaler.fit(all_conditions)
        else:
            if condition_scaler is None:
                raise ValueError("condition_scaler 为空且未允许拟合，请传入训练集的scaler")
            self.condition_scaler = condition_scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        strain_seq = torch.FloatTensor(sample['strain_seq']).unsqueeze(-1)

        stress_seq = sample['stress_seq'].reshape(-1, 1)
        stress_seq_norm = self.stress_scaler.transform(stress_seq)
        stress_seq = torch.FloatTensor(stress_seq_norm)

        cond_np = np.array(sample['conditions']).reshape(1, -1)
        cond_norm = self.condition_scaler.transform(cond_np).squeeze(0)
        conditions = torch.FloatTensor(cond_norm)

        return {
            'strain_seq': strain_seq,
            'stress_seq': stress_seq,
            'conditions': conditions,
            'material_id': sample['material_id'],
            'material': sample['material'],
            'temperature': sample['temperature'],
            'strain_rate': sample['strain_rate'],
            'E_GPa': sample['E_GPa'],
            'stress_scaler': self.stress_scaler,
            'strain_seq_raw': sample['strain_seq'],
        }

def collate_fn(batch):
    return {
        'strain_seq': torch.stack([item['strain_seq'] for item in batch]),
        'stress_seq': torch.stack([item['stress_seq'] for item in batch]),
        'conditions': torch.stack([item['conditions'] for item in batch]),
        'material_ids': torch.LongTensor([item['material_id'] for item in batch]),
        'materials': [item['material'] for item in batch],
        'temperatures': torch.FloatTensor([item['temperature'] for item in batch]),
        'strain_rates': torch.FloatTensor([item['strain_rate'] for item in batch]),
        'E_GPas': torch.FloatTensor([item['E_GPa'] for item in batch]),
        'stress_scalers': [batch[0]['stress_scaler']] * len(batch),  # 所有样本共享同一个scaler
        'strain_seqs_raw': np.array([item['strain_seq_raw'] for item in batch]),
    }

# ==================== 模型 ====================
class ImprovedStressStrainLSTM(nn.Module):
    def __init__(self, seq_len=100, hidden_size=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len

        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

        # 注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # 融合预测
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 256 + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.strain_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, strain_seq, conditions):
        cond_embed = self.condition_embed(conditions)
        lstm_out, _ = self.lstm(strain_seq)
        lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        seq_len = strain_seq.size(1)
        cond_expand = cond_embed.unsqueeze(1).expand(-1, seq_len, -1)
        fused = torch.cat([lstm_out, cond_expand, strain_seq], dim=-1)
        stress_pred = self.fusion(fused) + self.strain_head(strain_seq)
        # 零点锚定：强制首点应力为 0（假设序列已从最小应变开始）
        stress_pred = stress_pred - stress_pred[:, :1, :]
        return stress_pred

# ==================== 物理约束损失 ====================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha=0.70, beta=0.30, low_strain_weight=2.0, low_strain_scale=0.02):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.low_strain_weight = low_strain_weight
        self.low_strain_scale = low_strain_scale

        # 材料权重
        counts = torch.tensor([24, 20, 11, 16], dtype=torch.float32)
        self.material_weights = counts.sum() / (counts * 4)

    def forward(self, pred, target, batch_data):
        # MSE损失（低应变区域加权）
        mse = (pred - target) ** 2
        strain_raw = torch.FloatTensor(batch_data['strain_seqs_raw']).to(pred.device)
        point_weight = 1.0 + self.low_strain_weight * torch.exp(
            -strain_raw / max(self.low_strain_scale, 1e-6)
        )
        mse = mse * point_weight.unsqueeze(-1)
        weights = self.material_weights[batch_data['material_ids']].to(pred.device)
        weights = weights.view(-1, 1, 1)
        mse_loss = (mse * weights).mean()

        # 物理约束
        if self.beta > 0:
            pred_raw = self._denormalize(pred, batch_data['stress_scalers'])

            physics_loss = (
                self._elastic_modulus_constraint(strain_raw, pred_raw, batch_data['E_GPas']) * 0.22 +
                self._monotonicity_constraint(pred_raw) * 0.18 +
                self._stress_range_constraint(pred_raw, batch_data['material_ids']) * 0.18 +
                self._temperature_softening_constraint(pred_raw, batch_data) * 0.18 +
                self._strain_rate_sensitivity_constraint(pred_raw, batch_data) * 0.14 +
                self._zero_stress_constraint(strain_raw, pred_raw, batch_data['material_ids']) * 0.10
            )
        else:
            physics_loss = torch.tensor(0.0, device=pred.device)

        total_loss = self.alpha * mse_loss + self.beta * physics_loss

        return total_loss, {
            'mse': mse_loss.item(),
            'physics': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            'total': total_loss.item(),
        }

    def _denormalize(self, stress_norm, scalers):
        batch_size = stress_norm.size(0)
        stress_raw = torch.zeros_like(stress_norm)
        for i in range(batch_size):
            stress_np = stress_norm[i].detach().cpu().numpy()
            stress_denorm = scalers[i].inverse_transform(stress_np)
            stress_raw[i] = torch.FloatTensor(stress_denorm).to(stress_norm.device)
        return stress_raw

    def _elastic_modulus_constraint(self, strain, stress, E_GPas):
        E_true = E_GPas * 1000
        losses = []
        for i in range(strain.size(0)):
            mask = strain[i] < 0.02
            if mask.sum() < 3:
                continue
            eps = strain[i][mask]
            sig = stress[i][mask].squeeze()
            E_pred = (eps * sig).sum() / (eps ** 2).sum().clamp(min=1e-8)
            rel_error = torch.abs(E_pred - E_true[i]) / E_true[i].clamp(min=1e-8)
            losses.append(rel_error)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=strain.device)

    def _monotonicity_constraint(self, stress):
        delta_stress = stress[:, 1:, 0] - stress[:, :-1, 0]
        violations = torch.relu(-delta_stress)
        return violations.mean()

    def _stress_range_constraint(self, stress, material_ids):
        UTS = torch.tensor([470, 400, 310, 570], device=stress.device)
        max_stress = stress.max(dim=1)[0].squeeze()
        uts_limits = UTS[material_ids] * 1.2
        violations = torch.relu(max_stress - uts_limits)
        return violations.mean()

    def _zero_stress_constraint(self, strain, stress, material_ids):
        # 约束应变≈0 时应力接近 0（相对材料UTS归一化）
        if strain.ndim == 3:
            strain_vals = strain[:, :, 0]
        else:
            strain_vals = strain
        if stress.ndim == 3:
            stress_vals = stress[:, :, 0]
        else:
            stress_vals = stress

        idx = torch.argmin(strain_vals, dim=1)
        batch_idx = torch.arange(strain_vals.size(0), device=stress.device)
        stress_at_zero = stress_vals[batch_idx, idx]
        uts = torch.tensor([470, 400, 310, 570], device=stress.device)[material_ids]
        rel = stress_at_zero / uts.clamp(min=1e-6)
        return (rel ** 2).mean()

    def _temperature_softening_constraint(self, stress, batch_data):
        material_ids = batch_data['material_ids']
        strain_rates = batch_data['strain_rates']
        temperatures = batch_data['temperatures']

        violations = []
        for i in range(len(material_ids)):
            for j in range(i+1, len(material_ids)):
                if (material_ids[i] == material_ids[j] and
                    torch.abs(strain_rates[i] - strain_rates[j]) < 0.01):
                    if temperatures[i] > temperatures[j]:
                        violation = torch.relu(stress[i].mean() - stress[j].mean())
                        violations.append(violation)
                    elif temperatures[i] < temperatures[j]:
                        violation = torch.relu(stress[j].mean() - stress[i].mean())
                        violations.append(violation)
        return torch.stack(violations).mean() if violations else torch.tensor(0.0, device=stress.device)

    def _strain_rate_sensitivity_constraint(self, stress, batch_data):
        material_ids = batch_data['material_ids']
        strain_rates = batch_data['strain_rates']
        temperatures = batch_data['temperatures']

        violations = []
        for i in range(len(material_ids)):
            for j in range(i+1, len(material_ids)):
                if (material_ids[i] == material_ids[j] and
                    torch.abs(temperatures[i] - temperatures[j]) < 5.0):
                    if strain_rates[i] > strain_rates[j]:
                        violation = torch.relu(stress[j].mean() - stress[i].mean())
                        violations.append(violation)
                    elif strain_rates[i] < strain_rates[j]:
                        violation = torch.relu(stress[i].mean() - stress[j].mean())
                        violations.append(violation)
        return torch.stack(violations).mean() if violations else torch.tensor(0.0, device=stress.device)

# ==================== 训练函数 ====================
class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'mse': 0, 'physics': 0}

    for batch in loader:
        strain_seq = batch['strain_seq'].to(device)
        stress_seq = batch['stress_seq'].to(device)
        conditions = batch['conditions'].to(device)

        pred = model(strain_seq, conditions)
        loss, loss_dict = criterion(pred, stress_seq, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        metrics['mse'] += loss_dict['mse']
        metrics['physics'] += loss_dict['physics']

    n = len(loader)
    return {'loss': total_loss/n, 'mse': metrics['mse']/n, 'physics': metrics['physics']/n}

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'mse': 0, 'physics': 0}

    with torch.no_grad():
        for batch in loader:
            strain_seq = batch['strain_seq'].to(device)
            stress_seq = batch['stress_seq'].to(device)
            conditions = batch['conditions'].to(device)

            pred = model(strain_seq, conditions)
            loss, loss_dict = criterion(pred, stress_seq, batch)

            total_loss += loss.item()
            metrics['mse'] += loss_dict['mse']
            metrics['physics'] += loss_dict['physics']

    n = len(loader)
    return {'loss': total_loss/n, 'mse': metrics['mse']/n, 'physics': metrics['physics']/n}

def evaluate_model(model, loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    material_results = {'al2024': {'pred': [], 'true': []},
                       'al2219': {'pred': [], 'true': []},
                       'al6061': {'pred': [], 'true': []},
                       'al7075': {'pred': [], 'true': []}}

    with torch.no_grad():
        for batch in loader:
            strain_seq = batch['strain_seq'].to(device)
            stress_seq = batch['stress_seq'].to(device)
            conditions = batch['conditions'].to(device)

            pred = model(strain_seq, conditions)

            # 反归一化
            for i in range(len(batch['materials'])):
                pred_np = pred[i].cpu().numpy()
                target_np = stress_seq[i].cpu().numpy()

                scaler = batch['stress_scalers'][i]
                pred_denorm = scaler.inverse_transform(pred_np)
                target_denorm = scaler.inverse_transform(target_np)

                all_preds.append(pred_denorm.flatten())
                all_targets.append(target_denorm.flatten())

                mat = batch['materials'][i]
                material_results[mat]['pred'].append(pred_denorm.flatten())
                material_results[mat]['true'].append(target_denorm.flatten())

    # 整体指标
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    overall_metrics = {
        'r2': r2_score(all_targets, all_preds),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'mae': np.mean(np.abs(all_targets - all_preds)),
        'samples': len(loader.dataset),
    }

    # 每种材料指标
    per_material_metrics = {}
    for mat, data in material_results.items():
        if len(data['pred']) > 0:
            pred = np.concatenate(data['pred'])
            true = np.concatenate(data['true'])
            per_material_metrics[mat] = {
                'r2': r2_score(true, pred),
                'rmse': np.sqrt(mean_squared_error(true, pred)),
                'samples': len(data['pred']),
            }

    return overall_metrics, per_material_metrics

def plot_results(history, test_metrics, per_material_metrics, output_dir):
    """绘制结果图表"""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_mse'], label='Train')
    axes[0, 1].plot(history['val_mse'], label='Val')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_physics'], label='Train')
    axes[1, 0].plot(history['val_physics'], label='Val')
    axes[1, 0].set_title('Physics Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  训练曲线已保存: {fig_dir / 'training_curves.png'}")

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("铝合金应力应变曲线预测 - 完整训练程序")
    print("="*60)

    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    device = torch.device(CONFIG['device'])
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    data_dir = Path(CONFIG['data_dir'])

    if CONFIG.get('rebuild_dataset'):
        rebuild_dataset_from_merged(CONFIG)

    train_dataset = StressStrainDataset(
        str(data_dir / "train.pkl"),
        fit_scaler=True,
        fit_condition_scaler=True,
    )
    shared_stress_scaler = train_dataset.stress_scaler
    shared_condition_scaler = train_dataset.condition_scaler
    val_dataset = StressStrainDataset(
        str(data_dir / "val.pkl"),
        stress_scaler=shared_stress_scaler,
        condition_scaler=shared_condition_scaler,
    )
    test_dataset = StressStrainDataset(
        str(data_dir / "test.pkl"),
        stress_scaler=shared_stress_scaler,
        condition_scaler=shared_condition_scaler,
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn)

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 创建模型
    print("\n创建模型...")
    model = ImprovedStressStrainLSTM(
        seq_len=CONFIG['seq_len'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = PhysicsInformedLoss(alpha=CONFIG['alpha'], beta=CONFIG['beta'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=CONFIG['early_stop_patience'])

    # 训练历史
    history = {
        'train_loss': [], 'train_mse': [], 'train_physics': [],
        'val_loss': [], 'val_mse': [], 'val_physics': [], 'lr': []
    }

    # 训练循环
    print("\n开始训练...")
    print("="*60)

    for epoch in range(CONFIG['epochs']):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['train_mse'].append(train_metrics['mse'])
        history['train_physics'].append(train_metrics['physics'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_physics'].append(val_metrics['physics'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, Physics: {train_metrics['physics']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, Physics: {val_metrics['physics']:.4f}")

        if early_stopping(val_metrics['loss'], model):
            print(f"\n早停于epoch {epoch+1}")
            break

    # 恢复最佳模型
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
        print(f"\n恢复最佳模型 (val_loss={early_stopping.best_loss:.4f})")

    # 测试集评估
    print("\n测试集评估...")
    test_metrics, per_material_metrics = evaluate_model(model, test_loader, device)

    print(f"\n整体性能:")
    print(f"  R2: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f} MPa")
    print(f"  MAE: {test_metrics['mae']:.2f} MPa")

    print(f"\n各材料性能:")
    for mat, metrics in per_material_metrics.items():
        print(f"  {mat}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f} MPa, samples={metrics['samples']}")

    # 保存结果
    output_dir = Path(CONFIG['output_dir'])

    # 保存模型
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'history': history,
    }, models_dir / "lstm_physics_best.pth")
    print(f"\n模型已保存: {models_dir / 'lstm_physics_best.pth'}")

    # 保存指标（转换numpy类型为Python类型）
    metrics_dir = output_dir / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 转换为Python原生类型
    test_metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in test_metrics.items()}
    per_material_metrics_json = {
        mat: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
              for k, v in metrics.items()}
        for mat, metrics in per_material_metrics.items()
    }

    with open(metrics_dir / "test_metrics.json", 'w') as f:
        json.dump(test_metrics_json, f, indent=2)

    with open(metrics_dir / "per_material_metrics.json", 'w') as f:
        json.dump(per_material_metrics_json, f, indent=2)

    with open(output_dir / "results" / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # 绘制结果
    plot_results(history, test_metrics, per_material_metrics, output_dir)

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    return model, history, test_metrics

if __name__ == "__main__":
    model, history, test_metrics = main()
