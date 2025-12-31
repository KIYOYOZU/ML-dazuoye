#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本 - 生成原始数据vs预测数据的多温度对比图
"""

import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import Normalize
import shutil
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 导入模型定义
import sys
sys.path.append(str(Path(__file__).parent))
from train_complete import ImprovedStressStrainLSTM, StressStrainDataset

# 配置
CONFIG = {
    'model_path': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\models\lstm_physics_best.pth",
    'data_dir': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\processed",
    'output_dir': r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\results\figures",
    'clear_output_dir': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = ImprovedStressStrainLSTM(
        seq_len=config['seq_len'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def prepare_output_dir(output_dir, clear_output_dir=False):
    """准备输出目录，必要时清空旧文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_output_dir:
        removed = 0
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
                removed += 1
        print(f"  已清空输出目录: {output_dir} (删除 {removed} 项)")
    return output_dir

def predict_samples(model, dataset, device):
    """对数据集中的所有样本进行预测"""
    results = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            # 准备输入
            strain_seq = sample['strain_seq'].unsqueeze(0).to(device)
            conditions = sample['conditions'].unsqueeze(0).to(device)

            # 预测
            pred = model(strain_seq, conditions)

            # 反归一化
            pred_np = pred[0].cpu().numpy()
            target_np = sample['stress_seq'].numpy()

            scaler = sample['stress_scaler']
            pred_denorm = scaler.inverse_transform(pred_np).flatten()
            target_denorm = scaler.inverse_transform(target_np).flatten()

            results.append({
                'material': sample['material'],
                'temperature': sample['temperature'],
                'strain_rate': sample['strain_rate'],
                'strain': sample['strain_seq_raw'],
                'stress_true': target_denorm,
                'stress_pred': pred_denorm,
            })

    return results

def _aggregate_by_temperature(results, material, strain_rate, grid_points=200):
    filtered = [r for r in results if r['material'] == material and
                abs(r['strain_rate'] - strain_rate) < 0.01]

    if not filtered:
        return []

    grouped = {}
    for r in filtered:
        grouped.setdefault(r['temperature'], []).append(r)

    aggregated = []
    for temp, group in grouped.items():
        min_max_strain = min(np.max(g['strain']) for g in group)
        max_min_strain = max(np.min(g['strain']) for g in group)
        if min_max_strain <= max_min_strain:
            continue

        grid = np.linspace(max_min_strain, min_max_strain, grid_points)
        true_stack = []
        pred_stack = []
        for g in group:
            true_stack.append(np.interp(grid, g['strain'], g['stress_true']))
            pred_stack.append(np.interp(grid, g['strain'], g['stress_pred']))

        aggregated.append({
            'temperature': temp,
            'strain': grid,
            'stress_true': np.mean(true_stack, axis=0),
            'stress_pred': np.mean(pred_stack, axis=0),
            'count': len(group),
        })

    return sorted(aggregated, key=lambda x: x['temperature'])


def plot_multi_temperature_comparison(results, material, strain_rate, output_path):
    """
    绘制多温度对比图（原始数据 vs 预测数据）

    Args:
        results: 预测结果列表
        material: 材料名称
        strain_rate: 应变率
        output_path: 输出路径
    """
    aggregated = _aggregate_by_temperature(results, material, strain_rate)
    if len(aggregated) == 0:
        return

    temps = [r['temperature'] for r in aggregated]
    norm = Normalize(vmin=min(temps), vmax=max(temps))
    cmap = colormaps.get_cmap('coolwarm')

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7))

    for result in aggregated:
        temp = result['temperature']
        color = cmap(norm(temp))

        ax.plot(result['strain'], result['stress_true'],
                color=color, linewidth=2.2, alpha=0.9)
        ax.plot(result['strain'], result['stress_pred'],
                color=color, linewidth=2.2, linestyle='--', alpha=0.9)

    # 图表美化
    ax.set_xlabel('Strain ε', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stress σ (MPa)', fontsize=12, fontweight='bold')

    title = f'{material.upper()} - Strain Rate: {strain_rate} s⁻¹'
    ax.set_title(title, fontsize=14, fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Exp. Data'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Pred.'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Temperature (°C)', fontsize=10)

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path.name}")

def plot_all_materials_comparison(results, output_dir):
    """为所有材料生成对比图"""
    output_dir = Path(output_dir)

    # 按材料和应变率分组
    groups = {}
    for r in results:
        key = (r['material'], r['strain_rate'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    print("\n生成可视化图表...")
    print("="*60)

    for (material, strain_rate), group_results in groups.items():
        # 检查是否有多个温度
        temps = set(r['temperature'] for r in group_results)
        if len(temps) < 2:
            continue

        # 生成文件名
        rate_str = f"{strain_rate}".replace('.', '_')
        filename = f"{material}_rate{rate_str}_multi_temp_comparison.png"
        output_path = output_dir / filename

        # 绘制对比图
        plot_multi_temperature_comparison(group_results, material, strain_rate, output_path)

def main():
    print("="*60)
    print("生成原始数据 vs 预测数据对比图")
    print("="*60)

    device = torch.device(CONFIG['device'])
    print(f"\n使用设备: {device}")

    # 加载模型
    print("\n加载模型...")
    model = load_model(CONFIG['model_path'], device)
    print("  模型加载成功")

    # 加载数据集
    print("\n加载数据集...")
    data_dir = Path(CONFIG['data_dir'])

    # 加载训练集和测试集
    train_dataset = StressStrainDataset(
        str(data_dir / "train.pkl"),
        fit_scaler=True,
        fit_condition_scaler=True,
    )
    shared_stress_scaler = train_dataset.stress_scaler
    shared_condition_scaler = train_dataset.condition_scaler
    test_dataset = StressStrainDataset(
        str(data_dir / "test.pkl"),
        stress_scaler=shared_stress_scaler,
        condition_scaler=shared_condition_scaler,
    )

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 预测
    print("\n进行预测...")
    train_results = predict_samples(model, train_dataset, device)
    test_results = predict_samples(model, test_dataset, device)

    all_results = train_results + test_results
    print(f"  完成预测: {len(all_results)} 个样本")

    # 生成对比图
    output_dir = prepare_output_dir(CONFIG['output_dir'], CONFIG.get('clear_output_dir', False))
    plot_all_materials_comparison(all_results, output_dir)

    print("\n" + "="*60)
    print("可视化完成！")
    print("="*60)
    print(f"\n输出目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
