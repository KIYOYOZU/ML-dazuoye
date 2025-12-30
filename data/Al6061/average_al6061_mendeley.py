"""
Al6061 Mendeley数据批次平均处理脚本
按材料批次(A-I)和重复实验次数进行平均，同时对所有批次也进行平均

输入: data/Al6061/*.csv (T_和P_开头的154个文件)
输出: data/筛选数据/al6061_*.csv (12个平均文件)

处理逻辑:
1. 对同一(测试类型, 温度, 批次)的重复实验(1/2/3次)进行平均
2. 再对所有批次(A-I)进行二次平均
3. 最终输出：6个温度 × 2种测试类型 = 12个文件

作者: Claude
日期: 2025-12-30
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.interpolate import interp1d

# ==================== 配置 ====================
INPUT_DIR = Path(r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\Al6061")
OUTPUT_DIR = Path(r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\筛选数据")
NUM_INTERP_POINTS = 200  # 插值点数，用于对齐不同曲线

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename):
    """
    解析文件名，提取关键信息

    示例: T_020_B_1_004_059_01.csv
    返回: {
        'test_type': 'T',     # T=拉伸, P=平面应变
        'temperature': '020', # 温度 (20°C)
        'batch': 'B',         # 材料批次 (A-I)
        'repeat': '1'         # 重复次数 (1/2/3)
    }
    """
    # 匹配模式：{测试类型}_{温度}_{批次}_{重复}_{其他}_...
    pattern = r'^([TP])_(\d{3})_([A-I])_(\d)_.*\.csv$'
    match = re.match(pattern, filename)

    if match:
        return {
            'test_type': match.group(1),
            'temperature': match.group(2),
            'batch': match.group(3),
            'repeat': match.group(4)
        }
    return None


def load_stress_strain_curve(filepath):
    """加载应力应变曲线数据"""
    try:
        df = pd.read_csv(filepath)
        # 确保列名一致
        if 'Strain' in df.columns and 'Stress_MPa' in df.columns:
            return df[['Strain', 'Stress_MPa']].dropna()
        else:
            print(f"警告: {filepath.name} 列名不匹配")
            return None
    except Exception as e:
        print(f"错误: 读取 {filepath.name} 失败 - {e}")
        return None


def interpolate_curve(strain, stress, num_points=200):
    """
    将应力应变曲线插值到固定数量的点

    参数:
        strain: 应变数组
        stress: 应力数组
        num_points: 插值点数

    返回:
        (strain_interp, stress_interp): 插值后的数组
    """
    # 排序（确保单调性）
    sorted_idx = np.argsort(strain)
    strain_sorted = strain[sorted_idx]
    stress_sorted = stress[sorted_idx]

    # 去除重复的应变值（保留第一个）
    unique_idx = np.unique(strain_sorted, return_index=True)[1]
    strain_unique = strain_sorted[unique_idx]
    stress_unique = stress_sorted[unique_idx]

    # 插值
    strain_min, strain_max = strain_unique.min(), strain_unique.max()
    strain_interp = np.linspace(strain_min, strain_max, num_points)

    # 使用线性插值
    f_interp = interp1d(strain_unique, stress_unique, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    stress_interp = f_interp(strain_interp)

    return strain_interp, stress_interp


def average_curves(curves_data):
    """
    对多条曲线进行平均

    参数:
        curves_data: list of (strain, stress) tuples

    返回:
        (strain_avg, stress_avg): 平均后的应力应变曲线
    """
    if len(curves_data) == 0:
        return None, None

    if len(curves_data) == 1:
        return curves_data[0]

    # 插值所有曲线到相同的应变点
    interpolated = []
    for strain, stress in curves_data:
        strain_interp, stress_interp = interpolate_curve(strain, stress, NUM_INTERP_POINTS)
        interpolated.append((strain_interp, stress_interp))

    # 计算平均应变和应力
    strain_arrays = [s for s, _ in interpolated]
    stress_arrays = [st for _, st in interpolated]

    strain_avg = np.mean(strain_arrays, axis=0)
    stress_avg = np.mean(stress_arrays, axis=0)

    return strain_avg, stress_avg


def main():
    """主处理流程"""
    print("=" * 70)
    print("Al6061 Mendeley数据批次平均处理")
    print("=" * 70)

    # 1. 扫描所有CSV文件
    all_files = list(INPUT_DIR.glob("*.csv"))
    test_files = [f for f in all_files if f.name.startswith(('T_', 'P_'))]

    print(f"\n找到 {len(test_files)} 个测试文件")

    # 2. 按(测试类型, 温度)分组（批次也进行平均）
    grouped_files = defaultdict(list)

    for filepath in test_files:
        info = parse_filename(filepath.name)
        if info is None:
            print(f"跳过无法解析的文件: {filepath.name}")
            continue

        # 分组键: (测试类型, 温度) - 不包含批次，所有批次都会被平均
        group_key = (info['test_type'], info['temperature'])
        grouped_files[group_key].append(filepath)

    print(f"\n共分为 {len(grouped_files)} 组 (测试类型, 温度)")

    # 3. 处理每一组
    processed_count = 0
    skipped_count = 0

    for group_key, filepaths in sorted(grouped_files.items()):
        test_type, temperature = group_key
        print(f"\n处理组: {test_type}_{temperature} ({len(filepaths)} 条曲线)")

        # 加载所有曲线
        curves_data = []
        for filepath in filepaths:
            df = load_stress_strain_curve(filepath)
            if df is not None and len(df) > 0:
                strain = df['Strain'].values
                stress = df['Stress_MPa'].values
                curves_data.append((strain, stress))
                print(f"  - {filepath.name}: {len(df)} 个数据点")

        if len(curves_data) == 0:
            print(f"  警告: 该组无有效数据，跳过")
            skipped_count += 1
            continue

        # 平均曲线
        strain_avg, stress_avg = average_curves(curves_data)

        if strain_avg is None:
            print(f"  警告: 平均失败，跳过")
            skipped_count += 1
            continue

        # 生成输出文件名: al6061_{测试类型}_{温度}.csv
        # 测试类型: T->tensile, P->plane_strain
        test_type_name = 'tensile' if test_type == 'T' else 'plane_strain'
        output_filename = f"al6061_{test_type_name}_{temperature}c.csv"
        output_path = OUTPUT_DIR / output_filename

        # 保存
        output_df = pd.DataFrame({
            'Strain': strain_avg,
            'Stress_MPa': stress_avg
        })
        output_df.to_csv(output_path, index=False)

        print(f"  [OK] 保存到: {output_filename} ({len(output_df)} 个数据点)")
        processed_count += 1

    # 4. 统计报告
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"成功处理: {processed_count} 组")
    print(f"跳过: {skipped_count} 组")
    print(f"输出目录: {OUTPUT_DIR}")

    # 生成统计文件
    stats_file = OUTPUT_DIR / "al6061_mendeley_averaged_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Al6061 Mendeley数据批次平均处理统计（含批次平均）\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"输入文件总数: {len(test_files)}\n")
        f.write(f"分组数（按测试类型+温度）: {len(grouped_files)}\n")
        f.write(f"成功处理: {processed_count} 组\n")
        f.write(f"跳过: {skipped_count} 组\n")
        f.write(f"输出文件数: {processed_count}\n\n")

        f.write("分组详情:\n")
        for group_key, filepaths in sorted(grouped_files.items()):
            test_type, temperature = group_key
            f.write(f"  {test_type}_{temperature}: {len(filepaths)} 条曲线（含所有批次）\n")

    print(f"\n统计报告已保存: {stats_file}")


if __name__ == "__main__":
    main()
