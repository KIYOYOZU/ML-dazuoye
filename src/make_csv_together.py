#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合三种铝合金材料的应力应变数据并验证

功能：
1. 清洗fromimage中的数据（去除行号、箭头等）
2. 从文件名解析工况参数
3. 合并三种材料的所有数据
4. 数据验证（删除异常值、空值）
5. 生成统计报告
6. 生成可视化图表

作者: Claude Code
日期: 2025-12-29
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.signal import savgol_filter


class MaterialDataMerger:
    """铝合金材料数据合并器"""

    def __init__(self, base_dir: str):
        """
        初始化数据合并器

        Args:
            base_dir: 项目根目录
        """
        self.base_dir = Path(base_dir)
        self.fromimage_dir = self.base_dir / "data" / "fromimage"
        self.al6061_dir = self.base_dir / "data" / "Al6061"
        self.output_dir = self.base_dir / "data" / "processed"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.stats = {
            'Al2024': {'files': 0, 'curves': 0, 'points': 0},
            'Al6061': {'files': 0, 'curves': 0, 'points': 0},
            'Al7075': {'files': 0, 'curves': 0, 'points': 0},
        }

    def parse_al2024_filename(self, filename: str) -> Dict[str, float]:
        """
        解析Al2024文件名提取工况参数

        文件名格式: al2024_fig2a_200C_0.1s.csv

        Args:
            filename: 文件名

        Returns:
            包含温度和应变率的字典
        """
        # 匹配温度 (数字+C)
        temp_match = re.search(r'(\d+)C', filename)
        # 匹配应变率 (数字+s)
        rate_match = re.search(r'([\d.]+)s', filename)

        temperature = float(temp_match.group(1)) if temp_match else np.nan
        strain_rate = float(rate_match.group(1)) if rate_match else np.nan

        return {
            'temperature_C': temperature,
            'strain_rate': strain_rate
        }

    def parse_al7075_filename(self, filename: str) -> Dict[str, float]:
        """
        解析Al7075文件名提取工况参数

        文件名格式: al7075_fig1a_573K_0.01.csv

        Args:
            filename: 文件名

        Returns:
            包含温度和应变率的字典
        """
        # 匹配温度 (数字+K)
        temp_match = re.search(r'(\d+)K', filename)
        # 匹配应变率 (最后一个数字，可能是小数)
        rate_match = re.search(r'_([\d.]+)\.csv', filename)

        # 温度从K转换为C
        temperature_K = float(temp_match.group(1)) if temp_match else np.nan
        temperature_C = temperature_K - 273.15 if not np.isnan(temperature_K) else np.nan

        strain_rate = float(rate_match.group(1)) if rate_match else np.nan

        return {
            'temperature_C': temperature_C,
            'strain_rate': strain_rate
        }

    def clean_fromimage_data(self, filepath: str) -> pd.DataFrame:
        """
        清洗fromimage中的数据

        数据格式: "1→0.014308217034042031, 146.6101473848962"
        需要去除行号和箭头符号

        Args:
            filepath: CSV文件路径

        Returns:
            清洗后的DataFrame，包含strain和stress_MPa列
        """
        data_points = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # 跳过空行
                    if not line:
                        continue

                    # 去除行号和箭头（如果存在）
                    if '→' in line:
                        line = line.split('→')[1]

                    # 分割应变和应力
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            strain = float(parts[0].strip())
                            stress = float(parts[1].strip())
                            data_points.append({'strain': strain, 'stress_MPa': stress})
                        except ValueError:
                            # 跳过无法转换的行
                            continue

            df = pd.DataFrame(data_points)
            return df

        except Exception as e:
            print(f"WARNING 警告: 处理文件 {filepath} 时出错: {e}")
            return pd.DataFrame()

    def process_al2024_data(self) -> pd.DataFrame:
        """
        处理Al2024数据

        Returns:
            合并后的Al2024 DataFrame
        """
        print("\n" + "="*60)
        print("[DATA] 处理Al2024数据...")
        print("="*60)

        al2024_files = sorted(glob.glob(str(self.fromimage_dir / "al2024_*.csv")))
        all_data = []

        for filepath in al2024_files:
            filename = os.path.basename(filepath)
            print(f"  [FILE] 处理文件: {filename}")

            # 解析工况参数
            params = self.parse_al2024_filename(filename)

            # 清洗数据
            df = self.clean_fromimage_data(filepath)

            if df.empty:
                print(f"    WARNING 跳过空文件")
                continue

            # 添加元数据
            df['material'] = 'Al2024'
            df['temperature_C'] = params['temperature_C']
            df['strain_rate'] = params['strain_rate']
            df['source_file'] = filename
            df['curve_id'] = f"Al2024_{params['temperature_C']:.0f}C_{params['strain_rate']}s"

            all_data.append(df)

            # 更新统计
            self.stats['Al2024']['files'] += 1
            self.stats['Al2024']['points'] += len(df)

            print(f"    OK 提取 {len(df)} 个数据点")

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            self.stats['Al2024']['curves'] = len(al2024_files)
            print(f"\nDONE Al2024数据处理完成: {len(al2024_files)} 条曲线, {len(merged_df)} 个数据点")
            return merged_df
        else:
            print("ERROR 没有找到Al2024数据")
            return pd.DataFrame()

    def process_al7075_data(self) -> pd.DataFrame:
        """
        处理Al7075数据

        Returns:
            合并后的Al7075 DataFrame
        """
        print("\n" + "="*60)
        print("[DATA] 处理Al7075数据...")
        print("="*60)

        al7075_files = sorted(glob.glob(str(self.fromimage_dir / "al7075_*.csv")))
        all_data = []

        for filepath in al7075_files:
            filename = os.path.basename(filepath)
            print(f"  [FILE] 处理文件: {filename}")

            # 解析工况参数
            params = self.parse_al7075_filename(filename)

            # 清洗数据
            df = self.clean_fromimage_data(filepath)

            if df.empty:
                print(f"    WARNING 跳过空文件")
                continue

            # 添加元数据
            df['material'] = 'Al7075'
            df['temperature_C'] = params['temperature_C']
            df['strain_rate'] = params['strain_rate']
            df['source_file'] = filename
            df['curve_id'] = f"Al7075_{params['temperature_C']:.1f}C_{params['strain_rate']}s"

            all_data.append(df)

            # 更新统计
            self.stats['Al7075']['files'] += 1
            self.stats['Al7075']['points'] += len(df)

            print(f"    OK 提取 {len(df)} 个数据点")

        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            self.stats['Al7075']['curves'] = len(al7075_files)
            print(f"\nDONE Al7075数据处理完成: {len(al7075_files)} 条曲线, {len(merged_df)} 个数据点")
            return merged_df
        else:
            print("ERROR 没有找到Al7075数据")
            return pd.DataFrame()

    def process_al6061_data(self) -> pd.DataFrame:
        """
        处理Al6061数据（使用已合并的文件）

        Returns:
            Al6061 DataFrame
        """
        print("\n" + "="*60)
        print("[DATA] 处理Al6061数据...")
        print("="*60)

        merged_file = self.al6061_dir / "al6061_merged.csv"

        if not merged_file.exists():
            print(f"ERROR 未找到Al6061合并文件: {merged_file}")
            return pd.DataFrame()

        print(f"  [FILE] 读取文件: {merged_file.name}")

        try:
            df = pd.read_csv(merged_file)

            # 温度从K转换为C
            df['temperature_C'] = df['temperature'] - 273.15

            # 统一材料名称格式（首字母大写）
            df['material'] = 'Al6061'

            # 重命名列以匹配统一格式
            df = df.rename(columns={'stress': 'stress_MPa'})

            # 添加curve_id
            df['curve_id'] = (df['material'] + '_' +
                            df['temperature_C'].round(0).astype(str) + 'C_' +
                            df['strain_rate'].astype(str) + 's_' +
                            df['test_type'] + '_' +
                            df['lot'] + '_' +
                            df['specimen'].astype(str))

            # 选择需要的列
            df = df[['material', 'temperature_C', 'strain_rate', 'strain',
                    'stress_MPa', 'source_file', 'curve_id']]

            # 更新统计
            unique_curves = df['curve_id'].nunique()
            self.stats['Al6061']['files'] = 1
            self.stats['Al6061']['curves'] = unique_curves
            self.stats['Al6061']['points'] = len(df)

            print(f"  OK 读取 {unique_curves} 条曲线, {len(df)} 个数据点")
            print(f"\nDONE Al6061数据处理完成")

            return df

        except Exception as e:
            print(f"ERROR 读取Al6061数据失败: {e}")
            return pd.DataFrame()

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据验证和清洗

        Args:
            df: 待验证的DataFrame

        Returns:
            清洗后的DataFrame
        """
        print("\n" + "="*60)
        print("[CHECK] 数据验证与清洗...")
        print("="*60)

        initial_count = len(df)
        print(f"  初始数据点数: {initial_count}")

        # 1. 删除空值
        df = df.dropna(subset=['strain', 'stress_MPa', 'temperature_C', 'strain_rate'])
        print(f"  删除空值后: {len(df)} 个数据点 (删除 {initial_count - len(df)} 个)")

        # 2. 删除应力异常值（负应力或过大应力）
        # 保留负应变（压缩），但应力应该为正
        before_stress = len(df)
        df = df[(df['stress_MPa'] >= -1000) & (df['stress_MPa'] <= 1000)]  # 合理的应力范围
        print(f"  删除应力异常值后: {len(df)} 个数据点 (删除 {before_stress - len(df)} 个)")

        # 3. 删除应变异常值
        before_strain = len(df)
        df = df[(df['strain'] >= -1.0) & (df['strain'] <= 1.0)]  # 合理的应变范围
        print(f"  删除应变异常值后: {len(df)} 个数据点 (删除 {before_strain - len(df)} 个)")

        # 4. 删除温度异常值
        before_temp = len(df)
        df = df[(df['temperature_C'] >= -50) & (df['temperature_C'] <= 500)]
        print(f"  删除温度异常值后: {len(df)} 个数据点 (删除 {before_temp - len(df)} 个)")

        # 5. 删除应变率异常值
        before_rate = len(df)
        df = df[(df['strain_rate'] > 0) & (df['strain_rate'] <= 10000)]
        print(f"  删除应变率异常值后: {len(df)} 个数据点 (删除 {before_rate - len(df)} 个)")

        final_count = len(df)
        print(f"\nDONE 数据验证完成: 保留 {final_count}/{initial_count} 个数据点 ({100*final_count/initial_count:.1f}%)")

        return df

    def generate_statistics_report(self, merged_df: pd.DataFrame) -> str:
        """
        生成统计报告

        Args:
            merged_df: 合并后的DataFrame

        Returns:
            统计报告文本
        """
        report = []
        report.append("="*70)
        report.append("三种铝合金材料数据整合统计报告")
        report.append("="*70)
        report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 总体统计
        report.append("【总体统计】")
        report.append(f"  总数据点数: {len(merged_df):,}")
        report.append(f"  总曲线数: {merged_df['curve_id'].nunique()}")
        report.append(f"  材料种类: {merged_df['material'].nunique()}")
        report.append("")

        # 各材料统计
        report.append("【各材料统计】")
        for material in ['Al2024', 'Al6061', 'Al7075']:
            stats = self.stats[material]
            material_df = merged_df[merged_df['material'] == material]

            report.append(f"\n  {material}:")
            report.append(f"    - 源文件数: {stats['files']}")
            report.append(f"    - 曲线数: {stats['curves']}")
            report.append(f"    - 数据点数: {stats['points']:,}")

            if len(material_df) > 0:
                report.append(f"    - 温度范围: {material_df['temperature_C'].min():.1f}°C ~ {material_df['temperature_C'].max():.1f}°C")
                report.append(f"    - 应变率范围: {material_df['strain_rate'].min():.4f} ~ {material_df['strain_rate'].max():.1f} s^-1")
                report.append(f"    - 应变范围: {material_df['strain'].min():.4f} ~ {material_df['strain'].max():.4f}")
                report.append(f"    - 应力范围: {material_df['stress_MPa'].min():.2f} ~ {material_df['stress_MPa'].max():.2f} MPa")

        # 工况参数分布
        report.append("\n【工况参数分布】")
        report.append(f"\n  温度分布 (°C):")
        temp_counts = merged_df['temperature_C'].value_counts().sort_index()
        for temp, count in temp_counts.items():
            report.append(f"    {temp:6.1f}°C: {count:5d} 个数据点")

        report.append(f"\n  应变率分布 (s^-1):")
        rate_counts = merged_df['strain_rate'].value_counts().sort_index()
        for rate, count in rate_counts.items():
            report.append(f"    {rate:8.4f} s^-1: {count:5d} 个数据点")

        # 数据质量
        report.append("\n【数据质量】")
        report.append(f"  空值数量: {merged_df.isnull().sum().sum()}")
        report.append(f"  重复行数: {merged_df.duplicated().sum()}")

        report.append("\n" + "="*70)

        return "\n".join(report)

    def validate_merged_data(self, df: pd.DataFrame):
        """
        验证合并数据的质量

        Args:
            df: DataFrame
        """
        print("\n" + "="*70)
        print("[VALIDATE] 数据质量验证")
        print("="*70)

        # 检查列
        print("\n[1] 检查数据列...")
        required_columns = ['material', 'temperature_C', 'strain_rate', 'strain',
                           'stress_MPa', 'source_file', 'curve_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"    ERROR 缺少列: {missing_columns}")
        else:
            print(f"    OK 所有必需列存在")

        # 检查空值
        print("\n[2] 检查空值...")
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            print(f"    OK 无空值")
        else:
            print(f"    WARNING 发现空值:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"        {col}: {count}")

        # 检查重复
        print("\n[3] 检查重复...")
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            print(f"    OK 无重复行")
        else:
            print(f"    WARNING 发现 {duplicates} 个重复行")

        # 检查材料
        print("\n[4] 检查材料分布...")
        for material in df['material'].unique():
            count = len(df[df['material'] == material])
            curves = df[df['material'] == material]['curve_id'].nunique()
            print(f"    {material}: {count:,} 数据点, {curves} 条曲线")

        # 检查数值范围
        print("\n[5] 检查数值范围...")
        print(f"    温度: {df['temperature_C'].min():.1f}°C ~ {df['temperature_C'].max():.1f}°C")
        print(f"    应变率: {df['strain_rate'].min():.4f} ~ {df['strain_rate'].max():.1f} s^-1")
        print(f"    应变: {df['strain'].min():.4f} ~ {df['strain'].max():.4f}")
        print(f"    应力: {df['stress_MPa'].min():.2f} ~ {df['stress_MPa'].max():.2f} MPa")

        # 检查异常值
        print("\n[6] 检查异常值...")
        outliers = 0
        if df['stress_MPa'].abs().max() > 1000:
            print(f"    WARNING 发现异常应力值")
            outliers += 1
        if df['strain'].abs().max() > 1.0:
            print(f"    WARNING 发现异常应变值")
            outliers += 1
        if outliers == 0:
            print(f"    OK 无明显异常值")

        print("\n" + "="*70)
        print("验证完成！")
        print("="*70)

    def generate_visualization(self, df: pd.DataFrame):
        """
        生成可视化报告

        Args:
            df: DataFrame
        """
        print("\n" + "="*70)
        print("[PLOT] 生成可视化报告...")
        print("="*70)

        output_path = self.output_dir / "validation_plots"
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 材料分布饼图
        print("\n[1] 生成材料分布图...")
        plt.figure(figsize=(10, 6))
        material_counts = df['material'].value_counts()
        plt.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
        plt.title('Material Distribution')
        plt.savefig(output_path / 'material_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    OK 已保存: material_distribution.png")

        # 2. 温度分布直方图
        print("\n[2] 生成温度分布图...")
        plt.figure(figsize=(12, 6))
        for material in df['material'].unique():
            material_df = df[df['material'] == material]
            plt.hist(material_df['temperature_C'], bins=20, alpha=0.5, label=material)
        plt.xlabel('Temperature (C)')
        plt.ylabel('Frequency')
        plt.title('Temperature Distribution by Material')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'temperature_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    OK 已保存: temperature_distribution.png")

        # 3. 应变率分布
        print("\n[3] 生成应变率分布图...")
        plt.figure(figsize=(10, 6))
        rate_counts = df['strain_rate'].value_counts().sort_index()
        plt.bar(range(len(rate_counts)), rate_counts.values)
        plt.xticks(range(len(rate_counts)), [f'{r:.4f}' for r in rate_counts.index], rotation=45)
        plt.xlabel('Strain Rate (s^-1)')
        plt.ylabel('Number of Data Points')
        plt.title('Strain Rate Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'strain_rate_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    OK 已保存: strain_rate_distribution.png")

        # 4. 应力应变曲线示例（每种材料选1条）
        print("\n[4] 生成应力应变曲线示例...")
        plt.figure(figsize=(12, 8))
        for i, material in enumerate(['Al2024', 'Al6061', 'Al7075']):
            material_df = df[df['material'] == material]
            if len(material_df) > 0:
                # 选择第一条曲线
                first_curve_id = material_df['curve_id'].iloc[0]
                curve_data = material_df[material_df['curve_id'] == first_curve_id]
                plt.plot(curve_data['strain'], curve_data['stress_MPa'],
                        label=f'{material} ({first_curve_id})', linewidth=2)

        plt.xlabel('Strain', fontsize=12)
        plt.ylabel('Stress (MPa)', fontsize=12)
        plt.title('Stress-Strain Curves - Sample from Each Material', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'stress_strain_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    OK 已保存: stress_strain_samples.png")

        print("\n" + "="*70)
        print(f"可视化报告生成完成！输出目录: {output_path}")
        print("="*70)

    def merge_all_materials(self) -> Tuple[pd.DataFrame, str]:
        """
        合并所有材料数据

        Returns:
            (合并后的DataFrame, 统计报告文本)
        """
        print("\n" + "="*70)
        print(">>> 开始整合三种铝合金材料数据")
        print("="*70)

        # 处理各材料数据
        al2024_df = self.process_al2024_data()
        al7075_df = self.process_al7075_data()
        al6061_df = self.process_al6061_data()

        # 合并数据
        print("\n" + "="*60)
        print("[MERGE] 合并所有材料数据...")
        print("="*60)

        all_dfs = [df for df in [al2024_df, al6061_df, al7075_df] if not df.empty]

        if not all_dfs:
            print("ERROR 没有可合并的数据")
            return pd.DataFrame(), ""

        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"  OK 初步合并: {len(merged_df)} 个数据点")

        # 数据验证
        merged_df = self.validate_data(merged_df)

        # 重新排列列顺序
        column_order = ['material', 'temperature_C', 'strain_rate', 'strain',
                       'stress_MPa', 'source_file', 'curve_id']
        merged_df = merged_df[column_order]

        # 按材料、温度、应变率、应变排序
        merged_df = merged_df.sort_values(['material', 'temperature_C', 'strain_rate', 'strain'])
        merged_df = merged_df.reset_index(drop=True)

        # 生成统计报告
        report = self.generate_statistics_report(merged_df)

        return merged_df, report

    def save_results(self, merged_df: pd.DataFrame, report: str):
        """
        保存合并结果和统计报告

        Args:
            merged_df: 合并后的DataFrame
            report: 统计报告文本
        """
        print("\n" + "="*60)
        print("[SAVE] 保存结果...")
        print("="*60)

        # 保存CSV文件
        output_csv = self.output_dir / "all_materials_merged.csv"
        merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"  OK 数据已保存至: {output_csv}")
        print(f"    文件大小: {output_csv.stat().st_size / 1024 / 1024:.2f} MB")

        # 保存统计报告
        stats_file = self.output_dir / "merge_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  OK 统计报告已保存至: {stats_file}")

        print("\n" + "="*70)
        print("DONE 所有任务完成！")
        print("="*70)


def main():
    """主函数"""
    # 项目根目录
    base_dir = r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业"

    # 创建合并器
    merger = MaterialDataMerger(base_dir)

    # 执行合并
    merged_df, report = merger.merge_all_materials()

    if not merged_df.empty:
        # 打印统计报告
        print("\n" + report)

        # 保存结果
        merger.save_results(merged_df, report)

        # 验证数据质量
        merger.validate_merged_data(merged_df)

        # 生成可视化
        merger.generate_visualization(merged_df)

        print("\n\n" + "="*70)
        print("所有任务完成！数据整合、验证、可视化已完成。")
        print("="*70)
    else:
        print("\nERROR 数据合并失败，没有可用数据")


class FilteredDataProcessor:
    """筛选数据处理器 - 处理data/筛选数据/目录下的CSV文件"""

    def __init__(self):
        """初始化路径配置"""
        self.base_dir = Path(r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业")
        self.input_dir = self.base_dir / "data" / "筛选数据"
        self.output_fig_dir = self.base_dir / "data" / "figures"
        self.output_csv = self.base_dir / "data" / "processed" / "筛选数据_merged.csv"

        # 创建输出目录
        self.output_fig_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # 材料默认应力应变格式（基于论文文本线索）
        self.material_default_format = {
            'al2024': 'true',
            'al2219': 'true',
            'al7075': 'true',
            'al6061': 'engineering',
        }

    def parse_filename_robust(self, filename: str) -> dict:
        """
        鲁棒文件名解析（支持al6061/al2024/al7075/al2219）

        支持4种命名格式:
        - al6061_tensile_020c.csv          → material, test_type, temperature
        - al2024_fig2a_200C_0.1s.csv       → material, series, temperature, strain_rate
        - al7075_fig1a_573K_0.01.csv       → material, temperature(K→C), strain_rate
        - al2219_fig3a_300C_0.1s.csv       → material, temperature, strain_rate

        Args:
            filename: 文件名

        Returns:
            包含material, temperature, strain_rate, test_type, series的字典
        """
        # 规范化文件名（处理双重.csv后缀）
        filename = filename.replace('.csv.csv', '.csv')
        filename_lower = filename.lower()

        # 提取材料
        material_match = re.search(r'(al\d+)', filename_lower)
        material = material_match.group(1) if material_match else None

        # 提取温度（支持C和K）
        # 使用更精确的正则：要求温度值后紧跟C或K，且后面是下划线、点或文件结尾
        temp_c_match = re.search(r'_(\d+)c[_.]', filename_lower)
        temp_k_match = re.search(r'_(\d+)k[_.]', filename_lower)
        if temp_c_match:
            temperature = int(temp_c_match.group(1))
        elif temp_k_match:
            temperature = int(temp_k_match.group(1)) - 273  # K转C
        else:
            temperature = None

        # 提取应变率
        rate_match = re.search(r'([\d.]+)s', filename_lower)
        if rate_match:
            strain_rate = float(rate_match.group(1))
        else:
            # Al7075格式: al7075_fig1a_573K_0.01.csv （应变率在K和.csv之间）
            alt_rate_match = re.search(r'k_([\d.]+)\.csv', filename_lower)
            if alt_rate_match:
                strain_rate = float(alt_rate_match.group(1))
            else:
                strain_rate = 0.001  # 默认值（Al6061）

        # 提取测试类型
        if 'plane_strain' in filename_lower or 'planestrain' in filename_lower:
            test_type = 'plane_strain'
        else:
            test_type = 'tensile'

        # 提取系列标识（fig2a/fig8a等）
        series_match = re.search(r'(fig\d+[a-z])', filename_lower)
        series = series_match.group(1) if series_match else None

        return {
            'material': material,
            'temperature': temperature,
            'strain_rate': strain_rate,
            'test_type': test_type,
            'series': series  # 用于区分Al2024的fig2/fig8
        }

    def load_filtered_data(self) -> pd.DataFrame:
        """
        加载所有CSV文件并添加元数据

        Returns:
            包含所有数据和元数据的DataFrame
        """
        all_data = []

        csv_files = list(self.input_dir.glob("*.csv"))
        print(f"  发现 {len(csv_files)} 个CSV文件")

        for csv_file in csv_files:
            try:
                # 解析文件名
                metadata = self.parse_filename_robust(csv_file.name)

                # 读取CSV（自动检测列头）
                # 尝试读取第一行判断是否有列头
                with open(csv_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                # 如果第一行包含"Strain"，说明有列头
                has_header = 'Strain' in first_line or 'strain' in first_line

                if has_header:
                    df = pd.read_csv(csv_file)
                    # 标准化列名
                    df.columns = [col.lower().replace('_', '') for col in df.columns]
                    if 'stressmpa' in df.columns:
                        df = df.rename(columns={'stressmpa': 'stress_mpa'})
                else:
                    # 无列头，指定列名
                    df = pd.read_csv(csv_file, header=None, names=['strain', 'stress_MPa'])

                # 统一列名格式
                if 'stress_mpa' in df.columns:
                    df = df.rename(columns={'stress_mpa': 'stress_MPa'})

                # Al7075负值修正（取绝对值）
                if metadata['material'] == 'al7075':
                    df['strain'] = df['strain'].abs()
                    df['stress_MPa'] = df['stress_MPa'].abs()
                    # print(f"  ⚠️  Al7075方向修正: {csv_file.name}")

                # 添加元数据
                df['source_file'] = csv_file.name
                df['material'] = metadata['material']
                df['temperature'] = metadata['temperature']
                df['strain_rate'] = metadata['strain_rate']
                df['test_type'] = metadata['test_type']
                df['series'] = metadata['series']
                df['is_converted'] = False  # 初始标记

                all_data.append(df)

            except Exception as e:
                print(f"  ⚠️  解析失败: {csv_file.name} - {e}")

        # 合并所有数据
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)

            # 基础清洗
            merged_df = merged_df.dropna(subset=['strain', 'stress_MPa'])
            merged_df = merged_df[(merged_df['strain'].abs() < 2.0) & (merged_df['stress_MPa'].abs() < 1500)]

            return merged_df
        else:
            return pd.DataFrame()

    def smooth_curve(self, strain: np.ndarray, stress: np.ndarray, window_length: int = 11, polyorder: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Savitzky-Golay滤波器平滑曲线

        Args:
            strain: 应变数组
            stress: 应力数组
            window_length: 窗口长度（必须是奇数，默认11）
            polyorder: 多项式阶数（默认3）

        Returns:
            平滑后的应变和应力数组
        """
        if len(strain) < window_length:
            # 数据点太少，不平滑
            return strain, stress

        # 确保窗口长度是奇数
        if window_length % 2 == 0:
            window_length += 1

        # 按应变排序
        sorted_idx = np.argsort(strain)
        strain_sorted = strain[sorted_idx]
        stress_sorted = stress[sorted_idx]

        try:
            # 平滑应力数据（保持应变不变）
            stress_smoothed = savgol_filter(stress_sorted, window_length, polyorder)

            # 恢复原始顺序
            original_order = np.argsort(sorted_idx)
            return strain_sorted[original_order], stress_smoothed[original_order]
        except Exception as e:
            # 平滑失败，返回原始数据
            print(f"    警告：平滑失败 - {e}")
            return strain, stress

    def truncate_at_necking(self, strain: np.ndarray, stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在颈缩点截断数据（保留颈缩前的数据）

        Args:
            strain: 应变数组
            stress: 应力数组

        Returns:
            截断后的应变和应力数组
        """
        # 找到应力峰值点
        max_idx = np.argmax(np.abs(stress))

        # 截断到峰值点（包含峰值点）
        return strain[:max_idx+1], stress[:max_idx+1]

    def detect_and_convert_to_true_stress(self, df: pd.DataFrame, enable_smoothing: bool = True) -> pd.DataFrame:
        """
        批量应力转换（逐组检测颈缩特征）+ 数据平滑/截断

        转换规则（材料优先）:
        - al2024/al2219/al7075: 视为已是真应力应变，不再转换
        - al6061: 视为工程应力应变，统一转换为真应力应变
        - 其他材料: 使用原判定条件（颈缩特征/大变形）

        处理策略：
        - Al6061: 在颈缩点截断（不平滑）
        - 其他材料: 平滑处理

        Args:
            df: 包含所有数据的DataFrame
            enable_smoothing: 是否启用数据平滑（默认True）

        Returns:
            转换后的DataFrame
        """
        converted_groups = []

        # 按曲线分组（source_file）
        for file_name, group_df in df.groupby('source_file'):
            strain = group_df['strain'].values
            stress = group_df['stress_MPa'].values
            material = group_df['material'].iloc[0]

            # 颈缩检测
            max_idx = np.argmax(np.abs(stress))
            max_stress = np.abs(stress[max_idx])
            end_stress = np.abs(stress[-1]) if len(stress) > 0 else 0
            drop_ratio = (max_stress - end_stress) / max_stress if max_stress > 0 else 0

            # Al6061特殊处理：在颈缩点截断
            if material == 'al6061' and drop_ratio > 0.20:
                strain, stress = self.truncate_at_necking(strain, stress)
                print(f"  [截断] {file_name} (颈缩点截断，保留{len(strain)}个点)")
            # 其他材料：平滑处理
            elif enable_smoothing and len(strain) >= 11:
                strain, stress = self.smooth_curve(strain, stress, window_length=11, polyorder=3)

            # 判断是否需要转换（使用处理后的数据重新计算）
            max_idx = np.argmax(np.abs(stress))
            max_stress = np.abs(stress[max_idx])
            end_stress = np.abs(stress[-1]) if len(stress) > 0 else 0
            drop_ratio = (max_stress - end_stress) / max_stress if max_stress > 0 else 0
            is_engineering = (drop_ratio > 0.20) or (np.max(np.abs(strain)) > 0.10)

            material_format = self.material_default_format.get(material)
            if material_format == 'true':
                group_df = group_df.copy()
                if len(strain) < len(group_df):
                    group_df = group_df.iloc[:len(strain)].copy()
                group_df['strain'] = strain
                group_df['stress_MPa'] = stress
                group_df['is_converted'] = True
                converted_groups.append(group_df)
                continue

            if material_format == 'engineering':
                should_convert = True
            else:
                should_convert = is_engineering

            if should_convert and len(strain) > 0:
                # 应用转换公式
                strain_abs = np.maximum(strain, 0)  # 确保非负
                true_strain = np.log(1 + strain_abs)
                true_stress = stress * (1 + strain_abs)

                group_df = group_df.copy()
                # 如果数据被截断，需要重建DataFrame
                if len(strain) < len(group_df):
                    group_df = group_df.iloc[:len(strain)].copy()

                group_df['strain'] = true_strain
                group_df['stress_MPa'] = true_stress
                group_df['is_converted'] = True

                print(f"  [转换] {file_name} (drop_ratio={drop_ratio:.2f})")
            else:
                # 即使不转换，也更新处理后的数据
                group_df = group_df.copy()
                if len(strain) < len(group_df):
                    group_df = group_df.iloc[:len(strain)].copy()
                group_df['strain'] = strain
                group_df['stress_MPa'] = stress

            converted_groups.append(group_df)

        return pd.concat(converted_groups, ignore_index=True) if converted_groups else pd.DataFrame()

    def group_by_material_rate_series(self, df: pd.DataFrame) -> dict:
        """
        分组策略（考虑series字段）

        分组规则:
        - Al6061: 按test_type分组（tensile, plane_strain）
        - Al2024: 按series分组（fig2a, fig2b, fig8a等）
        - Al2219: 按series分组（fig3a, fig3b等）
        - Al7075: 按strain_rate分组（0.01, 0.1, 1.0, 10）

        Args:
            df: DataFrame

        Returns:
            {group_key: group_df} 字典
        """
        groups = {}

        for (material, test_type, strain_rate, series), group_df in df.groupby(
            ['material', 'test_type', 'strain_rate', 'series'], dropna=False
        ):
            # 生成唯一键
            if material == 'al6061':
                # Al6061按test_type分组
                key = f"{material}_{test_type}"
            elif material == 'al2024':
                # Al2024按series分组（fig2a, fig2b, fig8a等）
                if pd.notna(series):
                    key = f"{material}_{series}"
                else:
                    # 如果没有series，按应变率分组
                    rate_str = f"{strain_rate}".replace('.', '_')
                    key = f"{material}_{rate_str}s"
            elif material == 'al2219':
                # Al2219按series分组（fig3a, fig3b等）
                if pd.notna(series):
                    key = f"{material}_{series}"
                else:
                    rate_str = f"{strain_rate}".replace('.', '_')
                    key = f"{material}_{rate_str}s"
            elif material == 'al7075':
                # Al7075明确按strain_rate分组（忽略series）
                rate_str = f"{strain_rate}".replace('.', '_')
                key = f"{material}_{rate_str}s"
            else:
                # 其他材料按应变率分组
                rate_str = f"{strain_rate}".replace('.', '_')
                key = f"{material}_{rate_str}s"

            groups[key] = group_df

        return groups

    def plot_multi_temperature(self, group_df: pd.DataFrame, output_path: Path, title: str):
        """
        绘制多温度对比曲线

        Args:
            group_df: 同材料同应变率不同温度的数据
            output_path: 输出图片完整路径
            title: 图表标题
        """
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # 温度排序
        temps = sorted(group_df['temperature'].unique())

        # 颜色映射（蓝色→红色 = 低温→高温）
        norm = Normalize(vmin=min(temps), vmax=max(temps))
        try:
            from matplotlib import colormaps
            cmap = colormaps.get_cmap('coolwarm')
        except Exception:
            cmap = cm.get_cmap('coolwarm')

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        for temp in temps:
            temp_df = group_df[group_df['temperature'] == temp].sort_values('strain')
            color = cmap(norm(temp))
            ax.plot(
                temp_df['strain'],
                temp_df['stress_MPa'],
                label=f'{int(temp)}°C',
                color=color,
                linewidth=2,
                alpha=0.8
            )

        # 图表美化
        ax.set_xlabel('Strain', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # print(f"  ✓ 保存图片: {output_path.name}")

    def merge_to_csv(self, df: pd.DataFrame):
        """
        整合CSV文件

        Args:
            df: 包含所有数据的DataFrame
        """
        # 选择关键列
        output_df = df[[
            'source_file',      # 原始文件名
            'material',         # 材料
            'test_type',        # 测试类型
            'series',           # 系列（fig2a/fig8a等）
            'temperature',      # 温度（°C）
            'strain_rate',      # 应变率（s⁻¹）
            'strain',           # 应变值
            'stress_MPa',       # 应力值
            'is_converted'      # 是否为真应力应变（已转换或原始即为真实）
        ]].copy()

        # 排序
        output_df = output_df.sort_values([
            'material', 'test_type', 'strain_rate',
            'temperature', 'strain'
        ])

        # 保存
        output_df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')

        # 统计报告
        print(f"\n[统计] CSV整合统计:")
        print(f"  总数据点: {len(output_df):,}")
        print(f"  曲线数: {output_df['source_file'].nunique()}")
        converted_count = output_df.groupby('source_file')['is_converted'].first().sum()
        print(f"  真应力曲线数: {converted_count}")
        print(f"  材料分布:")
        for mat in sorted(output_df['material'].unique()):
            count = output_df[output_df['material'] == mat]['source_file'].nunique()
            print(f"    {mat}: {count}条")

    def run(self):
        """主执行流程"""
        print("=" * 60)
        print("筛选数据处理与可视化流程")
        print("=" * 60)

        # 步骤1: 加载数据
        print("\n[1/5] 加载筛选数据...")
        df = self.load_filtered_data()
        print(f"  OK 加载完成: {len(df):,} 行, {df['source_file'].nunique()} 个文件")

        # 步骤2: 应力转换
        print("\n[2/5] 应力应变格式统一（按材料规则）...")
        df = self.detect_and_convert_to_true_stress(df)
        converted = df.groupby('source_file')['is_converted'].first().sum()
        print(f"  OK 真应力曲线数: {converted} 条曲线")

        # 步骤3: 数据分组
        print("\n[3/5] 数据分组...")
        groups = self.group_by_material_rate_series(df)
        print(f"  OK 分组完成: {len(groups)} 个组")

        # 步骤4: 生成可视化
        print("\n[4/5] 生成多温度对比图...")
        plot_count = 0
        for group_key, group_df in groups.items():
            # 生成标题和文件名
            material = group_df['material'].iloc[0]
            test_type = group_df['test_type'].iloc[0]
            strain_rate = group_df['strain_rate'].iloc[0]
            series = group_df['series'].iloc[0] if pd.notna(group_df['series'].iloc[0]) else ''

            if series:
                title = f"{material.upper()} - {series} ({strain_rate} s^-1)"
            else:
                title = f"{material.upper()} - {test_type} ({strain_rate} s^-1)"

            filename = f"{group_key}_multi_temp.png"
            output_path = self.output_fig_dir / filename

            self.plot_multi_temperature(group_df, output_path, title)
            plot_count += 1

        print(f"  OK 生成完成: {plot_count} 张图片")

        # 步骤5: 整合CSV
        print("\n[5/5] 整合CSV文件...")
        self.merge_to_csv(df)
        print(f"  OK 保存成功: {self.output_csv}")

        print("\n" + "=" * 60)
        print("DONE 处理完成！")
        print("=" * 60)
        print(f"\n输出位置:")
        print(f"  图片: {self.output_fig_dir}")
        print(f"  CSV: {self.output_csv}")


if __name__ == "__main__":
    # main()  # 原有的MaterialDataMerger流程

    # 新增：筛选数据处理流程
    processor = FilteredDataProcessor()
    processor.run()
