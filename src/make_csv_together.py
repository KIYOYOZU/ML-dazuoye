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


if __name__ == "__main__":
    main()
