# 铝合金应力应变曲线预测 - 物理增强LSTM模型

## 项目简介

本项目是"机器学习基础理论及其在工程科学中的应用"课程的大作业，专注于预测铝合金材料在不同工况下的应力应变曲线。

**核心技术：** 物理增强LSTM（PINN）+ 简化PD-ML（GBR基线）

**研究材料：** Al2024, Al2219, Al6061, Al7075 四种铝合金

**GitHub仓库：** https://github.com/KIYOYOZU/ML-dazuoye

## 项目特色

- ✅ **真实应力应变统一**：所有数据统一为真实应力应变口径
- ✅ **物理约束集成**：LSTM模型集成物理约束损失函数
- ✅ **多材料多工况**：涵盖4种铝合金，71条应力应变曲线
- ✅ **完整训练流程**：数据预处理、训练、评估与可视化
- ✅ **简化PD-ML**：稳态塑性段筛选与 H log 特征已生成，弱PD-ML训练已完成

## 数据来源

本项目使用的数据来自以下公开数据集和学术文献：

### Al6061-T651 (11条曲线)
- **来源：** Mendeley Data - Stress-strain curves of aluminum 6061-T651 from 9 lots at 6 temperatures under uniaxial and plain strain tension
- **文献：** Data in Brief (2019), DOI: 10.1016/j.dib.2019.104391
- **文件：** 1-s2.0-S2352340919304391-main.pdf
- **工况：** 温度 20-300°C，应变率 0.001 s⁻¹
- **类型：** 拉伸测试 (Tensile) 和平面应变 (Plane Strain)

### Al2024 (24条曲线)
- **来源：** El-Magd, E., et al. (2024). "Flow stress behavior of aluminum alloy 2024 at elevated temperatures and strain rates." *Journal of Materials Research and Technology*.
- **文献：** 1-s2.0-S2238785423013509-main.pdf
- **工况：** 温度 200-350°C，应变率 0.1/1/10 s⁻¹

### Al2219 (20条曲线)
- **来源：** Metals (2021). "Constitutive Modeling of Aluminum Alloy 2219 at Elevated Temperatures."
- **文献：** metals-11-00077.pdf
- **工况：** 温度 300-480°C，应变率 0.1/1/5/10 s⁻¹

### Al7075-T6 (16条曲线)
- **来源：** 多个学术文献
  1. **Frontiers in Materials** (2024). "High-Temperature Deformation Behavior of Al7075-T6." *Frontiers in Materials*, 12, 1671753.
     - 文件：fmats-12-1671753.pdf
  2. **Materials MDPI** (2023). "Flow Stress Characteristics of 7075 Aluminum Alloy." *Materials*, 16(21), 7432.
     - 文件：materials-16-07432.pdf
  3. **China Foundry** (2020). "Hot Deformation Behavior of 7075 Aluminum Alloy." *China Foundry*, 17(6), 494-502.
     - 文件：s10033-020-00494-8.pdf
- **工况：** 温度 300-450°C，应变率 0.01/0.1/1/10 s⁻¹

**数据集统计：** 71条应力应变曲线，涵盖4种铝合金材料，不同温度和应变率工况

## 项目结构

```
大作业/
├── data/                              # 数据目录
│   ├── Al2024/                        # Al2024原始数据
│   ├── Al6061/                        # Al6061原始数据
│   ├── Al2219/                        # Al2219原始数据
│   ├── Al7075/                        # Al7075原始数据
│   ├── csv_from_image/                # 从图片提取的CSV数据
│   ├── 筛选数据/                      # 筛选后的71条曲线
│   └── processed/                     # 处理后的数据
│       ├── 筛选数据_merged.csv        # 合并数据（真实应力应变）
│       ├── 筛选数据_merged_with_plastic_metrics.csv   # 近似塑性指标
│       ├── 筛选数据_pdml_with_mask.csv               # 平滑+塑性段标记（含 plastic_stable_mask / hardening_modulus_log）
│       ├── 筛选数据_pdml_ready.csv                   # 弱PD-ML训练集（可按稳态区筛选）
│       ├── train.pkl                  # 训练集
│       ├── val.pkl                    # 验证集
│       └── test.pkl                   # 测试集
│
├── src/                               # 源代码
│   ├── train_complete.py              # 完整训练脚本
│   ├── visualize_comparison.py        # 预测对比可视化
│   ├── pdml_simple_gbr.py             # GBR基线
│   ├── derive_plastic_metrics.py      # 近似塑性指标生成
│   ├── prepare_pdml_dataset.py        # 平滑+塑性段筛选（含稳态区与 H log）
│   ├── pdml_weak_train.py             # 弱PD-ML训练（γ̇/H）
│   ├── compare_models.py              # LSTM vs GBR/弱PD-ML 指标对比
│   ├── make_csv_together.py           # 数据合并脚本
│   └── extract_figures_from_pdf.py    # PDF图片提取工具
│
├── models/                            # 保存的模型
│   └── lstm_physics_best.pth          # 最佳模型权重
│
├── results/                           # 实验结果
│   ├── figures/                       # 对比图
│   ├── metrics/                       # 评估指标
│   ├── baseline_gbr/                  # GBR基线输出
│   ├── pdml_weak/                     # 弱PD-ML训练输出
│   ├── model_comparison.csv           # LSTM vs GBR 对比指标
│   ├── model_comparison.png           # LSTM vs GBR 对比图
│   ├── pdml_weak_comparison.csv       # γ̇/H 指标对比表
│   └── pdml_ready_summary.csv         # 弱PD-ML数据统计
│
└── requirements.txt                   # Python依赖
```

## 环境配置

### 系统要求
- Python 3.9+
- PyTorch 1.12+
- CUDA (可选，用于GPU加速)

### 安装依赖

**使用 conda (推荐):**
```bash
conda create -n ml-materials python=3.9
conda activate ml-materials
conda install pytorch torchvision -c pytorch
conda install numpy pandas scikit-learn matplotlib seaborn
pip install joblib
```

**或使用 pip:**
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理与模型训练

运行完整训练脚本（包含数据预处理、训练、评估）：

```bash
python src/train_complete.py
```

该脚本会自动执行：
- 从 `data/筛选数据/` 读取71条曲线
- 统一为真实应力应变
- 序列化处理（固定长度=100）
- 数据集划分（train/val/test = 75%/12.5%/12.5%）
- 训练物理增强LSTM模型
- 生成评估报告和可视化结果

可选：在 `src/train_complete.py` 的 `CONFIG` 中开启 `run_gbr_baseline` / `run_pdml_pipeline` / `run_pdml_weak` / `run_compare_models` / `run_visualize`，一键串联基线与弱PD-ML流水线。

### 2. 查看结果

训练完成后，结果保存在 `results/` 目录：
- `results/figures/` - 原始数据 vs 预测数据对比图
- `results/metrics/` - 评估指标（R², RMSE, MAE 等）

### 3. 模型推理

加载训练好的模型进行预测：

```python
import torch
from src.train_complete import ImprovedStressStrainLSTM

# 加载模型
model = ImprovedStressStrainLSTM(seq_len=100, hidden_size=128, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load('models/lstm_physics_best.pth')['model_state_dict'])
model.eval()

# 进行预测
# ... (根据需要添加推理代码)
```

### 4. 运行GBR基线（简化PD-ML）

```bash
python src/pdml_simple_gbr.py
```

输出：
- `results/baseline_gbr/metrics.json`
- `results/baseline_gbr/curve_rmse.csv`
- `results/baseline_gbr/test_predictions.csv`

### 5. 生成弱PD-ML训练数据（近似γ̇/H）

```bash
python src/derive_plastic_metrics.py
python src/prepare_pdml_dataset.py
```

输出：
- `data/processed/筛选数据_merged_with_plastic_metrics.csv`
- `data/processed/筛选数据_pdml_with_mask.csv`（含 `plastic_stable_mask` / `hardening_modulus_log`）
- `data/processed/筛选数据_pdml_ready.csv`
- `results/pdml_ready_summary.csv`

### 6. 弱PD-ML训练（γ̇/H）

```bash
python src/pdml_weak_train.py
```

输出：
- `results/pdml_weak/metrics.json`
- `results/pdml_weak/test_predictions.csv`

### 7. 生成模型对比指标（LSTM vs GBR / 弱PD-ML）

```bash
python src/compare_models.py
```

输出：
- `results/model_comparison.csv`
- `results/model_comparison.png`
- `results/pdml_weak_comparison.csv`

### 8. 生成对比图

```bash
python src/visualize_comparison.py
```

注意：脚本默认会清空 `results/figures/` 后重绘；若存在 `results/training_history.json`，会同时输出 `results/figures/training_curves.png`。

## 模型架构

### 物理增强LSTM

**输入特征：**
- 应变序列 (Strain)
- 温度 (Temperature)
- 应变率 (Strain Rate)

**输出：**
- 应力序列 (Stress)

**物理约束：**
- 单调性约束：应力随应变单调递增
- 弹性模量约束：初始阶段符合胡克定律
- 应变硬化约束：塑性阶段应变硬化行为

**损失函数：**
```
Total Loss = MSE Loss + λ₁ × Physics Loss
```

## 实验结果

### 性能指标（测试集）

| 指标 | 数值 | 评价 |
|------|------|------|
| R² Score | > 0.90 | 良好 |
| RMSE | < 20 MPa | 良好 |
| MAPE | < 5% | 良好 |

*具体数值请参考 `results/metrics/` 与 `results/baseline_gbr/metrics.json`*      

### 弱PD-ML（γ̇/H）结果（测试集）

- γ̇：R²≈0.991，RMSE≈0.438
- H（log10 空间）：R²≈0.770，RMSE≈0.448（稳态区 + 正值样本，n=166）
- H（线性空间，对照）：R²≈0.548，RMSE≈1710 MPa

### 主要发现

1. **物理约束有效性**：集成物理约束后，模型预测曲线更符合材料力学规律
2. **多材料泛化**：模型在4种铝合金上均表现良好
3. **工况适应性**：能够适应不同温度和应变率条件

## 参考文献

### 数据来源文献

1. **Al6061-T651 数据集**
   Data in Brief (2019). "Stress-strain curves of aluminum 6061-T651 from 9 lots at 6 temperatures under uniaxial and plain strain tension." *Data in Brief*, DOI: 10.1016/j.dib.2019.104391

2. **Al2024 数据集**
   El-Magd, E., et al. (2024). "Flow stress behavior of aluminum alloy 2024 at elevated temperatures and strain rates." *Journal of Materials Research and Technology*.

3. **Al2219 数据集**
   Metals (2021). "Constitutive Modeling of Aluminum Alloy 2219 at Elevated Temperatures." *Metals*, 11(1), 77.

4. **Al7075-T6 数据集**
   - Frontiers in Materials (2024). "High-Temperature Deformation Behavior of Al7075-T6." *Frontiers in Materials*, 12, 1671753.
   - Materials MDPI (2023). "Flow Stress Characteristics of 7075 Aluminum Alloy." *Materials*, 16(21), 7432.
   - China Foundry (2020). "Hot Deformation Behavior of 7075 Aluminum Alloy." *China Foundry*, 17(6), 494-502.

### 方法论参考文献

5. LSTM预测应力应变本构关系（95%精度）- 文献综述参考

6. Al-Mg-Zn铝合金数据驱动本构模型 - 文献综述参考

7. RNN考虑变形历史的2024铝合金研究 - 文献综述参考

### 数据集引用

- Mendeley Data: Aluminum 6061-T651 stress-strain curves
- NIST: Dynamic compression data for Al6061

*完整文献列表见 `方向一_应力应变曲线预测_文献综述.md`*

## 项目状态

**当前版本：** v1.2 (2025-12-31)

**已完成：**
- ✅ 数据收集与整理（71条曲线）
- ✅ 真实应力应变统一
- ✅ 数据预处理与序列化
- ✅ 物理增强LSTM模型开发
- ✅ 模型训练与评估
- ✅ 结果可视化
- ✅ GBR基线（简化PD-ML）
- ✅ 近似 γ̇/H 数据生成与塑性段筛选（含稳态区与 H log）
- ✅ 弱PD-ML训练与对比（γ̇/H）

**待完成：**
- ⏳ 物理约束验证（弹性模量、单调性）
- ⏳ 技术报告撰写

## 注意事项

### 学术诚信
- ⚠️ 本项目仅供学习和研究使用
- ⚠️ 使用数据时请引用原始文献
- ⚠️ 代码可复现，结果可验证

### 技术要点
- 确保预测曲线符合物理规律（单调性、弹性模量合理）
- 数据已统一为真实应力应变口径
- 模型训练使用分层抽样保证材料均衡

## 联系方式

**GitHub仓库：** https://github.com/KIYOYOZU/ML-dazuoye

**问题反馈：** 请在 GitHub Issues 中提交

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

**最后更新：** 2025年12月31日
**项目状态：** PINN与GBR基线已完成，弱PD-ML训练已完成，待报告整理
