# 铝合金应力应变曲线预测 - GBR基线 + PINN

## 项目简介
本项目是"机器学习基础理论及其在工程科学中的应用"课程的大作业，专注于预测铝合金材料在不同工况下的应力应变曲线。

**核心技术：** Gradient Boosting Regressor (GBR) + Physics-Informed Neural Network (PINN)
**研究材料：** Al2024, Al2219, Al6061, Al7075
**默认筛选：** 仅保留拉伸 (tensile) 数据
**GitHub仓库：** https://github.com/KIYOYOZU/ML-dazuoye

## 项目特色
- ✅ **双模型对比**：GBR基线 vs PINN物理约束模型
- ✅ **物理约束学习**：温度软化、应变率强化、边界条件
- ✅ **多材料多工况**：涵盖4种铝合金，多温度/应变率
- ✅ **Collocation Points**：零应变边界条件增强

## 模型性能

### PINN (Physics-Informed Neural Network)
| 材料 | R² | RMSE (MPa) |
|------|-----|------------|
| Al2024 | 0.947 | 29.97 |
| Al2219 | 0.912 | 5.84 |
| Al6061 | 0.515 | 29.52 |
| Al7075 | 0.739 | 31.74 |
| **整体** | **0.949** | **28.54** |

### 物理约束验证
| 约束 | 状态 |
|------|------|
| ✅ 温度软化 (T↑ → σ↓) | 正常 |
| ✅ 应变率强化 (ε̇↑ → σ↑) | 正常 |
| ⚠️ 边界条件 σ(ε=0)≈0 | 部分改善 |

## 数据来源

### Al6061-T651
- **来源：** Mendeley Data - Stress-strain curves of aluminum 6061-T651
- **工况：** 温度 20-300°C，应变率 0.001 s⁻¹

### Al2024
- **来源：** El-Magd, E., et al. (2024). Journal of Materials Research and Technology
- **工况：** 温度 200-350°C，应变率 0.1/1/10 s⁻¹

### Al2219
- **来源：** Metals (2021). Constitutive Modeling of Aluminum Alloy 2219
- **工况：** 温度 300-480°C，应变率 0.1/1/5/10 s⁻¹

### Al7075-T6
- **来源：** Frontiers in Materials, Materials MDPI, China Foundry
- **工况：** 温度 300-450°C，应变率 0.01/0.1/1/10 s⁻¹

## 项目结构
```
大作业/
├── data/
│   ├── processed/
│   │   └── 筛选数据_merged.csv    # 合并数据（真实应力应变）
│   └── 筛选数据/                   # 71条曲线源数据
│
├── src/
│   ├── stress_pinn.py              # PINN训练
│   ├── stress_pinn_visualize.py    # PINN可视化
│   ├── stress_train.py             # GBR训练
│   └── stress_visualize.py         # GBR可视化
│
├── results/
│   ├── baseline_gbr/               # GBR输出
│   ├── figures/                    # GBR可视化图
│   ├── figures_pinn/               # PINN可视化图
│   └── stress_pinn/                # PINN输出
│
└── requirements.txt
```

## 环境配置

### 安装依赖
```bash
conda create -n ml-materials python=3.9
conda activate ml-materials
conda install pytorch -c pytorch
conda install numpy pandas scikit-learn matplotlib seaborn
pip install joblib
```

## 使用方法

### 1. 训练 GBR 基线
```bash
python src/stress_train.py
```

### 2. GBR 可视化
```bash
python src/stress_visualize.py
```

### 3. 训练 PINN
```bash
python src/stress_pinn.py
```

### 4. PINN 可视化
```bash
python src/stress_pinn_visualize.py
```

## PINN 模型说明

### 输入特征（21维）
- 应变、温度、log(应变率)
- 材料ID、弹性模量
- Johnson-Cook参数 (c1~c5)
- 元素成分 (Si, Fe, Cu, Mn, Mg, Cr, Zn, Ti, Zr, V, Al)

### 物理约束
1. **单调性约束**: dσ/dε ≥ 0
2. **弹性模量约束**: dσ/dε|ε→0 ≈ E
3. **温度软化约束**: dσ/dT ≤ 0
4. **应变率强化约束**: dσ/d(log ε̇) ≥ 0

### Collocation Points
在应变=0处生成零应力训练点，强制边界条件 σ(ε=0) = 0

## 注意事项
- ⚠️ 本项目仅供学习和研究使用
- ⚠️ 使用数据时请引用原始文献
- ⚠️ Al6061 数据量较少，预测精度有限

## 联系方式
**GitHub仓库：** https://github.com/KIYOYOZU/ML-dazuoye

---
**最后更新：** 2026年1月5日
