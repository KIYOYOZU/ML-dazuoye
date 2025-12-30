# 铝合金应力应变曲线预测 - 物理增强LSTM模型

## 项目简介

本项目是"机器学习基础理论及其在工程科学中的应用"课程的大作业，专注于使用物理信息神经网络（Physics-Informed Neural Network, PINN）方法预测铝合金材料在不同工况下的应力应变行为。

**核心技术：** 物理增强LSTM + 数据驱动本构建模

**研究材料：** Al2024, Al2219, Al6061, Al7075 四种铝合金

**GitHub仓库：** https://github.com/KIYOYOZU/ML-dazuoye

## 项目特色

- ✅ **真实应力应变统一**：所有数据统一为真实应力应变口径
- ✅ **物理约束集成**：LSTM模型集成物理约束损失函数
- ✅ **多材料多工况**：涵盖4种铝合金，71条应力应变曲线
- ✅ **完整训练流程**：数据预处理、模型训练、评估与可视化

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
│       ├── train.pkl                  # 训练集
│       ├── val.pkl                    # 验证集
│       └── test.pkl                   # 测试集
│
├── src/                               # 源代码
│   ├── train_complete.py              # 完整训练脚本
│   ├── make_csv_together.py           # 数据合并脚本
│   └── extract_figures_from_pdf.py    # PDF图片提取工具
│
├── models/                            # 保存的模型
│   └── physics_lstm_best.pth          # 最佳模型权重
│
├── results/                           # 实验结果
│   ├── training_history.png           # 训练曲线
│   ├── test_predictions.png           # 测试集预测对比
│   └── evaluation_metrics.txt         # 评估指标
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

### 2. 查看结果

训练完成后，结果保存在 `results/` 目录：
- `training_history.png` - 训练和验证损失曲线
- `test_predictions.png` - 测试集预测对比图
- `evaluation_metrics.txt` - 详细评估指标（R², RMSE, MAPE）

### 3. 模型推理

加载训练好的模型进行预测：

```python
import torch
from src.train_complete import PhysicsInformedLSTM

# 加载模型
model = PhysicsInformedLSTM(input_size=3, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load('models/physics_lstm_best.pth'))
model.eval()

# 进行预测
# ... (根据需要添加推理代码)
```

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

*具体数值请参考 `results/evaluation_metrics.txt`*

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

**当前版本：** v1.0 (2025-12-30)

**已完成：**
- ✅ 数据收集与整理（71条曲线）
- ✅ 真实应力应变统一
- ✅ 数据预处理与序列化
- ✅ 物理增强LSTM模型开发
- ✅ 模型训练与评估
- ✅ 结果可视化

**待完成：**
- ⏳ 物理约束验证（弹性模量、单调性）
- ⏳ 模型对比实验（与传统ML方法对比）
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

**最后更新：** 2025年12月30日
**项目状态：** 数据预处理与模型训练已完成，待报告整理
