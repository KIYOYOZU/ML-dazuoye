# 筛选数据 CSV 参数指南

生成时间：2025-12-30 22:21
数据目录：D:/AAA_postgraduate/FirstSemester/教材/机器学习工程应用/大作业/data/筛选数据

## 参数字段说明
- material：由文件名解析得到（al2024/al2219/al6061/al7075）
- test_type：tensile 或 plane_strain（仅 al6061 文件名显式含该字段）
- series：图号系列（如 fig2a/fig8a/fig3a），无则为 -
- temperature_C：摄氏温度（al7075 文件名为 K，按 -273 转为 °C）
- strain_rate：应变率，单位 s^-1（al6061 文件名未包含时默认 0.001）

## 应力应变格式说明
- 本节指 `data/筛选数据/` 原始 CSV 的格式归类（基于论文文本线索与处理规则）
- `data/processed/筛选数据_merged.csv` 已统一为真实应力应变格式

| material | data/筛选数据 原始格式 | processed/筛选数据_merged.csv |
|---|---|---|
| al2024 | 真实应力应变 | 真实应力应变 |
| al2219 | 真实应力应变 | 真实应力应变 |
| al6061 | 工程应力应变 | 真实应力应变（已转换） |
| al7075 | 真实应力应变 | 真实应力应变 |

## 材料差异要点（结合 src/make_csv_together.py 与 src/train_complete.py）
- al6061：文件名包含 test_type（tensile/plane_strain），温度单位为 °C；无显式应变率时默认 0.001 s^-1。
- al6061：来源为 Data in Brief 2019（6061-T651，单轴/平面应变拉伸），对应论文PDF：D:/AAA_postgraduate/FirstSemester/教材/机器学习工程应用/大作业/data/Al6061/1-s2.0-S2352340919304391-main.pdf；对应Mendeley数据集压缩包：D:/AAA_postgraduate/FirstSemester/教材/机器学习工程应用/大作业/data/Al6061/stress-strain-curves-of-aluminum-6061-t651-from-9-lots-at-6-temperatures-under-uniaxial-and-plain-strain-tension.zip。
- al2024：文件名包含 series（fig2*/fig8*），温度为 °C，应变率为 0.1/1.0/10 s^-1。
- al2219：文件名包含 series（fig3*），温度为 °C，应变率为 0.1/1.0/5.0/10 s^-1。
- al7075：文件名温度为 K，解析时减 273 转 °C；应变率位于 K 与 .csv 之间；筛选数据加载时应力/应变取绝对值（防止方向符号影响）。
- 训练脚本要求：train_complete.py 中按 material_id 索引 4 类材料的物理约束参数（UTS/权重），预处理阶段需保证 material_id 与材料顺序一致。

## 材料参数概览
| material | 文件数 | 温度范围(°C) | 应变率范围(s^-1) | test_type | series |
|---|---:|---|---|---|---|
| al2024 | 24 | 200 ~ 350 | 0.1 ~ 10 | tensile | fig2a, fig2b, fig2c, fig8a, fig8b, fig8c |
| al2219 | 20 | 300 ~ 480 | 0.1 ~ 10 | tensile | fig3a, fig3b, fig3d, fig3f |
| al6061 | 11 | 20 ~ 300 | 0.001 ~ 0.001 | plane_strain, tensile | - |
| al7075 | 16 | 300 ~ 450 | 0.01 ~ 10 | tensile | fig1a |

## CSV 参数清单

### al2024
| 文件 | material | test_type | series | temperature_C | strain_rate |
|---|---|---|---|---:|---:|
| al2024_fig2a_200C_0.1s.csv | al2024 | tensile | fig2a | 200 | 0.1 |
| al2024_fig8a_200C_0.1s_exp.csv.csv | al2024 | tensile | fig8a | 200 | 0.1 |
| al2024_fig2b_200C_1.0s.csv | al2024 | tensile | fig2b | 200 | 1 |
| al2024_fig8b_200C_1.0s_exp.csv | al2024 | tensile | fig8b | 200 | 1 |
| al2024_fig2c_200C_10s.csv | al2024 | tensile | fig2c | 200 | 10 |
| al2024_fig8c_200C_10s_exp.csv | al2024 | tensile | fig8c | 200 | 10 |
| al2024_fig2a_250C_0.1s.csv | al2024 | tensile | fig2a | 250 | 0.1 |
| al2024_fig8a_250C_0.1s_exp.csv | al2024 | tensile | fig8a | 250 | 0.1 |
| al2024_fig2b_250C_1.0s.csv | al2024 | tensile | fig2b | 250 | 1 |
| al2024_fig8b_250C_1.0s_exp.csv | al2024 | tensile | fig8b | 250 | 1 |
| al2024_fig2c_250C_10s.csv | al2024 | tensile | fig2c | 250 | 10 |
| al2024_fig8c_250C_10s_exp.csv | al2024 | tensile | fig8c | 250 | 10 |
| al2024_fig2a_300C_0.1s.csv | al2024 | tensile | fig2a | 300 | 0.1 |
| al2024_fig8a_300C_0.1s_exp.csv | al2024 | tensile | fig8a | 300 | 0.1 |
| al2024_fig2b_300C_1.0s.csv | al2024 | tensile | fig2b | 300 | 1 |
| al2024_fig8b_300C_1.0s_exp.csv | al2024 | tensile | fig8b | 300 | 1 |
| al2024_fig2c_300C_10s.csv | al2024 | tensile | fig2c | 300 | 10 |
| al2024_fig8c_300C_10s_exp.csv | al2024 | tensile | fig8c | 300 | 10 |
| al2024_fig2a_350C_0.1s.csv | al2024 | tensile | fig2a | 350 | 0.1 |
| al2024_fig8a_350C_0.1s_exp.csv | al2024 | tensile | fig8a | 350 | 0.1 |
| al2024_fig2b_350C_1.0s.csv | al2024 | tensile | fig2b | 350 | 1 |
| al2024_fig8b_350C_1.0s_exp.csv | al2024 | tensile | fig8b | 350 | 1 |
| al2024_fig2c_350C_10s.csv | al2024 | tensile | fig2c | 350 | 10 |
| al2024_fig8c_350C_10s_exp.csv | al2024 | tensile | fig8c | 350 | 10 |

### al2219
| 文件 | material | test_type | series | temperature_C | strain_rate |
|---|---|---|---|---:|---:|
| al2219_fig3a_300C_0.1s.csv | al2219 | tensile | fig3a | 300 | 0.1 |
| al2219_fig3b_300C_1.0s.csv | al2219 | tensile | fig3b | 300 | 1 |
| al2219_fig3d_300C_5.0s.csv | al2219 | tensile | fig3d | 300 | 5 |
| al2219_fig3f_300C_10s.csv | al2219 | tensile | fig3f | 300 | 10 |
| al2219_fig3a_350C_0.1s.csv | al2219 | tensile | fig3a | 350 | 0.1 |
| al2219_fig3b_350C_1.0s.csv | al2219 | tensile | fig3b | 350 | 1 |
| al2219_fig3d_350C_5.0s.csv | al2219 | tensile | fig3d | 350 | 5 |
| al2219_fig3f_350C_10s.csv | al2219 | tensile | fig3f | 350 | 10 |
| al2219_fig3a_400C_0.1s.csv | al2219 | tensile | fig3a | 400 | 0.1 |
| al2219_fig3b_400C_1.0s.csv | al2219 | tensile | fig3b | 400 | 1 |
| al2219_fig3d_400C_5.0s.csv | al2219 | tensile | fig3d | 400 | 5 |
| al2219_fig3f_400C_10s.csv | al2219 | tensile | fig3f | 400 | 10 |
| al2219_fig3a_450C_0.1s.csv | al2219 | tensile | fig3a | 450 | 0.1 |
| al2219_fig3b_450C_1.0s.csv | al2219 | tensile | fig3b | 450 | 1 |
| al2219_fig3d_450C_5.0s.csv | al2219 | tensile | fig3d | 450 | 5 |
| al2219_fig3f_450C_10s.csv | al2219 | tensile | fig3f | 450 | 10 |
| al2219_fig3a_480C_0.1s.csv | al2219 | tensile | fig3a | 480 | 0.1 |
| al2219_fig3b_480C_1.0s.csv | al2219 | tensile | fig3b | 480 | 1 |
| al2219_fig3d_480C_5.0s.csv | al2219 | tensile | fig3d | 480 | 5 |
| al2219_fig3f_480C_10s.csv | al2219 | tensile | fig3f | 480 | 10 |

### al6061
| 文件 | material | test_type | series | temperature_C | strain_rate |
|---|---|---|---|---:|---:|
| al6061_plane_strain_020c.csv | al6061 | plane_strain | - | 20 | 0.001 |
| al6061_tensile_020c.csv | al6061 | tensile | - | 20 | 0.001 |
| al6061_tensile_100c.csv | al6061 | tensile | - | 100 | 0.001 |
| al6061_plane_strain_150c.csv | al6061 | plane_strain | - | 150 | 0.001 |
| al6061_tensile_150c.csv | al6061 | tensile | - | 150 | 0.001 |
| al6061_plane_strain_200c.csv | al6061 | plane_strain | - | 200 | 0.001 |
| al6061_tensile_200c.csv | al6061 | tensile | - | 200 | 0.001 |
| al6061_plane_strain_250c.csv | al6061 | plane_strain | - | 250 | 0.001 |
| al6061_tensile_250c.csv | al6061 | tensile | - | 250 | 0.001 |
| al6061_plane_strain_300c.csv | al6061 | plane_strain | - | 300 | 0.001 |
| al6061_tensile_300c.csv | al6061 | tensile | - | 300 | 0.001 |

### al7075
| 文件 | material | test_type | series | temperature_C | strain_rate |
|---|---|---|---|---:|---:|
| al7075_fig1a_573K_0.01.csv | al7075 | tensile | fig1a | 300 | 0.01 |
| al7075_fig1a_573K_0.1.csv | al7075 | tensile | fig1a | 300 | 0.1 |
| al7075_fig1a_573K_1.csv | al7075 | tensile | fig1a | 300 | 1 |
| al7075_fig1a_573K_10.csv | al7075 | tensile | fig1a | 300 | 10 |
| al7075_fig1a_623K_0.01.csv | al7075 | tensile | fig1a | 350 | 0.01 |
| al7075_fig1a_623K_0.1.csv | al7075 | tensile | fig1a | 350 | 0.1 |
| al7075_fig1a_623K_1.csv | al7075 | tensile | fig1a | 350 | 1 |
| al7075_fig1a_623K_10.csv | al7075 | tensile | fig1a | 350 | 10 |
| al7075_fig1a_673K_0.01.csv | al7075 | tensile | fig1a | 400 | 0.01 |
| al7075_fig1a_673K_0.1.csv | al7075 | tensile | fig1a | 400 | 0.1 |
| al7075_fig1a_673K_1.csv | al7075 | tensile | fig1a | 400 | 1 |
| al7075_fig1a_673K_10.csv | al7075 | tensile | fig1a | 400 | 10 |
| al7075_fig1a_723K_0.01.csv | al7075 | tensile | fig1a | 450 | 0.01 |
| al7075_fig1a_723K_0.1.csv | al7075 | tensile | fig1a | 450 | 0.1 |
| al7075_fig1a_723K_1.csv | al7075 | tensile | fig1a | 450 | 1 |
| al7075_fig1a_723K_10.csv | al7075 | tensile | fig1a | 450 | 10 |

## 备注
- 文件名 `al2024_fig8a_200C_0.1s_exp.csv.csv` 在解析时会自动归一化为 `.csv` 后缀。
- `al6061_mendeley_stats.txt` 与 `al6061_mendeley_averaged_stats.txt` 为统计文件，不参与 CSV 参数解析。
- `data/processed/筛选数据_merged.csv` 中 `is_converted=True` 表示该曲线已处于真实应力应变格式（原始即为真实或已完成转换）。
