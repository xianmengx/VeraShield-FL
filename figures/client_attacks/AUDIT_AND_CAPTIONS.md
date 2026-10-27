# 三类客户端攻击：核对结果与图表说明

核对通过：135/135 项，三种场景 × 三种攻击 × 五种方法 × 三个 seed；每项 60 轮，另含初始化第 0 轮。
数据集 CIFAR-10，20 客户端，全部参与，攻击从第 4 轮开始且持续；符号翻转强度 5，随机梯度替换和幅值缩放强度 10。无掉线、无服务器篡改。

## 版本与排除规则

- Non-IID 20% 主方法采用全部九项 unified_v2_candidate 结果，替代旧的单项替换混合版本。四种基线复用对应 Non-IID 结果。
- IID 20% 的 scaling 攻击下，VeraShield-FL 三个 seed 使用 unified_v2_candidate 与 precommit 配置；其 sign-flip、random-gradient 及四种基线仍采用原选定运行。IID 40% 保留原选定配置。
- IID 20% scaling 的 VeraShield-FL 结果更新后，三个攻击场景中的方法配置并非完全相同；图表按攻击场景分别汇报。
- Non-IID 40% 仅有已知失败的 seed 7 诊断批次，不作为正式压力对照，不补数、不合并。
- 修正版策略曾参考已观察到的失败案例；这些结果不属于未见数据上的独立验证。
- IID 40% Krum f=4，20% 场景 f=2；Trimmed Mean 的总裁剪比例均为 0.2。压力测试的基线参数变化需在正文说明。
- 场景清单划分字段与实际 config 不一致条目：3；具体条目见 configuration_audit.csv。实际执行划分以 config 及已存审计为依据。

## 最终准确率（均值 ± 样本标准差，%）

| 场景 | 攻击 | 方法 | seed 7 | seed 21 | seed 42 | 均值 ± SD |
|---|---|---|---:|---:|---:|---:|
| iid20 | sign_flip | VeraShield-FL | 50.30 | 49.04 | 50.10 | 49.81 ± 0.68 |
| iid20 | sign_flip | FedAvg | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid20 | sign_flip | Krum | 34.14 | 35.94 | 34.51 | 34.86 ± 0.95 |
| iid20 | sign_flip | Trimmed Mean | 10.02 | 10.00 | 10.00 | 10.01 ± 0.01 |
| iid20 | sign_flip | FedDMC | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid20 | random_gradient | VeraShield-FL | 50.01 | 50.61 | 50.71 | 50.44 ± 0.38 |
| iid20 | random_gradient | FedAvg | 36.49 | 38.37 | 36.83 | 37.23 ± 1.00 |
| iid20 | random_gradient | Krum | 37.81 | 38.61 | 39.66 | 38.69 ± 0.93 |
| iid20 | random_gradient | Trimmed Mean | 42.67 | 43.47 | 42.15 | 42.76 ± 0.66 |
| iid20 | random_gradient | FedDMC | 50.08 | 47.33 | 48.24 | 48.55 ± 1.40 |
| iid20 | scaling | VeraShield-FL | 49.45 | 51.00 | 49.54 | 50.00 ± 0.87 |
| iid20 | scaling | FedAvg | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid20 | scaling | Krum | 44.84 | 41.31 | 43.92 | 43.36 ± 1.83 |
| iid20 | scaling | Trimmed Mean | 48.41 | 50.11 | 51.08 | 49.87 ± 1.35 |
| iid20 | scaling | FedDMC | 10.00 | 27.45 | 10.00 | 15.82 ± 10.07 |
| noniid20 | sign_flip | VeraShield-FL | 45.13 | 47.93 | 47.98 | 47.01 ± 1.63 |
| noniid20 | sign_flip | FedAvg | 10.00 | 10.00 | 8.69 | 9.56 ± 0.76 |
| noniid20 | sign_flip | Krum | 26.58 | 23.88 | 23.18 | 24.55 ± 1.80 |
| noniid20 | sign_flip | Trimmed Mean | 10.00 | 11.15 | 10.00 | 10.38 ± 0.66 |
| noniid20 | sign_flip | FedDMC | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| noniid20 | random_gradient | VeraShield-FL | 46.04 | 48.65 | 49.18 | 47.96 ± 1.68 |
| noniid20 | random_gradient | FedAvg | 36.43 | 23.57 | 29.43 | 29.81 ± 6.44 |
| noniid20 | random_gradient | Krum | 28.22 | 22.74 | 23.16 | 24.71 ± 3.05 |
| noniid20 | random_gradient | Trimmed Mean | 39.17 | 37.70 | 39.10 | 38.66 ± 0.83 |
| noniid20 | random_gradient | FedDMC | 44.92 | 40.71 | 46.96 | 44.20 ± 3.19 |
| noniid20 | scaling | VeraShield-FL | 47.22 | 48.04 | 48.15 | 47.80 ± 0.51 |
| noniid20 | scaling | FedAvg | 10.02 | 10.00 | 10.00 | 10.01 ± 0.01 |
| noniid20 | scaling | Krum | 28.68 | 21.97 | 10.00 | 20.22 ± 9.46 |
| noniid20 | scaling | Trimmed Mean | 46.30 | 47.12 | 44.65 | 46.02 ± 1.26 |
| noniid20 | scaling | FedDMC | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid40 | sign_flip | VeraShield-FL | 47.54 | 49.97 | 46.31 | 47.94 ± 1.86 |
| iid40 | sign_flip | FedAvg | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid40 | sign_flip | Krum | 31.09 | 32.76 | 30.93 | 31.59 ± 1.01 |
| iid40 | sign_flip | Trimmed Mean | 10.00 | 9.96 | 10.02 | 9.99 ± 0.03 |
| iid40 | sign_flip | FedDMC | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid40 | random_gradient | VeraShield-FL | 49.75 | 48.50 | 45.00 | 47.75 ± 2.46 |
| iid40 | random_gradient | FedAvg | 9.81 | 10.00 | 23.77 | 14.53 ± 8.01 |
| iid40 | random_gradient | Krum | 39.01 | 33.95 | 35.32 | 36.09 ± 2.62 |
| iid40 | random_gradient | Trimmed Mean | 24.08 | 24.51 | 25.91 | 24.83 ± 0.96 |
| iid40 | random_gradient | FedDMC | 35.43 | 37.12 | 10.00 | 27.52 ± 15.19 |
| iid40 | scaling | VeraShield-FL | 48.85 | 47.57 | 46.41 | 47.61 ± 1.22 |
| iid40 | scaling | FedAvg | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid40 | scaling | Krum | 44.38 | 43.82 | 42.70 | 43.63 ± 0.86 |
| iid40 | scaling | Trimmed Mean | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |
| iid40 | scaling | FedDMC | 10.00 | 10.00 | 10.00 | 10.00 ± 0.00 |

## PoisonGuard 检测指标（统一第 4–60 轮）

从 detection_statistics.csv 重新计算，保证每 seed 分母为 1140 个客户端轮次。历史 detection_metrics_active.json 部分包含前 3 轮，不能直接与新版指标混合。本表先逐 seed 计算比率，再报告三 seed 均值 ± 样本标准差。

| 场景 | 攻击 | Recall (%) | FPR (%) | Precision (%) | 验证失败记录数 |
|---|---|---:|---:|---:|---:|
| iid20 | sign_flip | 100.00 ± 0.00 | 0.51 ± 0.79 | 98.06 ± 3.00 | 0 |
| iid20 | random_gradient | 100.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | 0 |
| iid20 | scaling | 100.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | 0 |
| noniid20 | sign_flip | 97.95 ± 3.55 | 1.61 ± 2.42 | 94.45 ± 8.17 | 0 |
| noniid20 | random_gradient | 100.00 ± 0.00 | 0.51 ± 0.62 | 98.03 ± 2.37 | 0 |
| noniid20 | scaling | 100.00 ± 0.00 | 0.33 ± 0.57 | 98.73 ± 2.19 | 0 |
| iid40 | sign_flip | 100.00 ± 0.00 | 0.73 ± 0.73 | 98.92 ± 1.07 | 0 |
| iid40 | random_gradient | 99.42 ± 1.01 | 0.29 ± 0.29 | 99.56 ± 0.44 | 0 |
| iid40 | scaling | 100.00 ± 0.00 | 0.10 ± 0.08 | 99.85 ± 0.13 | 0 |

## 论文图注

CIFAR-10 在三种客户端投毒攻击下的测试准确率。子图 (a)–(c) 分别为符号翻转、随机梯度替换和幅值缩放。每条曲线表示 seed 7、21、42 的逐轮均值，无平滑、无标准差阴影；最终准确率的逐 seed 数值与样本标准差见配套表。各图分别对应 IID 20%、Non-IID 20%（Dirichlet α=0.5）与 IID 40% 恶意客户端。攻击始于第 4 轮，无掉线与服务器篡改。

三张图反映完整方法的抗投毒效用，检测能力由上述 Recall/FPR/Precision 表支撑；未做模块消融时，不能将准确率优势全部归因于 PoisonGuard。IID scaling 下 Trimmed Mean 的最终均值可能略高，禁止声称所有攻击均绝对最优。

## 文件

PDF/SVG 为论文用矢量文件；PNG 为 600 dpi 预览；灰度图用于检查线型区分。source_manifest.csv 保存所有输入路径与 SHA-256。原始结果保持不变。
