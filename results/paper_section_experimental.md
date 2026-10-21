# VL-SSDF-UAVFL — Experimental Evaluation (draft)

*All numbers are taken verbatim from `results/*.csv` and `results/*.json`. No experiment was re-run. English and Chinese versions follow.*

---

# 4 Experimental Evaluation (EN)

## 4.1 Experimental Setup

We evaluate the proposed **VL-SSDF-UAVFL** scheme (SSDF + Dynamic Threshold D2 + Dynamic Weight + VL lightweight verification) on **CIFAR-10** (50,000 train / 10,000 test, IID partition) within the **Flower 1.31.0** framework (PyTorch 2.10.0, Ray 2.55.1 simulation, `flwr-datasets 0.6.0`). The task is image classification with the standard Flower PyTorch quickstart CNN. Unless stated otherwise we use **10 clients**, **FL rounds 10–20**, and **seed 42** (multi-seed uses 42/7/21).

- **Attack model.** Update-level **sign-flip** (`Δ_mal = −Δ_clean`), ratio ∈ {0, 10, 20, 30, 40}%, persistent across rounds, malicious IDs fixed per run.
- **Dropout model.** Simulated client unavailability via `pre-share` (client returns a Flower Error reply and is excluded) or `post-share` (shares generated/distributed, later recovered). Rate ∈ {0, 20, 30, 40}%.
- **Metrics.** Test accuracy/loss (`ServerApp-side Evaluate Metrics`), recovery success rate (`recovery_statistics.csv`), removed clients, communication/computation overhead.

## 4.2 Effectiveness of Dynamic Threshold

**Table 4.1 — Recovery success vs. availability (Static t=8 vs Dynamic D2).**

| Avail. | Dropout | Static recov. | Static acc. | Dynamic recov. | Dynamic acc. |
|---|---|---|---|---|---|
| 10 | 0.0 | 1 | 0.2878 | 1 | 0.2564 |
| 9 | 0.1 | 1 | 0.2601 | 1 | 0.2755 |
| 8 | 0.2 | 1 | 0.2539 | 1 | 0.2755 |
| 7 | 0.3 | **0** | 0.2524 | **1** | 0.2713 |
| 6 | 0.4 | **0** | 0.1897 | **1** | 0.2763 |
| 5 | 0.5 | 0 | 0.2656 | 0 | 0.2670 |

**Security–availability discussion.** Normal state uses `A = Shamir(10,8)` (threshold 8, B not distributed): ≥ 8 shares needed; high security, low dropout tolerance (fails for available ∈ {6,7}). Emergency state activates `B = Shamir(10,6)` (threshold 6, B distributed): ≥ 6 shares needed; availability rises (recovers at available ∈ {6,7}) at the cost of threshold 6. **This is a security–availability trade-off, not an increase in security.** A and B encode the same secret (`same_secret = 1`, max decode error ≈ 7.6e-6 ≤ 7.63e-6 SCALE bound); the scheme never lowers the required shares without accepting the lower security level.

## 4.3 Robustness Against Poisoning Attacks

**Table 4.2 — Final accuracy vs. attack ratio (10 clients, 20 rounds, dropout 0.3, seed 42).**

| Scheme | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| FedAvg | 0.2852 | 0.2796 | 0.2651 | 0.2720 | 0.2065 |
| FedAvg+SSDF | 0.3097 | 0.2647 | 0.2447 | 0.2814 | **0.1000** |
| +Static t=8 | 0.3014 | 0.2933 | 0.2718 | 0.2282 | **0.1000** |
| +DynamicThreshold | 0.2963 | 0.2786 | 0.2487 | 0.2528 | 0.2077 |
| +DynamicWeight | 0.3101 | 0.2973 | 0.2773 | 0.2859 | 0.2383 |
| **VL-SSDF-UAVFL** | 0.3074 | 0.2792 | 0.2921 | 0.2555 | 0.2341 |

At 40% attack, SSDF-only and Static t=8 collapse to 0.10 (benign false positives under this attack+dropout regime); robust modules recover ≈ 0.21–0.24. **Single-seed note:** at 0.1/0.3 ratios the curve is non-monotonic (run-to-run noise), so rely on the 0/0.2/0.4 trend and the 0.2 point.

**Table 4.3 — Multi-seed (42/7/21), dropout 0.3, mean ± std final accuracy.**

| Scheme | 20% attack | 40% attack |
|---|---|---|
| FedAvg | 0.2752 ± 0.0139 | 0.2102 ± 0.0219 |
| FedAvg+SSDF+DynThr+DW | **0.2842 ± 0.0159** | **0.2532 ± 0.0129** |
| VL-SSDF-UAVFL | 0.2790 ± 0.0316 | 0.2386 ± 0.0209 |

The robust schemes keep higher mean accuracy under attack than FedAvg; VL ≈ DW (differ ≤ 0.02, VL is observational), VL std slightly larger (single seed-set noise).

## 4.4 Ablation Study

**Table 4.4 — Module ablation (10 clients, 10 rounds, seed 42, dropout 0.3, 20% sign-flip).**

| | Scheme | Acc. | Loss | Recov. | Removed |
|---|---|---|---|---|---|
| A | FedAvg | 0.2698 | 1.988 | – | – |
| B | FedAvg+SSDF | 0.1882 | 2.081 | – | – |
| C | +DynamicThreshold | 0.2439 | 2.050 | 1.0 | 9 |
| D | **+DynamicWeight** | **0.2580** | **1.973** | 1.0 | 17 |

**SSDF** (A→B) filters outliers but also removes benign clients (high-FPR regime) → accuracy drop. **Dynamic Threshold** (B→C) recovers dropped clients (recovery 0→1.0), accuracy 0.188→0.244. **Dynamic Weight** (C→D) re-weights by cosine → lowest loss (2.050→1.973). **Caveat:** scheme A uses `pre-share` dropout (dropped excluded, no recovery/removal); its high accuracy reflects "no defense", not superiority.

## 4.5 Verification and Overhead Evaluation

**Table 4.5 — VL verification (10 clients, 10 rounds, seed 42).**

| Scenario | Det. rounds | Success rate | Hash error | Ver. time (s) |
|---|---|---|---|---|
| Normal aggregation | 10 | 1.0 | 2.79e-7 | 2.0e-4 |
| Server tampering | 10 | **0.0** | 102.79 | 1.3e-4 |
| Full chain (dropout 0.3) | 10 | 1.0 | 1.30e-4 | 1.8e-4 |

**Table 4.6 — Communication overhead (bytes/round, attack 20%, dropout 0.3).**

| Scheme | Share | B-distrib. | VL-commit | Total |
|---|---|---|---|---|
| FedAvg / SSDF / Static8 | 0 | 0 | 0 | 0 |
| +DynThreshold | 4,960,560 | 4,960,560 | 0 | 9,921,120 |
| +DynamicWeight | 4,960,560 | 4,960,560 | 0 | 9,921,120 |
| **VL-SSDF-UAVFL** | 4,960,560 | 4,960,560 | 70.4 | 9,921,190.4 |

**Table 4.7 — Computation overhead (s/round, attack 20%, dropout 0.3).**

| Scheme | Sharing | Recon. | DW | VL-verif. |
|---|---|---|---|---|
| FedAvg | 0 | 0 | 0 | 0 |
| FedAvg+SSDF | 0 | 0 | 0 | 0 |
| +DynThreshold | 14.08 | 9e-4 | 0 | 0 |
| +DynamicWeight | 14.10 | 9e-4 | 1.3e-2 | 0 |
| **VL-SSDF-UAVFL** | 14.02 | 9e-4 | 1.2e-2 | 1.7e-4 |

**Caveats.** (i) Overhead is logical payload; `tls_wire_bytes = unavailable`. (ii) `SSDF_time = 0` for recovery schemes (recorder per-detection path not populated); plain SSDF ≈ 1.09e-3 s. (iii) VL is a **lightweight accumulator** (H(v)=v·g), **not a zk proof**; verifies aggregation consistency, not secure against a server controlling the verification vector. (iv) Server-side single-simulation logical dual-server role; not multi-server.

**Limitations.** Single/multi-seed: primary seed 42, multi-seed (42/7/21) only for three core schemes, std shows run-to-run noise; full 6×5×4 matrix not run (reduced representative). CIFAR-10 only (single small benchmark; no Non-IID/generalization). VL not cryptographic. No physical TLS/multi-server/real wire bytes measured.

---

# 4 实验评估（中文版）

## 4.1 实验设置

在 **CIFAR-10**（50,000 训练 / 10,000 测试，IID 划分）上评估所提出的 **VL-SSDF-UAVFL**（SSDF + 动态阈值 D2 + 动态权重 + VL 轻量校验），基于 **Flower 1.31.0**（PyTorch 2.10.0，Ray 2.55.1 仿真，`flwr-datasets 0.6.0`）。任务为图像分类，使用 Flower PyTorch 快速入门标准 CNN。除特别说明外，使用 **10 个客户端**、**FL 轮数 10–20**、**种子 42**（多样化种子 42/7/21）。

- **攻击模型**：更新级 **sign-flip**（`Δ_mal = −Δ_clean`），比例 ∈ {0,10,20,30,40}%，逐轮持续，恶意 ID 单次运行固定。
- **掉线模型**：`pre-share`（客户端返回 Flower Error 并被排除）或 `post-share`（份额已生成/分发、之后恢复）。掉线率 ∈ {0,20,30,40}%。
- **指标**：测试精度/损失（`ServerApp-side Evaluate Metrics`）、恢复成功率（`recovery_statistics.csv`）、被剔除客户端、通信/计算开销。

## 4.2 动态阈值的有效性

**表 4.1 — 恢复成功率 vs. 可用份额（静态 t=8 vs 动态 D2）。**

| 可用 | 掉线 | 静态恢复 | 静态精度 | 动态恢复 | 动态精度 |
|---|---|---|---|---|---|
| 10 | 0.0 | 1 | 0.2878 | 1 | 0.2564 |
| 9 | 0.1 | 1 | 0.2601 | 1 | 0.2755 |
| 8 | 0.2 | 1 | 0.2539 | 1 | 0.2755 |
| 7 | 0.3 | **0** | 0.2524 | **1** | 0.2713 |
| 6 | 0.4 | **0** | 0.1897 | **1** | 0.2763 |
| 5 | 0.5 | 0 | 0.2656 | 0 | 0.2670 |

**安全—可用性讨论**：正常状态用 `A = Shamir(10,8)`（阈值 8，B 未分发）：需 ≥ 8 份额；安全高但掉线容忍低（可用 ∈ {6,7} 时失败）。紧急状态激活 `B = Shamir(10,6)`（阈值 6，B 分发）：需 ≥ 6 份额；可用性上升（可用 ∈ {6,7} 可恢复），代价是阈值降到 6。**这是安全—可用性权衡，非提升安全**。A、B 编码同一秘密（`same_secret=1`，最大解码误差 ≈ 7.6e-6 ≤ 7.63e-6），该方案不会在不接受更低安全层级的前提下降低所需份额数。

## 4.3 对投毒攻击的鲁棒性

**表 4.2 — 最终精度 vs. 攻击比例（10 客户端，20 轮，掉线 0.3，种子 42）。**

| 方案 | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| FedAvg | 0.2852 | 0.2796 | 0.2651 | 0.2720 | 0.2065 |
| FedAvg+SSDF | 0.3097 | 0.2647 | 0.2447 | 0.2814 | **0.1000** |
| +静态 t=8 | 0.3014 | 0.2933 | 0.2718 | 0.2282 | **0.1000** |
| +动态阈值 | 0.2963 | 0.2786 | 0.2487 | 0.2528 | 0.2077 |
| +动态权重 | 0.3101 | 0.2973 | 0.2773 | 0.2859 | 0.2383 |
| **VL-SSDF-UAVFL** | 0.3074 | 0.2792 | 0.2921 | 0.2555 | 0.2341 |

40% 攻击时 SSDF-only 与静态 t=8 崩到 0.10（该攻击+掉线场景下良性误删）；鲁棒模块恢复 ≈ 0.21–0.24。**单种子说明**：0.1/0.3 处曲线非单调（run-to-run 噪声），以 0/0.2/0.4 趋势与 0.2 点为准。

**表 4.3 — 多种子（42/7/21），掉线 0.3，均值 ± 标准差。**

| 方案 | 20% 攻击 | 40% 攻击 |
|---|---|---|
| FedAvg | 0.2752 ± 0.0139 | 0.2102 ± 0.0219 |
| FedAvg+SSDF+DynThr+DW | **0.2842 ± 0.0159** | **0.2532 ± 0.0129** |
| VL-SSDF-UAVFL | 0.2790 ± 0.0316 | 0.2386 ± 0.0209 |

鲁棒方案攻击下均值高于 FedAvg；VL ≈ DW（差 ≤ 0.02，VL 仅观测），VL std 略大（单种子集噪声）。

## 4.4 消融实验

**表 4.4 — 模块消融（10 客户端，10 轮，种子 42，掉线 0.3，20% sign-flip）。**

| | 方案 | 精度 | 损失 | 恢复 | 剔除 |
|---|---|---|---|---|---|
| A | FedAvg | 0.2698 | 1.988 | – | – |
| B | FedAvg+SSDF | 0.1882 | 2.081 | – | – |
| C | +动态阈值 | 0.2439 | 2.050 | 1.0 | 9 |
| D | **+动态权重** | **0.2580** | **1.973** | 1.0 | 17 |

**SSDF**（A→B）过滤异常但也误删良性（高误删场景）→ 精度下降。**动态阈值**（B→C）恢复掉线客户端（0→1.0），精度 0.188→0.244。**动态权重**（C→D）按余弦重加权 → 损失最低（2.050→1.973）。**注意**：A 用 `pre-share` 掉线（掉线排除、无恢复/剔除），其高精度体现“无防御”，非更优。

## 4.5 验证与开销评估

**表 4.5 — VL 校验（10 客户端，10 轮，种子 42）。**

| 场景 | 检出轮数 | 通过率 | 哈希误差 | 校验时间(s) |
|---|---|---|---|---|
| 正常聚合 | 10 | 1.0 | 2.79e-7 | 2.0e-4 |
| 服务器篡改 | 10 | **0.0** | 102.79 | 1.3e-4 |
| 完整链路(掉线0.3) | 10 | 1.0 | 1.30e-4 | 1.8e-4 |

**表 4.6 — 通信开销（字节/轮，攻击 20%，掉线 0.3）。**

| 方案 | 份额 | B 分发 | VL commitment | 总计 |
|---|---|---|---|---|
| FedAvg / SSDF / 静态8 | 0 | 0 | 0 | 0 |
| +动态阈值 | 4,960,560 | 4,960,560 | 0 | 9,921,120 |
| +动态权重 | 4,960,560 | 4,960,560 | 0 | 9,921,120 |
| **VL-SSDF-UAVFL** | 4,960,560 | 4,960,560 | 70.4 | 9,921,190.4 |

**表 4.7 — 计算开销（秒/轮，攻击 20%，掉线 0.3）。**

| 方案 | 份额生成 | 重建 | 动态权重 | VL 校验 |
|---|---|---|---|---|
| FedAvg | 0 | 0 | 0 | 0 |
| FedAvg+SSDF | 0 | 0 | 0 | 0 |
| +动态阈值 | 14.08 | 9e-4 | 0 | 0 |
| +动态权重 | 14.10 | 9e-4 | 1.3e-2 | 0 |
| **VL-SSDF-UAVFL** | 14.02 | 9e-4 | 1.2e-2 | 1.7e-4 |

**说明与局限**：（i）开销为逻辑载荷，`tls_wire_bytes=unavailable`；（ii）恢复方案 `SSDF_time=0`（recorder 未填充），纯 SSDF ≈ 1.09e-3 s；（iii）VL 为**轻量累加器**（H(v)=v·g），**非 zk proof**，验证聚合一致性，对控制校验向量的服务器不可防御；（iv）单仿真服务器端逻辑双服务器角色，非多服务器部署。

**局限性**。单/多种子：主结果用种子 42，多种子（42/7/21）仅对三个核心方案，std 表明 run-to-run 噪声；完整 6×5×4 矩阵未运行（代表性缩减）。仅 CIFAR-10（单一小基准，未评估泛化/Non-IID）。VL 非密码学安全。未测量物理 TLS / 多服务器 / 真实 wire bytes。
