# CIFAR-10 联合主实验 seed 7 审计报告

审计对象：`joint_precommit_cifar10_seed7_20261015`。本报告为只读复核结果，未修改原始实验文件。

## 结论

实验已完成 60 轮，配置与场景文件一致，最终准确率为 46.07%。第 20、40、59 轮的篡改候选均在提交前被拒绝；这三轮没有提交新模型或新状态，不能表述为“下一轮发现后回退”。掉线恢复在实际尝试中全部成功，但恢复尝试数为 220，而不是完整计划的 240：三个篡改轮跳过了对应的恢复/筛选流程。

总体审计结论：**PASS WITH LIMITATIONS**。

## 已核对事项

| 项目 | 结果 | 证据 |
|---|---|---|
| 训练完整性 | 通过 | `results.csv` 含 round 0–60，共 61 行；训练轮 1–60 连续，loss 均为有限值 |
| 配置一致性 | 通过 | `scenario.json` 与 `verashield_fl/config.json`：CIFAR-10、20 客户端、60 轮、Dirichlet α=0.5、seed 7、20% sign-flip、强度 5、第 4 轮起攻击 |
| 提交前篡改拒绝 | 通过 | `vl_verification_results.csv`：round 20/40/59 均 `tamper_applied=1`、`precommit_verification_success=0`、`candidate_rejected=1`、`model_committed=0`、`state_committed=0` |
| 篡改后状态 | 通过 | 三个篡改轮的 `committed_model_digest` 等于前一已提交状态；准确率在 19→20、39→40、58→59 保持不变，下一轮继续推进 |
| 门限切换 | 通过 | round 1–40 为 A/16；round 41–60 为 B/12；仅 round 41 标记切换 |
| 掉线恢复 | 条件通过 | 20% 阶段每轮 4 次、40% 阶段每轮 8 次；220/220 次尝试成功，0 次失败 |
| 动态权重 | 通过 | 逐轮保留客户端权重和为 1.0；拒绝客户端权重为 0 |
| 训练数值稳定性 | 通过 | 60 个训练轮没有非有限 loss |

## 统计口径

检测统计按 `detection_statistics.csv` 重新计算：

- 全部 1200 个客户端-轮次：TP=228、TN=955、FP=5、FN=12；Recall=95.00%，FPR=0.5208%，Precision=97.8541%。该口径包含攻击未启用的前 3 轮。
- 第 4 轮起的 1140 个可检测样本：TP=228、TN=908、FP=4、FN=0；Recall=100%，FPR=0.4386%，Precision=98.2759%。
- 仅 `attack_active=1` 的 228 个恶意样本：TP=228、FN=0。该子集只能说明恶意样本在本次标签与筛选口径下均被识别，不能单独代表完整检测器性能。

恢复统计来自 1200 行客户端记录。实际 `recovery_attempted=1` 的记录为 220，全部 `recovery_success=1`。三个篡改轮分别跳过 4、8、8 个本应恢复的掉线事件，因此不能写成 240/240。

## 必须保留的限制

1. 这是单个 seed 的描述性结果，不能支持多 seed 均值或显著性结论。
2. 实验使用逻辑双服务器顺序模拟，`independent_server_processes=false` 且没有真实服务器间通信测量；不能据此证明物理双服务器隐私隔离。
3. 掉线是 post-share 的逻辑掉线，实验中每轮仍预先生成更新/份额；它验证协议恢复流程，不等同于真实 UAV 永久离线或无线链路故障。
4. `results.csv` 的参与/回复字段仍显示 20，掉线语义应以 `scenario.json`、`recovery_statistics.csv` 和协议轨迹中的 `unavailable_set` 为准，不能把它解释为真实网络层掉线率。
5. 本次基线或其他方法没有在同一批次中执行，因此不能做公平的同批次方法排名，也不能把效果归因于某一个模块。
6. 自动 `validation_report.json` 的 PASS 只覆盖完成检查和篡改拒绝，不是完整安全证明。

## 推荐论文表述

“在 CIFAR-10、Dirichlet α=0.5、20 个客户端、60 轮和 seed 7 的联合威胁逻辑原型中，VeraShield-FL 完成训练并取得 46.07% 最终准确率。第 20、40、59 轮的篡改候选均在模型提交前被拒绝。预置正常/应急门限为 16/12，实际 220 次掉线恢复尝试全部成功。检测统计按第 4 轮起的可检测样本计算为 Recall 100%、FPR 0.4386%、Precision 98.2759%。这些是单 seed、逻辑双服务器和 post-share 掉线条件下的描述性结果。”

