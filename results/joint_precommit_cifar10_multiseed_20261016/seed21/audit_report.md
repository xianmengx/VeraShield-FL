# CIFAR-10 联合主实验 seed 21 审计

审计日期：2026-10-16（Asia/Shanghai）。模式：numeric-audit / claim-audit。范围：本目录中 VeraShield-FL 的实际结果、注册命令与场景；未修改原始结果、批次状态、论文或训练代码，未启动训练。结论：**PASS WITH LIMITATIONS**。

## 数字与证据

|核验项目|结果|证据|
|---|---|---|
|训练完整性|round 0–60 连续，共 61 行；60 轮 loss 均有限|verashield_fl/results.csv、summary.json、run.log|
|最终准确率|48.07%；最终 loss 1.5950552300561833；三处一致|results.csv、summary.json、上级 run.log|
|最高准确率|48.79%，第 58、59 轮并列；summary 的 best_round=59 是并列之一|results.csv、summary.json|
|实验配置|CIFAR-10，seed 21，20 客户端，60 轮，Dirichlet α=0.5；20% sign-flip、强度 5、第 4 轮起；precommit|config.json、scenario.json、registration.json|
|场景完整性|运行副本与注册场景一致；SHA-256 与记录一致|scenario.json、scenario.sha256、registration.json|
|篡改拦截|20/40/59 轮实际施加并拒绝，模型/状态均不提交；其余 57 轮提交|vl_verification_results.csv|
|拒绝轮模型与状态|模型摘要等于上一已提交摘要；聚合状态前后摘要相同；全轮父模型摘要链连续|vl_verification_results.csv|
|拒绝轮准确率|19→20：36.72%→36.72%；39→40：46.40%→46.40%；58→59：48.79%→48.79%|results.csv|
|Crypto-VL 离线复验|用匹配 setup_id 的本地密钥重新验证全部 60 份证明：57 份通过、3 份仅返回 returned_model_digest_mismatch，无其他不一致|vl_verification_results.csv 中 proof、candidate_model_digest；CryptoVLVerifier|
|客户端认证|986/986 回执签名有效；986/986 参与记录签名有效|client_receipts、participation_records；verify_receipt(seed=21)、verify_record|
|客户端回执语义|918 条无失败原因且 aggregate_vector_check=match；20 条初始轮不适用；48 条 unchanged_cached_model|vl_verification_results.csv|
|门限|正常 16 共 40 轮，应急 12 共 20 轮；第 41 轮切换|runtime_threshold.csv|
|恢复|240/240 次尝试成功，0 次失败；最大绝对恢复误差 7.62939453125e-06|recovery_statistics.csv|
|权重|60 轮实数权重和均为 1；被移除客户端权重为 0；证明整数权重和均为 65536|dynamic_weight_results.csv、vl_verification_results.csv|

场景 SHA-256：`29af9b3a986ea1c5610f00eccc30f0aedc5850ab908ddf6c0f29198b1e9402e7`。

## 投毒检测重新计数

检测记录共 1200 条客户端-轮次，包含前 3 个未攻击轮次。全部记录按恶意身份计数为 TP=214、TN=960、FP=0、FN=26；其中 12 个 FN 来自攻击未启用的前 3 轮，不适合解释为实际攻击漏检。

正式主动攻击口径采用第 4–60 轮、detection_eligible=1，共 1140 条记录：TP=214、TN=912、FP=0、FN=14。Recall=93.8596%，FPR=0%，Precision=100%；与 detection_metrics_active.json 一致。FPR=0 仅为本次有限样本的观察值。

## 与此前统计必须区分的事项

本次 seed 21 在三个篡改轮仍记录了检测和恢复：20% 掉线阶段 20×4=80 次、40% 阶段 20×8=160 次，总计 240 次。不能套用此前 seed 7 报告中的 220 次恢复及“篡改轮跳过检测/恢复”表述。不同 seed 的机制计数应分别从原始记录汇总；在确认执行流程差异前，不将它们直接称为完全相同实现的合并机制统计。

## 中断原因

训练已正常结束并输出 summary。批次完成检查使用 csv 默认单字段上限 131072，读取 proof 字段时报错。本次最大 proof 字段为 778641 字符。审计读取时仅提高当前进程字段上限，成功解析全部记录并完成复验；未发现原始文件截断或轮次缺失。

因此批次 FAILED 是完成检查失败，不能解释为训练失败。当前审计没有改写 status.json，也没有恢复后续运行。

## 主张边界与未核验范围

1. 本次支持 seed 21 的最终效用、实际三次提交前拦截、恢复和检测计数，不支持完整三 seed 均值或显著性结论。
2. 拦截发生在服务器端 precommit 检查；986 条客户端回执全部成功，不能写成“客户端在这三个当轮回执中拒绝了篡改结果”，也不能写成下一轮回退。rollback_applied 在三个拒绝轮均为 0。
3. 离线证明复验以证明内携带的 client_records 和 recovery_digest 为输入，并核对预期轮次、父模型摘要和全体候选集合摘要；这验证内部密码学一致性，不是从独立外部事实重建全部客户端更新或恢复输入。
4. setup_manifest 明确 logical_two_server、单进程逻辑角色、无 secure MPC、无真实服务器间通信测量；本结果不证明物理双服务器隐私隔离或全流程密态安全。
5. 掉线为 post-share 逻辑模拟，不能解释为真实 UAV 无线故障实测；恢复计数是客户端-轮次事件数。
6. registration 保存命令和场景哈希，但未见启动时源代码哈希；无法仅凭这些材料证明运行时全部代码与其他 seed 完全一致。setup_manifest 中旧协议名 VL-SSDF-UAVFL 是元数据旧称，应在汇总映射中注明，不能修改原始记录掩盖它。
7. 本次未重审基线、其他 seed、论文引文或完整安全证明；未编造缺失结果。

## 严重程度与后续事项

- 高：完成检查的 CSV 字段上限导致批次停止，后续 seed 和 MNIST 没有自动启动。修复读取后应复用本次完成结果，避免重复训练。
- 中：seed 7 与 seed 21 的恢复/检测执行口径存在差异，跨 seed 汇总前需明确版本与统计口径。
- 中：缺少启动时源代码哈希，代码版本追溯存在缺口。

本审计无需修改 48.07% 原始数据。后续实验维护应修复完成检查并续跑；完整论文方法评估不在本次数值审计范围内。
