# VeraShield-FL: code and experiment artifacts

本仓库保存 VeraShield-FL 原型代码、选定实验结果及其图表来源记录。保留原实验目录结构，便于逐项追溯图表数据。

## 实现范围

Flower/PyTorch逻辑双服务器联邦学习原型，包含PoisonGuard、Dynamic Weighting、预置16/12层的Dynamic Recovery和Crypto-VL聚合结果验证。两个服务器角色在逻辑边界中协同计算；检测中的中位数、范数、余弦等使用普通数值运算，未实现完整恶意安全MPC后端。本包不提供真实UAV硬件、无线wire流量、能耗或物理双服务器隔离的实测结论。

单独篡改实验使用提交前验证主方法结果，早于后续客户端精确分子修复；联合主实验使用后续修正结果。Crypto-VL验证认证记录与聚合关系的一致性，不证明投毒检测的统计正确性。

## 包含哪些实验

| 实验 | 选定运行 | 数据与范围 |
|---|---:|---|
| 联合主实验 | 30 | CIFAR-10/MNIST × 五方法 × seed 7/21/42 |
| 单独客户端投毒 | 135 | IID20%、Non-IID20%、IID40% × 三攻击 × 五方法 × 三seed |
| 单独服务器篡改 | 15 | CIFAR-10 IID × 五方法 × 三seed |
| 恢复、开销及修复审计 | 另列 | 保留恢复门限边界、分阶段掉线、客户端微基准及必要审计 |

180项是六张主要准确率图的来源数量，并非本包全部支持实验数量。四种基线与部分攻击场景复用历史运行；IID 20% scaling 的 VeraShield-FL 三个 seed 已更新为当前 `unified_v2_candidate`/`precommit` 配置，其余 IID 20% 攻击运行保持原选定结果。当前代码不是所有历史数据的原始源码快照。原registration中的source_sha256属于运行时记录，不应误认为发布代码哈希。

## 六张主要图

| 图 | PNG预览 | 配套数据 |
|---|---|---|
| CIFAR-10联合实验 | [图1](figures/joint/cifar10_joint_accuracy_3seed_mean.png) | figures/joint/ |
| MNIST联合实验 | [图2](figures/joint/mnist_joint_accuracy_3seed_mean.png) | figures/joint/ |
| IID、20%投毒 | [图3](figures/client_attacks/iid20_three_attacks_accuracy.png) | figures/client_attacks/ |
| Non-IID、20%投毒 | [图4](figures/client_attacks/noniid20_three_attacks_accuracy.png) | figures/client_attacks/ |
| IID、40%投毒压力 | [图5](figures/client_attacks/iid40_three_attacks_accuracy.png) | figures/client_attacks/ |
| 五方法服务器篡改 | [图6](figures/server_tamper/cifar10_server_tamper_5methods_3seed_mean.png) | figures/server_tamper/ |

另保留服务器篡改事件辅助图。主要图同时提供PDF/SVG，曲线为三seed逐轮均值、无平滑和阴影；逐seed值、标准差、来源与图注见各图包。恢复以统计表为主。

## 验证已有数据与重新绘图

Python 3.11环境下，数据完整性验证仅需标准库：

```sh
python scripts/verify_release.py
python -m pip install -r requirements-figures.txt
python scripts/plot_joint_accuracy_paper.py
python scripts/audit_plot_client_attacks_paper.py --export
python scripts/plot_server_tamper_comparison_paper.py
```

绘图默认刷新figures中的导出文件；可用环境变量VERASHIELD_FIGURE_ROOT指定另一个输出根目录。绘图采用项目内release_visual_qa，不依赖桌面应用技能目录。该本地工具检查有限坐标和可见文字裁切，不能替代人工审图。

## 训练环境与配置复用

正式运行config.json记录Python 3.11.9、Flower 1.31.0、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121等。environment/original-wsl-gpu-snapshot.txt是保留的WSL GPU环境快照，不能直接作为跨平台安装锁文件。发布pyproject将明显不符的torch2.10/vision0.25声明改为2.5.1/0.20.1；具体CUDA安装按PyTorch官方对应版本说明选择。本机打包验证不等于已在所有平台重新训练。

```sh
python -m venv .venv
# 激活.venv后执行；GPU版本按平台先安装匹配的torch/torchvision
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python scripts/init_crypto_vl_keys.py --key-dir local_keys --clients 20
```

密钥必须本地新建，原实验私钥不发布。预先缓存Hugging Face的uoft-cs/cifar10与ylecun/mnist数据集；数据集缓存不包含在本包中。已有批次运行器保留用于配置追溯，建议使用下面的安全入口复用记录的完整run_config，并把新结果放入reproductions，避免覆盖原结果：

```sh
python scripts/replay_run.py --run results/joint_precommit_cifar10_seed7_20261015/verashield_fl
# 确认计划、环境、数据缓存与新密钥后再执行训练
python scripts/replay_run.py --run results/joint_precommit_cifar10_seed7_20261015/verashield_fl --execute
```

默认只打印计划，不训练、不改记录。执行模式使用20客户端及记录配置，CPU逻辑模拟；不能据此复现原GPU墙钟时间。重新生成密钥、框架/平台差异及当前修复后的源码意味着结果不保证逐位等于历史运行。

## 数值及来源说明

- 联合主实验：20客户端、60轮、Dirichlet α=0.5、20% Sign-flip，第20/40/59轮篡改，分阶段0/20/40%共享后掉线，门限16/12。
- Non-IID单独投毒使用完整九项新版主方法；IID场景检测策略不同，不能把跨图差异全部归因于划分。
- Krum在MNIST seed42达到94.65%；IID scaling下VeraShield-FL均值为50.00%，Trimmed Mean为49.87%，逐seed结果见 `results/final_v2/ATTACK20_IID_FINAL_ANALYSIS.md`。
- 单独篡改45个预设事件只有44个实际执行，原因见图包审计；曲线不是完全相同实际暴露下的检测率排名。
- 掉线对照还关闭检测与加权，不是恢复单模块消融。
- 开销数据属于逻辑原型；通信为逻辑载荷，部分方法成功工作量不同，不作等工作量的速度排名或客户端通信轻量结论。

发布时对机器路径进行了去标识，未改180份results.csv的原始字节。manifests/source_to_release.csv分别记录原文件及发布副本哈希；figures中source_manifest按发布副本更新。旧审计报告中的历史配置哈希、签名或源码哈希仍属于原运行，不应用来验证经过路径转换的元数据。原始来源身份另见manifests/original_figure_hashes.csv。

## GitHub资产

大型 `vl_verification_results.csv` 使用Git LFS管理；新增或替换此类文件前先运行 `git lfs install`。历史归档、私钥、虚拟环境、数据集缓存及论文草稿不属于发布内容。实验结果完整性与来源核对记录见 `manifests/VALIDATION.md`。

LICENSE保留原项目Apache-2.0许可文本；相关文献仅作为方法背景，不包含Li或zkFL的第三方实现。
