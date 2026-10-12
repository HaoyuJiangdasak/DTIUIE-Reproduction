# 官方代码默认配置完整训练

用户已明确以官方开源代码为准，代码没有说明的再参考论文。
本次目标是从头执行 100 epochs，不加载发布的 epoch 28 DTIUIE 权重。
之前完成的预训练推理属于准备验证，不能作为本次训练完成结果。

**完成更新：**2026-10-03 20:45:46（北京时间）已完成全部 100 epochs、38100 steps，
耗时 8 小时 52 分钟 14 秒，重启次数 0。最终和两套最佳权重均已保存，
三套各 110 张 JPEG/PNG 已导出。结果和指标口径见
`../cloud_training_results/RESULTS.md`，回收进度以本机监控记录为准。

## 已启动的正式运行

- 开始时间：2026-10-03 11:53:30（北京时间）。
- 实例：`connect.westc.seetacloud.com:18824`，RTX 4090 D，24,564 MiB，驱动 595.71.05。
- 项目：`/root/autodl-tmp/DTIUIE`。
- 环境：`/root/autodl-tmp/envs/dtiuie-legacy`，Python 3.9.16，Torch 1.13.1+cu116，torchvision 0.14.1+cu116。
- 正式训练 PID：21263；独立 supervisor PID：21262。恢复后 PID 会变化，以最新状态为准。
- run ID：`463f314c2d0645cfaa3568eae4df0753`。
- run 目录：`training_runs/from_scratch_20261003/`。
- 原作者源码 SHA256：`213bdf088fc32e8294bcb9e7f1c13f5cf1dd5270884926141ccdc2d5405143e0`。

启动设置为 batch=4、100 epochs、256×256、warmup=10、单卡 CUDA。
三网络按作者代码交替更新；E 随机初始化，两个任务编码器使用作者指定的
ImageNet VGG16-BN 初始化，下载文件 SHA256 为
`6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a`。

每轮 trainval 有 381 个 batch（1525 张、drop_last=True），测试验证 110 张。
从头训练没有 `--pretrained`、没有加载发布的 DTIUIE 检查点。
已实际核验第 1 轮完整断点：epoch=1、global_step=381、三模型参数有限、
三优化器/调度器/scaler 和完整随机状态均存在。第 100 轮已保存同样的完整断点。

## 采用的真实官方实现设置

| 项目 | 实际设置 |
|---|---|
| 像素损失 | L1，权重 1 |
| TDP | 第 11 轮起，S 的 C5 特征 L1，权重 1 |
| 混合图 | sigmoid(randn(B,1,8,8))，nearest 放大至 256 |
| E optimizer | AdamW lr=1e-4，betas=(0.9,0.999)，实际 weight_decay=0.01 |
| S/SA optimizer | SGD lr=0.02，momentum=0.9，weight_decay=1e-4 |
| E/S schedule | cosine restart [10,90]；SA 保留原实现的恒定 lr |
| 三项 S 交叉熵 | 各权重实际为 1，求和；CLI .34 未生效 |
| 随机种子 | 原代码种子调用注释，未另行固定 |
| 数值模式 | float32，无 autocast，保留三个 GradScaler |
| 性能 | 开启 cuDNN 自动选核，CPU 线程 8，DataLoader 原默认 workers=8 |

没有改变作者损失、插值、BatchNorm 行为、梯度计算或验证口径。
独立运维副本增加日志心跳、非有限损失停止、完整断点保存与恢复。
AST 对比确认剥除这些运维插入后与作者原训练 AST 相同。
CPU 小模型测试确认恢复后的下一次更新与不中断参考逐张量一致。
原源码保留；生成细节见 PATCH_NOTES.md。

训练集三目录各 1525 张、测试三目录各 110 张已上传且配对完整。
37 张训练掩码的原始尺寸差异在官方 ZIP 中也存在，按官方数据原样保留。
与论文公式及交接文字不一致的实际设置已记录，按用户指定优先官方代码运行。

## 日志、检查点和监控

正式日志：`training_runs/from_scratch_20261003.train.log`。
轻量状态：run 目录中的 `heartbeat.json`、`run_metadata.json`、`supervisor_state.json`。
每个完整 epoch 保存 `latest_full.pth`，包含全部训练/随机状态，可从下一轮恢复。
原作者最佳 SSIM/mIoU 检查点和最终检查点照常保存。
第 1 轮滚动断点约 767 MB，已实际加载核验。

云端 supervisor 每 60 秒检查运行、心跳和磁盘；外部终止且已有断点时最多恢复两次。
非有限损失或代码错误会停下并保留 `failed.json`、日志和最近完整断点，
不通过盲目重启掩盖算法问题。

本机 Windows 任务 `DTIUIE-AutoDL-Monitor-20261003` 每 5 分钟通过 SSH 检查，
已经实际执行成功（LastTaskResult=0）。当前状态可看：
`C:\Users\Lenovo\Desktop\DTIUIE\cloud_training_results\latest_training_status.json`。
检查历史：`C:\Users\Lenovo\autodl-ssh\monitor\`。
每跨过 10 个保存的 epoch 备份一次完整断点；本机睡眠/离线时无法连接，
恢复后继续，云端训练和检查不受本机状态影响。

密码保存于 `C:\Users\Lenovo\autodl-ssh\credentials\autodl-westc-18824.xml`，
使用 Windows 当前用户 DPAPI 加密；私钥继续保留在本机受限权限文件。
密码不写入源码、日志、上传包或任务参数。

## 第 100 轮后的自动处理

独立收尾脚本核验 epoch=100、global_step=38100、三网络参数有限；
对最终/最佳 SSIM/最佳 mIoU 权重分别用 model_SA prior 导出 110 张测试图，
保存 JPEG、PNG 和 manifest，随后打包训练权重、完整断点、TensorBoard 日志和运行记录。
产物：`/root/autodl-tmp/DTIUIE-training-results-20261003.tar.gz` 及 SHA256。

本机监控检测到该包后下载、校验、安全解压到 `cloud_training_results/`，
完成回收后自动禁用该监控任务。
训练内部 SSIM/mIoU 和最后作者 model_S 验证记录会与 model_SA 导出协议分别注明；
这些内部验证不是论文表 II 重新训练独立下游网络的分数。

训练时长根据实际速度估计，100 轮全部执行，不为凑固定小时数增加或缩短训练。
以 `completed.json` 和第 100 轮权重实际存在作为完成证据。
