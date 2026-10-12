# DTIUIE 完整训练结果

本次按用户确认的官方代码默认方案从头训练。代码未说明的再参考论文；
未加载发布的 DTIUIE epoch 28 权重。两个任务网络的 VGG16-BN 编码器按作者代码
使用 ImageNet 初始化。

## 完成证据

- RTX 4090 D，24 GB；Python 3.9.16、Torch 1.13.1+cu116、torchvision 0.14.1+cu116。
- batch=4、100 epochs、256×256，warmup 10，L1 与最近邻混合，单卡 float32。
- 1525 张 trainval，每轮 381 个训练 batch；110 张内部验证图。
- 日志包含第 1–100 轮全部完成记录，总 global step=38100。
- 开始：2026-10-03 11:53:32；完成：2026-10-03 20:45:46（北京时间）。
- 总耗时：8 小时 52 分钟 14 秒；监督程序记录重启次数 0。
- 未发现训练 Traceback、CUDA OOM、NaN/Inf 损失或失败标记。
- 最终、最佳 SSIM、最佳 mIoU 检查点已保存，三套各导出 110 张 JPEG 和 PNG。

`completed.json`、`training_result_summary.json` 和
`from_scratch_20261003.train.log` 已下载，可独立核验。
主结果包包含完整断点、三组权重、图像、TensorBoard 和运维源代码。

## 内部验证结果

| 检查点 | epoch | PSNR | SSIM | mIoU | Dice |
|---|---:|---:|---:|---:|---:|
| 最佳 SSIM / mIoU | 71 | 24.7174 | 0.9268 | 0.5888 | 0.6085 |
| 第 100 轮常规验证 | 100 | 24.6120 | 0.9249 | 0.5622 | 0.6077 |
| 最终作者评测 | 100 | 24.5528 | 0.9247 | 0.5148 | 0.5402 |

最佳两项均落在 epoch 71。前两行来自每轮 `model_SA` 验证；最后一行来自
作者训练入口最终的 `model_S` 验证，连 prior 来源也不同，不能直接将两行
差异解读为训练后期退化。导出的三套图片明确采用 `model_SA` prior，
各自 manifest 记录协议、版本、checkpoint SHA256、尺寸和逐图有限值检查。

这些是官方代码的内部验证，不等同于论文表 II 的独立下游网络重训结果。
原代码的 pixAcc 有截断错误；UCIQE/UIQM meter 未更新，因此原日志两个 0
不代表实际质量分数。本次保留原实现，没有改变验证或训练算法。

## 结果包与检查点

远端结果包：`/root/autodl-tmp/DTIUIE-training-results-20261003.tar.gz`。
压缩包大小：1,966,805,588 字节。
SHA256：`ebcddb62d62408689e06aca12e8f5f2f8892290aaeded0004f4b8afc1c4a04a2`。

本机已下载、SHA256 校验并解压完成，解压目录为：
`results-20261003-ebcddb62d624/`。
具体回收状态以 `C:\Users\Lenovo\autodl-ssh\monitor\final_download.json` 为准。
三份检查点各自 SHA256 校验也通过；三套 JPEG/PNG 共 660 张均已在本机解码核验，
尺寸和模式均正确。记录见 `local_result_validation.json`；定时巡检任务已自动禁用。

| 权重 | SHA256 |
|---|---|
| epoch 100 final | `c21e6720775c5219f00bb56e4a52933a22ae38838f645fbcb71247eb657f704a` |
| epoch 71 best_ssim | `e1e2d13b09fd69965679aa26ab4785e7ef34a372851beeef8c11730c352c6292` |
| epoch 71 best_miou | `6ccb6e58cc62fc00f76e00b3b5c97f16f5d7c57e69f15eab20a98704964b65b6` |

解压后 `training_runs/from_scratch_20261003/latest_full.pth` 为最终完整恢复断点。
`outputs/trained_final_model_SA_20261003/` 是第 100 轮图像，
`outputs/trained_best_ssim_model_SA_20261003/` 和
`outputs/trained_best_miou_model_SA_20261003/` 是两套最佳图像。
每套 JPEG 与 PNG 各 110 张，固定 RGB 256×256。

运行过程及实际优化器配置见项目 `cloud_reproduction/TRAINING_RUN.md`。
