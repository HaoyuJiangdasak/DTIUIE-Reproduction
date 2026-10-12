# DTIUIE 云端复现准备与核对记录

当前正式目标与运行状态见 **TRAINING_RUN.md**：已按用户确认的官方代码默认配置
启动从头训练 100 epochs。本文下方保留的是此前预训练推理准备及论文核对记录。

核对日期：2026-10-02。依据桌面上的 16 页 IEEE TIP 论文、原交接文档、
当前代码提交 `fb09f91`、本地数据和官方检查点。

本次目标是已有权重的完整测试推理，不重新训练。原有源码、权重和数据保持原样；
本目录提供独立推理入口、环境脚本、盘点记录及本机小样本验证结果。
云端尚未连接，实际租用的显卡、镜像和驱动将在连接后读取。
交接文档里的 4090D 是建议配置，目前不能当作实际实例信息。

## 已有资产与还需安装的环境

| 项目 | 核对结果 |
|---|---|
| 官方权重 | `checkpoints/dtiuie_ckpt.pth`，432,055,901 字节，epoch 28 |
| 检查点内容 | `model_E`、`model_S`、`model_SA` 及训练状态；三模型已 strict 加载核验 |
| 测试数据 | images / reference / masks_png 各 110 张，名称和尺寸配对完整，掩码类别 0–7 |
| 训练数据 | 三目录各 1525 张；37 张掩码尺寸不一致，原始 ZIP 也有相同问题 |
| 最小出图上传内容 | 代码 + 权重 + test/images，约 442.44 MB，未压缩 |
| 含 test 参考图和掩码 | 约 465.58 MB，未压缩；本次上传包采用此方案 |
| 可以排除 | 原始 ZIP、trainval、历史记录、旧 outputs、.git、__pycache__ |

原始 ZIP 约 534 MB，本次不上传。完整数据与权重已在本地，不必云端重新下载。
推理不需要 torchaudio、Lightning、pyiqa、LPIPS、OpenCV、CUDA 编译器或训练优化器。
新入口也不会下载 VGG16-BN 的约 528 MB ImageNet 权重：直接完整加载检查点中的任务网络。

| 环境路线 | Python | torch / torchvision | 用途 |
|---|---|---|---|
| legacy | 3.9.16 | 1.13.1+cu116 / 0.14.1+cu116 | 对齐作者核心框架版本；适用于实际 GPU 运算检查通过的卡 |
| modern | 3.9.25 | 2.7.1+cu128 / 0.22.1+cu128 | 本机 Blackwell 已验证；云端需确认驱动和 GPU 运算 |

共同直接依赖见 `requirements-inference.txt`：NumPy 1.23.5、Pillow 9.4.0、
einops 0.6.1、timm 0.9.7、tqdm 4.65.0、huggingface-hub 0.26.2、safetensors 0.4.3。
安装器会解析这些包的必要间接依赖。环境建在独立的数据盘目录，使用命令级镜像参数，
不会全局修改 pip/conda 配置。`requirements-metrics-optional.txt` 仅供额外指标分析。

这不是原 YAML 的几百项包逐项重装；它是按当前推理入口实际依赖整理的环境。
论文未报告具体 Python / torch / CUDA 版本，原 YAML 是仓库提供的环境依据。

## 必须记录的推理差异

1. 原测试入口 `underwater_test.py:80` 仍导入不存在的 `TIUIED`。
   README 的 `--model model_dtiuie` 也不是可用的测试命令。
   原脚本还覆盖 `--pretrained`，忽略 `--save_dir`。直接使用本目录的新入口。
2. 官方测试入口用检查点 `model_S` 产生增强引导特征；训练和验证使用 `model_SA`。
   两份状态字典的 278 个键全部不同，本机对照输出也确实不同。
   新入口要求显式指定 `--guidance`，manifest 记录选择。
   首次按发布测试代码复现采用 `model_S`；按论文 prior/训练协议的对照采用 `model_SA`。
   不能在没有记录的情况下把二者互换。
3. 输入为 PIL RGB 转 tensor 的 [0,1]，不做 ImageNet mean/std 或 BGR 处理。
   复刻原测试脚本：先 bicubic 对齐到 32 倍数，再 bicubic 到 256×256，
   两次均 `align_corners=False`。原测试脚本最终保存也是 256×256，
   交接文档所说“恢复原图尺寸”不符合当前代码。
4. 保留网络自己的条件 min-max 归一化。固定 batch=1、float32、eval、no_grad，
   不额外套 sigmoid/clamp，不启用 AMP。逐图检查 shape、有限值和输出范围。
5. JPEG 按原脚本的 `×255 → uint8` 截断及 PIL 默认编码保存；另存无损 PNG。
   如需要 PSNR/SSIM，应在 JPEG 压缩前的浮点张量上评估，而非直接读 JPEG 对表。
6. 新版 Torch 的 `weights_only=True` 已实测能读该检查点；旧版 1.13 的受限读取器
   会因训练状态中的浮点字段报 `Unsupported operand 71`。
   新入口仅对 SHA256 完全匹配的这份官方检查点使用旧版完整读取器，
   其它文件在旧版路径下拒绝加载；现代环境仍用受限读取器。

官方检查点 SHA256：
`e9449842c0f81211f72bceacba63a7d1d2c085c920e00901e26e2e40dea0c7b6`。

## 论文结果与本次推理的对应关系

论文 PDF 第 8 页（印刷页 2880）规定输入 256×256，AdamW，初始 lr=1e-4、
cosine 调度，在 RTX 3090Ti 上训练，并明确增强后的图像用于重新训练下游网络。
100 epoch / batch=4 来自代码默认值，论文正文没有给出这两个数。
本次推理不执行训练、微调或 210 个参考图选取模型的数据集重建。

| 论文位置 | 指标与复现条件 |
|---|---|
| 表 II，PDF p9 | SUIM/LIACi 上 UNet、DeepLabV3、FPN、DPT 的 Dice/mIoU；需要对应下游重训或检查点 |
| 表 III/IV，PDF p11 | RUOD 检测、UIIS 实例分割；需额外数据和对应检测/实例分割模型 |
| 表 V，PDF p12 | SUIM-E/UIEB 的 UIQM/UCIQE/NUIQ；当前 TI-UIED test 不等价于这些评测数据 |
| 表 VI，PDF p12 | 论文增强模型 9.5M/361.0G；当前代码 model_E 为 8,923,734 参数，差异需注明 |

全文未报告 PSNR/SSIM。当前检查点内的任务网络可以用于辅助分析，
其分割分数不能当作表 II 的下游重训结果。
另外，论文 p8 把 LIACi 的 78.02/83.13/81.68/82.82 称为 mIoU，
但 p9 表 II 明确这些是 Dice；真实 mIoU 是 55.27/55.47/53.72/57.17。
核对论文结果时以表头和列对应为准。

现有指标代码还存在这些问题，故本次不把它们直接作为论文指标验收：

- `underwater_test.py` 只出图，不自动算指标；两个 eval 文件只是函数库。
- `underwater_train.py` 的 UCIQE/UIQM meter 未 update，日志中的 0 不是测得结果。
- `uie_eval.py` 的 UCIQE 用 HSV；不能未经核对与论文表 V 的 54.xx 直接比较。
- `uie_eval.py` 的 UIQM 原地乘 255，调用时需复制输入以防修改后续指标数据。
- `seg_eval.py:85` 在 softmax 后先 `.long()` 再 argmax，pixAcc 有真实错误。
  mIoU 对含背景的 8 类求均值，Dice 排除类别 0，空类别规则也需记录。
- 训练集的 37 张尺寸异常掩码暂不裁剪；已有权重在完整匹配的 test 上推理不受影响。

交接中的其它推断也需区别：梯度累积不会等价模拟大 batch 的 BatchNorm 统计；
旧的四张 smoke 输出没有完整运行协议，不能当作这次可追溯结果的基准。
本次模型部署还要运行约 41.90M 参数的引导网络，不能只按增强网参数估计总成本。

## 云端运行顺序

连接后先读取实际实例：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /root/autodl-tmp
conda --version
```

将准备包上传到 `/root/autodl-tmp/`，检查解压目标尚不存在后再解压，避免覆盖其它项目。
包顶层目录为 `DTIUIE/`。随后按实际卡和驱动选择一条环境路线：

```bash
cd /root/autodl-tmp/DTIUIE
bash cloud_reproduction/setup_env.sh legacy
/root/autodl-tmp/envs/dtiuie-legacy/bin/python cloud_reproduction/verify_gpu.py
```

Blackwell 或需要现代框架时把 `legacy` 换为 `modern`。
`verify_gpu.py` 不仅检查 `cuda.is_available()`，还实际执行 CUDA/cuDNN 卷积。
现有平台镜像已提供完全匹配的框架时，可验证后复用它，避免重复下载。

先跑 4 张官方测试协议：

```bash
/root/autodl-tmp/envs/dtiuie-legacy/bin/python cloud_reproduction/infer_checkpoint.py \
  --guidance model_S --device cuda:0 --limit 4 --output-dir outputs/cloud_smoke_model_S
```

确认通过后跑全 110 张：

```bash
/root/autodl-tmp/envs/dtiuie-legacy/bin/python cloud_reproduction/infer_checkpoint.py \
  --guidance model_S --device cuda:0 --output-dir outputs/cloud_full_model_S
```

需要论文 prior 对照时另跑 `--guidance model_SA`，使用独立输出目录。
新入口拒绝覆盖已有非空目录。完成后检查 `run_manifest.json` 的
`status=complete`、`expected_count=110`、`completed_count=110`、检查点 hash、
JPEG/PNG 各 110 张，并下载输出、manifest 和包版本记录。

## 已完成的本机验证及其限制

- 实际 GPU：RTX 5060 Laptop 8 GB；Python 3.9.25、Torch 2.7.1+cu128。
- `model_S` 与 `model_SA` 各 4 张均成功，PNG/JPEG 能解码，固定 RGB 256×256，
  无 NaN/Inf、输出 [0,1]。两组峰值 allocated 均约 674 MiB。
- 旧环境 Torch 1.13.1+cu116 在 CPU 完成三模型 strict 加载及错误 hash 拒绝验证。
- Bash 安装脚本语法、Python 语法、实际 CUDA 小卷积检查通过。
- 本机实际 Pillow 是 11.3.0，和作者/云端计划的 9.4.0 不同，已写入 manifest，
  不宣称本机输出与作者环境逐位一致。
- 本机 4 张推理循环约 1 秒，不含解释器启动；此数不代表云端运行时长。
- 本次尚未运行云端 Linux 环境安装、云端 GPU 前向或全 110 张推理。

记录：`data_audit.json`、`local_validation.json`、两组 smoke 的 `run_manifest.json`。
图像对照：`smoke_comparison.png`。源码原始 Git tracked diff 为空。
