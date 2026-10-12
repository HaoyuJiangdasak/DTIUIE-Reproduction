# DTIUIE 复现工程

对以下论文的**完整复现与验证**：

> **Downstream Task-Inspired Underwater Image Enhancement: A Perception-Aware Study From Dataset Construction to Network Design**
> Bosen Lin, Feng Gao\*, Yanwei Yu, Junyu Dong, Qian Du
> *IEEE Transactions on Image Processing*, 2026
> 官方实现：<https://github.com/oucailab/DTIUIE> ｜ arXiv: <https://arxiv.org/abs/2603.01767>

本仓库记录从**环境搭建 → 显存可行性评估 → 云端租卡 → 从零训练 100 epoch → 指标验证 → 报告成文**的全过程，
包含可直接复用的脚本、完整的训练日志与曲线、三套训练权重、660 张增强结果图，以及一份可提交的复现报告。

---

## 一、复现结论速览

| 事项 | 结论 |
|---|---|
| 云端训练硬件 | RTX 4090D 24 GB，Python 3.9.16，PyTorch 1.13.1+cu116 |
| 训练配置 | batch=4、100 epochs、256×256、warmup 10，单卡 float32 |
| 数据规模 | 训练 1525 张，内部验证 110 张，每轮 381 个 batch |
| **训练耗时** | **8 小时 52 分 14 秒**（2026-10-03 11:53:32 → 20:45:46，北京时间） |
| 完成度 | 100/100 轮全部完成，global step = 38100，0 次重启，无 OOM / NaN / Traceback |
| 本机可行性 | RTX 5060 Laptop **8 GB**：推理绰绰有余（峰值 674 MiB）；训练仅 batch=1 可用（4434 MiB，0.87 s/iter） |

### 内部验证指标

| 检查点 | epoch | PSNR | SSIM | mIoU | Dice |
|---|---:|---:|---:|---:|---:|
| **最佳 SSIM / mIoU** | **71** | **24.7174** | **0.9268** | **0.5888** | **0.6085** |
| 第 100 轮常规验证 | 100 | 24.6120 | 0.9249 | 0.5622 | 0.6077 |
| 最终作者评测口径 | 100 | 24.5528 | 0.9247 | 0.5148 | 0.5402 |

> 前两行来自每轮 `model_SA` 验证；最后一行来自官方训练入口最终的 `model_S` 验证。
> 两者 prior 来源不同，**不能**把行间差异直接解读为训练后期退化。
>
> ⚠️ 这些是官方代码的**内部验证**结果，不等同于论文 Table II 的「增强图重新训练下游网络」口径，不可直接对表。

### 与论文的一致性核对

- `model_E` 参数量实测 **8,923,734**（论文 Table VI 报 9.5 M）——差异已在报告中注明。
- 论文全文**未报告** PSNR/SSIM，故本复现的 PSNR/SSIM 属补充测量。
- 论文 p8 把 LIACi 的 78.02/83.13/81.68/82.82 称为 mIoU，但 p9 表 II 表头明确这些是 **Dice**；真实 mIoU 为 55.27/55.47/53.72/57.17。
- 官方 `environment.yaml` 钉死 torch 1.13.1+cu116（最高支持 sm_86），在 Blackwell（RTX 50 系，sm_120）上**无法用于 GPU 运算**，只能退化 CPU；云端 4090D（sm_89）可正常使用。

---

## 二、仓库结构

```
DTIUIE-Reproduction/
├── README.md                      本文件
├── README_upstream.md             官方原版 README（存档）
├── 网盘上传清单.md                 大文件（权重/数据集）下载指引
├── 复现交接文档.md                 完整复现手册：环境、显存实测、训练、踩坑、微调
├── environment.yaml               官方环境定义（作者原版）
│
├── underwater_train.py            官方训练入口（三阶段循环）
├── underwater_test.py             官方测试入口（加载权重出增强图）
├── uie_eval.py / seg_eval.py      官方指标函数库（PSNR/SSIM/UCIQE/UIQM、mIoU/Dice）
├── dataloader_seg.py / loss.py    官方数据加载与损失（含 TDP FeatureLoss）
├── models/                        官方网络定义（FRB + DEB + TA-CTB / VGG16-UNet）
├── utils/                         官方工具（logger、余弦退火 LR 调度等）
│
├── cloud_reproduction/            云端复现脚本与核对记录
│   ├── README.md                  推理准备、论文核对、资产盘点
│   ├── TRAINING_RUN.md            本次训练的运行协议与实际优化器配置
│   ├── PATCH_NOTES.md             对官方代码的补丁说明
│   ├── infer_checkpoint.py        独立推理入口（可指定 prior 来源）
│   ├── underwater_train_monitored.py  带监控的训练入口
│   ├── training_observer.py       训练巡检与快照
│   ├── setup_env.sh / verify_gpu.py   环境安装与 GPU 实测校验
│   └── data_audit.json / local_validation.json
│
├── cloud_training_results/        本次云端训练产物
│   ├── RESULTS.md                 结果汇总与全部 SHA256
│   ├── from_scratch_20261003.train.log   完整训练日志（100 轮）
│   ├── training_result_summary.json / completed.json
│   ├── latest_training_status.json / local_result_validation.json
│   ├── trained_comparison.png     增强前后对照图
│   └── results-20261003-ebcddb62d624/
│       ├── outputs/               三套结果各 110 张（JPEG + PNG）
│       │   ├── trained_final_model_SA_20261003/
│       │   ├── trained_best_ssim_model_SA_20261003/
│       │   └── trained_best_miou_model_SA_20261003/
│       ├── tensorboard_logs/      训练曲线原始事件文件
│       └── cloud_reproduction/    云端实际使用的运维代码
│
├── reproduction_report/           复现报告（核心交付物）
│   ├── index.html                 离线可视化仪表板（总览 / 训练 / GPU 曲线）
│   ├── DTIUIE_复现报告.pdf         可提交的正式报告
│   ├── DTIUIE_交接文档.pdf         交接版
│   ├── docs/                      复现过程与结果 / 复现交接文档
│   ├── figures/                   高清图（PNG / SVG / PDF 三格式）
│   ├── data/                      重建曲线用的数据、facts.json、来源 SHA256
│   ├── evidence/                  原始证据（日志、TensorBoard、样例图、105 个来源映射）
│   ├── scripts/                   重建与绘图源码
│   ├── tools/latex/               LaTeX 源码与双遍编译记录
│   └── tools/matlab/              MATLAB FIG / MAT / PDF / PNG
│
└── _历史记录/                      早期环境安装记录
```

---

## 三、快速开始

### 3.1 云端严格复现（推荐，24 GB 卡）

```bash
# 1) 建作者环境（约 20~40 分钟）
conda env create -f environment.yaml
conda activate dtiuie

# 2) 校验 GPU 真实可用（不只查 cuda.is_available()，会跑一次卷积）
python cloud_reproduction/verify_gpu.py

# 3) 数据与权重就位（见第四节）
mkdir -p dataset checkpoints

# 4) 训练（默认 batch=4、100 epochs）
python underwater_train.py --database TIUIED
```

### 3.2 本机小显存复现（8 GB 卡）

```bash
# 本机必须换用支持 sm_120 的 torch
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

mkdir -p outputs/DTIUIE
python underwater_test.py --draw_images                              # 官方权重出图，几分钟
python underwater_train.py --database TIUIED --batch_size 1          # 约 1.5~2 天跑完 100 轮
```

> **注意**：官方 `environment.yaml`（torch 1.13.1+cu116）在 RTX 50 系（sm_120）上**只能跑 CPU**。
> 详见 `复现交接文档.md` §2.4。

### 3.3 已知必须修复的问题

官方代码有两处会导致直接报错，跑之前先修（详见 `复现交接文档.md` §8）：

1. `underwater_test.py` 第 80、81 行导入不存在的 `TIUIED` 类 → 改为 `DTIUIE`；
2. README 里的 `--model model_dtiuie` 参数名有误 → 测试直接用 `python underwater_test.py --draw_images`；
3. `--draw_images` 前需先 `mkdir outputs/DTIUIE`，否则保存时报错。

---

## 四、预训练权重与数据集

GitHub 单文件上限 100 MB，本项目的大文件（共 11.9 GB）**无法进入仓库**，统一走百度网盘：

| 文件 | 大小 | 说明 | 下载 |
|---|---|---|---|
| 完整训练结果包 | 1.83 GB | 含全部断点、三套权重、660 张图、TensorBoard | ⏳ 待补充链接 |
| best_ssim.pth | 412 MB | 最佳 SSIM 权重（epoch 71） | ⏳ 待补充链接 |
| best_miou.pth | 412 MB | 最佳 mIoU 权重（epoch 71） | ⏳ 待补充链接 |
| final（epoch 100） | 412 MB | 最终权重 | ⏳ 待补充链接 |
| latest_full.pth | 732 MB | 最终完整断点，可续训 | ⏳ 待补充链接 |

**第三方资产**（论文作者发布，非本仓库产出）：

| 文件 | 作者原始链接 |
|---|---|
| DTI-UIE 官方权重（epoch 28） | <https://pan.baidu.com/s/1A3nXzvM7dLWm6OxAeIHWgw?pwd=nkbi>（提取码 `nkbi`） |
| TI-UIED 数据集 | <https://pan.baidu.com/s/1bn0ZRFZYGz1TaNzTwrDwew?pwd=v7fv>（提取码 `v7fv`） |

数据目录须摆成（**注意官方 README 写的 `train` 是错的，代码实际读 `trainval`**）：

```
dataset/
├── trainval/{images, masks_png, reference}      # 各 1525 张
└── test/{images, masks_png, reference}          # 各 110 张
```

### 完整性校验

```
ebcddb62d62408689e06aca12e8f5f2f8892290aaeded0004f4b8afc1c4a04a2  DTIUIE-training-results-20261003.tar.gz
e1e2d13b09fd69965679aa26ab4785e7ef34a372851beeef8c11730c352c6292  ...-best_ssim.pth
6ccb6e58cc62fc00f76e00b3b5c97f16f5d7c57e69f15eab20a98704964b65b6  ...-best_miou.pth
c21e6720775c5219f00bb56e4a52933a22ae38838f645fbcb71247eb657f704a  ...-final
```

---

## 五、复现过程中发现的问题

官方代码共发现 **11 处**问题，按严重程度排序（完整版见 `复现交接文档.md` §8）：

| # | 问题 | 修复 |
|---|---|---|
| ① | `underwater_test.py:80` 导入不存在的 `TIUIED` 类，一跑就 `ImportError` | 改为 `DTIUIE` |
| ② | README 的 `--model` 参数名错误，测试时会 `NameError` | 测试不加该参数 |
| ③ | `--draw_images` 输出目录不存在导致保存失败 | 先建 `outputs/DTIUIE` |
| ④ | 8 GB 卡跑默认 batch=4 必 OOM（实测需 14.3 GB） | `--batch_size 1` |
| ⑤ | `--save_iter_checkpoint True` 引用了不存在的 `args.model` | 不要开这个开关 |
| ⑥ | 日志中 UCIQE/UIQM 恒为 0 | meter 从未 update，属未实现 |
| ⑦ | `--amp` 开关无效 | 训练循环缺少 `autocast` 上下文 |
| ⑧ | torch ≥ 2.6 的 `torch.load` 默认 `weights_only=True` | 显式传 `weights_only=False` |
| ⑨ | 首次运行会自动下载 vgg16_bn（528 MB） | 断网环境需手动放置 |
| ⑩ | 首次 CUDA 迭代特别慢（实测 78 秒） | cuDNN 调优，属正常现象 |
| ⑪ | Windows 下 `num_workers=8` 启动慢/偶发卡死 | 改为 2~4 |

另有两处**指标实现问题**，报告中已排除其干扰：

- `seg_eval.py:85` 在 softmax 后先 `.long()` 再 argmax，`pixAcc` 存在真实错误；
- `uie_eval.py` 的 UIQM 会原地乘 255，调用前需复制输入。

---

## 六、复现范围说明

| ✅ 本仓库已覆盖 | ❌ 论文有但官方仓库未提供，无法复现 |
|---|---|
| DTI-UIE 网络在 TI-UIED 上的完整训练（100 epoch） | TI-UIED 数据集构建管线（7 分割网 × 10 UIE 方法投票） |
| 官方权重推理出增强图 | SUIM / LIACi 上的下游网络重训（Table II） |
| 增强质量 PSNR/SSIM + 分割 mIoU/Dice 内部验证 | 目标检测 RUOD、实例分割 UIIS 对比（Table III/IV） |
| 从官方 ckpt 继续微调的方法与命令 | 传统方法与其它 DL 基线对比 |

---

## 七、引用

如果本复现对你的研究有帮助，请引用原论文：

```bibtex
@article{lin2026dtiuie,
  title={Perception-Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study From Dataset Construction to Network Design},
  author={Bosen Lin, Feng Gao, Yanwei Yu, Junyu Dong, Qian Du},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}
```

## 八、许可

原始代码与数据集遵循 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（**仅限非商业用途**），
任何商业用途需先取得正式授权。本仓库的复现文档、报告与脚本沿用同一许可。

## 九、致谢

感谢中国海洋大学 oucailab 提供官方实现与预训练权重。
本仓库为个人复现学习用途，代码主体版权归原作者所有。
