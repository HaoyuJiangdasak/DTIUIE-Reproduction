# Origin 专项工具检查与可导入材料

**本机 Origin 未能实际启动。本目录没有生成 Origin 项目或 Origin 图像。**

实际调用使用现有 `D:\miniconda3\envs\dtiuie-gpu\python.exe`（Python 3.9.25）的 `win32com.client.DispatchEx("Origin.Application")`，以新建隔离 COM 实例的方式调用。失败码为 `-2147221164`（Class not registered）。没有连接用户已打开的 Origin 窗口，也没有改写注册表或安装软件。

注册表中的旧 Origin COM 配置指向 `D:\origin2026\Origin2026正式发行版\Origin64.exe`；此文件已经不存在，`D:\origin2026` 为空。对 D 盘，以及 C 盘的 Program Files、Program Files (x86)、AppData/Local 查找 `Origin64.exe` / `Origin.exe` / `OriginLab.exe`，均未找到可执行文件。实际检查证据见 `origin_availability.json` 和 `origin_probe.log`。

可导入材料：

- `epochs.csv`：100 轮真实训练与验证记录，复制自报告 `data/epochs.csv`。
- `gpu_samples.csv`：94 条真实 SSH 巡检快照，含训练及完成后的采样。状态由 `training_status` 给出。
- `origin_builtin.py`：供运行正常的 Origin 内置 Python 重建可编辑工作表、SSIM 和三项损失折线图，以及 GPU 利用率、显存、温度散点图。此重建脚本已通过 Python 语法检查，接口与公开 originpro 包源码核对，**尚未完成 Origin 软件内运行验证**。

GPU 每条采样保留真实时间，采样间隔中位数约 300 秒，最大间隔约 1,200 秒。绘图采用散点，既不插值也不把采样缺口填成连续监测。CSV 的 `power_w` 为空时表示未记录功耗，不能据此计算耗电量。

未来安装或恢复可用 Origin 后，可在 PowerShell 执行下列命令。脚本仅在独立、隐藏的 Origin COM 实例内运行；输出仍写到本目录：

```powershell
& 'D:\miniconda3\envs\dtiuie-gpu\python.exe' 'C:\Users\Lenovo\Desktop\DTIUIE\reproduction_report\scripts\plot_origin.py'
```

成功时才会产生 `DTIUIE_training_and_GPU.opju`、各图 PNG 和 `origin_export_success.json`。也可在一份**新建** Origin 项目的内置 Python 中运行 `origin_builtin.py`。项目会清空该新建会话再创建本报告内容，请勿在包含用户其他工作的会话中直接运行。
