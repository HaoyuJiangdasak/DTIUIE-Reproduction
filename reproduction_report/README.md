# DTIUIE 复现可视化与交接

**100轮、38,100个训练step已完成，耗时8小时52分14秒。** 本目录从归档日志重建曲线，官方代码优先；内部验证结果与论文独立下游重训表格分开说明。

| 想查看什么 | 打开这里 |
|---|---|
| 可浏览的总览、训练与GPU曲线 | [离线仪表板](index.html) |
| 完整过程、难点、结果与10组图解释 | [复现过程与结果](docs/复现过程与结果.md) |
| 接手路径、重画、训练、恢复和推理命令 | [复现交接文档](docs/复现交接文档.md) |
| 可提交导师的PDF | [复现报告](DTIUIE_复现报告.pdf)、[交接文档](DTIUIE_交接文档.pdf) |
| LaTeX源码与实际编译记录 | [LaTeX目录](tools/latex/)、[双遍编译记录](tools/latex/latex_execution.json) |
| 高清图及矢量图 | [figures：PNG / SVG / PDF](figures/)、[图表目录及来源](figure_catalog.json) |
| 实际MATLAB调用与可编辑图 | [MATLAB FIG / MAT / PDF / PNG](tools/matlab/) |
| Origin实际可用性与导入材料 | [Origin说明](tools/origin/README.md)：本机COM受阻，未生成OPJU |
| 100轮、38,100step和94次GPU快照 | [data](data/)、[事实汇总](facts.json)、[源文件SHA256](data/provenance.json) |
| 重建与绘图源码 | [scripts](scripts/) |
| 便携原始证据与样例 | [evidence](evidence/)、[105个来源的映射与哈希](evidence/source_map.json) |
| 原始日志、权重与全部测试输出 | [完整结果归档目录](../cloud_training_results/results-20261003-ebcddb62d624/) |

GPU为离散巡检：92次训练中、2次完成后；8个超过7分钟的观测缺口，最长约20分钟。未测功率，不提供能耗；无效pixAcc、UIQM/UCIQE占位0已排除。最佳第71轮内部SSIM=0.926810、mIoU=0.588789。报告不重复复制大权重，不包含登录凭据。
