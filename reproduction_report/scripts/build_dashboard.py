"""Build an offline, self-contained interactive overview from recovered run data."""
import base64
from html import escape
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#D55E00', '#009E73', '#E69F00']


def finish_plot(fig, height=470):
    fig.update_layout(height=fig.layout.height or height, paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Microsoft YaHei, sans-serif', size=13, color='#18324A'),
                      margin=dict(l=62, r=25, t=45, b=55), hovermode='x unified',
                      legend=dict(orientation='h', y=1.16, x=0),
                      colorway=COLORS)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='#E4E9ED', zeroline=False)
    return fig


def div(fig, identifier):
    return finish_plot(fig).to_html(full_html=False, include_plotlyjs=False,
             div_id=identifier, config={'responsive': True, 'displaylogo': False,
             'toImageButtonOptions': {'format': 'png', 'scale': 2}})


def embedded_image(name):
    path = ROOT / 'figures' / (name + '.png')
    return '<img class="figure" alt="'+escape(name)+'" loading="lazy" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def main():
    e = pd.read_csv(ROOT/'data/epochs.csv')
    g = pd.read_csv(ROOT/'data/gpu_samples.csv')
    summary = json.loads((ROOT/'data/summary.json').read_text(encoding='utf-8'))
    duration = summary['elapsed_seconds_completed_record']/3600
    loss = go.Figure()
    for column,label in [('loss_E','E：增强总损失'),('loss_S','S：三项CE总损失'),('loss_SA_mean','SA：先验CE均值')]:
        loss.add_trace(go.Scatter(x=e.epoch,y=e[column],name=label,mode='lines'))
    loss.add_vrect(x0=1,x1=10,fillcolor='#E69F00',opacity=.1,line_width=0)
    loss.add_vline(x=10.5,line_dash='dash',line_color='#E69F00',annotation_text='TDP启用')
    loss.update_xaxes(title='训练轮次 (epoch)');loss.update_yaxes(title='每轮平均损失')
    metrics = make_subplots(rows=1,cols=2,subplot_titles=['内部验证 SSIM','内部验证 mIoU'])
    metrics.add_trace(go.Scatter(x=e.epoch,y=e.ssim,name='SSIM',line_color=COLORS[2]),row=1,col=1)
    metrics.add_trace(go.Scatter(x=e.epoch,y=e.miou*100,name='mIoU (%)',line_color=COLORS[0]),row=1,col=2)
    for col in (1,2):
        metrics.add_vline(x=71,line_dash='dot',line_color=COLORS[3],row=1,col=col)
        metrics.update_xaxes(title_text='训练轮次 (epoch)',row=1,col=col)
    metrics.update_yaxes(title_text='SSIM (0–1)',row=1,col=1)
    metrics.update_yaxes(title_text='mIoU (%)',row=1,col=2)
    telemetry = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.1)
    for row,column,label,color in [(1,'gpu_util_pct','GPU利用率 (%)',COLORS[0]),
                                   (2,'vram_used_mib','已使用显存 (MiB)',COLORS[1]),
                                   (3,'temp_c','温度 (°C)',COLORS[2])]:
        telemetry.add_trace(go.Scatter(x=g.elapsed_hours,y=g[column],mode='markers',
                           name=label,marker=dict(size=6,color=color),
                           customdata=g[['epoch','batch','timestamp_cst']],
                           hovertemplate='距启动 %{x:.3f} h<br>值 %{y}<br>epoch %{customdata[0]} / batch %{customdata[1]}<br>%{customdata[2]}<extra></extra>'),row=row,col=1)
        telemetry.update_yaxes(title_text=label,row=row,col=1)
        telemetry.add_vline(x=duration,line_dash='dot',line_color='#788792',row=row,col=1)
    telemetry.update_xaxes(title_text='距正式训练启动时间 (h)',row=3,col=1)
    telemetry.update_layout(height=660)
    progress = go.Figure(go.Scatter(x=e.elapsed_hours,y=e.epoch,mode='lines',name='完成的epoch',line_color=COLORS[2]))
    progress.update_xaxes(title='距启动时间 (h)');progress.update_yaxes(title='已完成epoch数量')
    gpu_stat=summary['gpu']
    rows=[]
    for name,details in [('最佳SSIM/mIoU epoch 71 / SA',summary['best_internal_validation']['ssim']['metrics']),
                         ('epoch 100 / SA',summary['epoch_100_model_SA'])]:
        rows.append('<tr><td>'+name+'</td><td>'+f'{details["psnr"]:.4f}'+'</td><td>'+f'{details["ssim"]:.6f}'+'</td><td>'+f'{details["miou"]*100:.2f}%'+'</td><td>'+f'{details["dice"]*100:.2f}%'+'</td></tr>')
    final=summary['final_log_model_S_different_protocol']
    rows.append('<tr><td>最终额外验证 / S</td><td>'+f'{final["psnr_rounded"]:.4f}'+'</td><td>'+f'{final["ssim_rounded"]:.4f}'+'</td><td>'+f'{final["miou"]*100:.2f}%'+'</td><td>'+f'{final["dice"]*100:.2f}%'+'</td></tr>')
    blocks = [
        '<section id="workflow"><h2>1. 做了什么：完整训练与证据链</h2>'+embedded_image('fig01_workflow')+'<p class="caption">100 epochs、batch=4，从头训练增强网；任务编码器按作者代码用 ImageNet 初始化。一次训练，不加载发布的 DTIUIE 权重。</p></section>',
        '<section id="training"><h2>2. 深度学习训练曲线</h2>'+div(loss,'loss-chart')+'<p class="caption">来源：TensorBoard完整scalar与每轮日志。第11轮目标由L1变为L1+TDP，因此总损失有跳变，不能直接当作发散。</p>'+div(metrics,'metric-chart')+'<p class="caption">来源：100轮内部验证，110张图，SA产生prior并分割。虚线epoch71；这不是论文下游模型重新训练的成绩。</p></section>',
        '<section id="gpu"><h2>3. GPU真实采样与运行时长</h2>'+div(telemetry,'gpu-chart')+'<p class="caption">94个离散采样，92个训练中、2个完成后。中位采样间隔约5分钟；8处超过7分钟的缺口，最长20分钟。仅显示实测点，不补造连续曲线；没有完整功率或能耗日志。</p>'+div(progress,'progress-chart')+embedded_image('fig07_timeline')+'</section>',
        '<section id="results"><h2>4. 得到的结果与可比性</h2><div class="table-wrap"><table><thead><tr><th>评价协议</th><th>PSNR (dB)</th><th>SSIM</th><th>mIoU</th><th>Dice</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p class="caption">S和SA是不同网络；最后一次日志评测还改变prior来源。指标为官方代码内部口径，不能将它们直接与论文表II比较。</p>'+embedded_image('fig09_image_comparison')+'<p class="caption">四个明确文件名的样例，三套训练权重各导出110张JPEG与PNG，合计660张可解码输出；样例展示不代替完整数据集评价。</p></section>',
        '<section id="audit"><h2>5. 数据、监控和困难</h2>'+embedded_image('fig08_data_audit')+'<details open><summary>最需要说明的真实问题</summary><ul><li>官方大轮子下载源超时：用可用镜像分块下载，按官方SHA256校验。</li><li>旧Torch受限加载器不支持该完整检查点：仅对核验SHA的文件采用完整读取器。</li><li>原resume参数无实现：增加完整模型、三优化器、调度器、scaler与RNG断点，保持原训练数学。</li><li>37张mask尺寸差异来自官方ZIP，按代码原样训练并记录风险。</li><li>pixAcc代码错误、UIQM/UCIQE与训练PSNR零占位均排除，不包装成有效成果。</li><li>代码优先：L1、nearest、实际AdamW weight_decay=0.01等均保留；没有谎称与论文公式逐项一致。</li></ul></details></section>',
        '<section id="deliverables"><h2>6. 本地工具与可重画交付</h2><ul><li>Matplotlib：10张主题图，每张PNG / PDF / SVG，真实CSV可重新绘制。</li><li>MATLAB R2025b：3套实际生成并重开验证的FIG、300dpi PNG和矢量PDF。</li><li>LaTeX：提供中文PDF报告与可编译.tex源码。</li><li>Origin：本机COM注册失效、可执行文件缺失；保留调用证据与导入/重建脚本，未生成OPJU。</li></ul><p><a href="docs/复现过程与结果.md">阅读完整Markdown复现报告</a> · <a href="docs/复现交接文档.md">交接文档</a> · <a href="DTIUIE_复现报告.pdf">PDF报告</a></p><p class="caption">本页面脚本、图像、训练与采样数据均嵌入，可离线打开。独立CSV、来源hash和工具运行日志在同目录中。</p></section>'
    ]
    style = '''
:root{--bg:#f4f5f2;--surface:#fff;--border:#dce2e4;--text:#18324a;--text-dim:#5e6e7a;--accent:#0072b2;--green:#009e73;--gold:#b77710}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;line-height:1.7}
.wrap{max-width:1450px;margin:auto;display:grid;grid-template-columns:190px minmax(0,1fr);gap:36px;padding:32px}.toc{position:sticky;top:24px;align-self:start;display:grid;gap:12px;font-size:14px}.toc a{color:var(--text-dim);text-decoration:none;border-left:2px solid transparent;padding-left:12px}.toc a.active{color:var(--accent);border-color:var(--accent)}.main{min-width:0}h1{font-size:30px;line-height:1.4;max-width:850px;margin:8px 0}h2{font-size:23px;margin:0 0 20px}.subtitle,.caption{color:var(--text-dim)}.caption{font-size:13px;margin:12px 0 30px}.hero{border-bottom:2px solid var(--border);padding-bottom:24px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}.kpi{padding:12px 14px;border-left:3px solid var(--accent);background:var(--surface)}.kpi strong{display:block;font-family:Consolas,monospace;font-size:25px;color:var(--accent)}.kpi span{font-size:13px;color:var(--text-dim)}section{padding:30px 0;border-bottom:1px solid var(--border);scroll-margin-top:20px}.figure{display:block;max-width:100%;height:auto;background:white;margin:15px auto}.table-wrap{overflow-x:auto;background:white;border:1px solid var(--border)}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:12px;border-bottom:1px solid var(--border);white-space:nowrap}th{background:#edf1f3}a{color:var(--accent);text-underline-offset:3px}details{padding:16px 22px;background:white;border:1px solid var(--border)}summary{cursor:pointer;font-weight:bold}li{margin:9px 0}footer{font-size:12px;padding:25px 0;color:var(--text-dim)}.js-plotly-plot{min-height:400px;background:white}
@media(max-width:900px){.wrap{display:block;padding:18px}.toc{z-index:3;top:0;display:flex;overflow-x:auto;background:var(--bg);padding:12px 0;position:sticky}.toc a{white-space:nowrap;border-left:0;border-bottom:2px solid transparent;padding:0 9px}h1{font-size:25px}.kpis{grid-template-columns:repeat(2,1fr)}section{scroll-margin-top:58px}}@media print{.toc{display:none}.wrap{display:block}.js-plotly-plot{break-inside:avoid}section{break-inside:avoid}.modebar{display:none}}
'''
    nav = '<nav class="toc"><b>DTIUIE 复现档案</b>'+''.join('<a href="#'+identifier+'">'+label+'</a>' for identifier,label in [('workflow','01 流程与证据'),('training','02 训练与验证'),('gpu','03 GPU与时间线'),('results','04 权重与图像'),('audit','05 难点与限制'),('deliverables','06 工具与交接')])+'</nav>'
    hero = '<header class="hero"><span class="subtitle">真实日志重建 · 2026-10-03 · 单次完整运行</span><h1>DTIUIE：从头训练100轮的完整复现记录</h1><p class="subtitle">官方开源代码优先，代码未说明的再参考论文。图表、CSV、软件工程产物与来源校验形成可追溯证据链。</p><div class="kpis"><div class="kpi"><strong>100 / 100</strong><span>完成训练轮次</span></div><div class="kpi"><strong>38,100</strong><span>真实训练step</span></div><div class="kpi"><strong>8h 52m</strong><span>完整训练耗时</span></div><div class="kpi"><strong>93%</strong><span>训练中GPU采样中位数</span></div></div></header>'
    javascript = '''
const links=[...document.querySelectorAll('.toc a')];
const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+entry.target.id));}}},{rootMargin:'-5% 0px -78% 0px'});
document.querySelectorAll('section').forEach(section=>observer.observe(section));
'''
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DTIUIE完整复现可视化报告</title><style>'+style+'</style><script>'+get_plotlyjs()+'</script></head><body><div class="wrap">'+nav+'<main class="main">'+hero+''.join(blocks)+'<footer>来源：训练日志、TensorBoard、94次SSH巡检和已校验权重导出。有效数据范围及采样缺口在各图中明示；未记录项不补造。</footer></main></div><script>'+javascript+'</script></body></html>'
    (ROOT/'index.html').write_text(html,encoding='utf-8')
    print('Offline dashboard saved:',ROOT/'index.html')


if __name__ == '__main__':
    main()
