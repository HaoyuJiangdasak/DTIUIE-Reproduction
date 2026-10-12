"""Draw source-backed, publication-ready figures for the completed DTIUIE run."""
from pathlib import Path
import json
import textwrap
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
EXTRACTED = PROJECT / 'cloud_training_results/results-20261003-ebcddb62d624'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)
C = dict(blue='#0072B2', green='#009E73', orange='#D55E00', gold='#E69F00', gray='#6A737D')
FONT = font_manager.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
CATALOG = []
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)


def save(fig, name, title, caption, sources):
    for ax in fig.axes:
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(labelsize=13)
        legend = ax.get_legend()
        if legend:
            for label in legend.get_texts():label.set_fontsize(13)
        for label in ax.texts:
            if label.get_fontsize() < 14:label.set_fontsize(14)
    fig.text(0.02, 0.018, textwrap.fill(caption, width=90), ha='left', va='bottom', fontsize=8, color='#424B57')
    fig.subplots_adjust(bottom=0.15, top=0.83, hspace=0.45, wspace=0.34)
    fig.suptitle(title, y=0.985, fontweight='bold', fontsize=20)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / (name + '.' + extension), dpi=400, bbox_inches='tight')
    plt.close(fig)
    CATALOG.append({'name': name, 'title': title, 'caption': caption, 'sources': sources,
                    'formats': ['png', 'pdf', 'svg']})


def shade(ax):
    ax.axvspan(0.5, 10.5, color=C['gold'], alpha=0.09, zorder=0)
    ax.axvline(10.5, color=C['gold'], linestyle='--', linewidth=1)
    ax.axvline(71, color=C['green'], linestyle=':', linewidth=1)
    ax.set_xlim(1, 100)
    ax.set_xlabel('Epoch / 训练轮次')
    ax.grid(axis='y', color='#D9DEE4', linewidth=0.5, alpha=0.65)


def box(ax, x, y, w, h, title, body, color=C['blue']):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.025',
                         fc='white', ec=color, lw=1.6)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h*0.76, title, ha='center', va='center', fontsize=11, weight='bold', color=color)
    ax.text(x+w/2, y+h*0.35, body, ha='center', va='center', fontsize=9, linespacing=1.5)


def arrow(ax, start, end, text=''):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                 lw=1.2, color=C['gray']))
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+0.025, text, ha='center', fontsize=8)


def workflow():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    box(ax, .02, .74, .27, .2, '数据与环境', '训练1525 / 验证110\n4090 D · Torch 1.13.1+cu116')
    box(ax, .365, .74, .27, .2, '完整训练', 'batch 4 · 256×256 · 100轮\n38100 steps · 8h52m')
    box(ax, .71, .74, .27, .2, '验证、导出与回收', '最佳71 / 最终100\n3组权重 · 660张输出 · SHA256')
    arrow(ax, (.30, .84), (.35, .84)); arrow(ax, (.65, .84), (.695, .84))
    ax.text(.02, .655, '每个 batch 内的三阶段更新（保留官方实现）', fontsize=12, weight='bold')
    box(ax, .03, .37, .27, .23, 'Phase 0：训练先验网 SA', '原图 → Segprior\nCE更新先验网络SA', C['green'])
    box(ax, .365, .37, .27, .23, 'Phase 1：训练增强网 E', 'SA特征引导E\nL1 + C5特征L1（第11轮起）', C['blue'])
    box(ax, .70, .37, .27, .23, 'Phase 2：训练任务网 S', '增强 / reference / 8×8混合图\n三项CE求和，更新S', C['orange'])
    arrow(ax, (.31, .485), (.35, .485)); arrow(ax, (.646, .485), (.685, .485))
    box(ax, .07, .08, .86, .17, '运行保障不改变训练目标',
        '保活巡检 · 非有限损失检测 · 每轮完整断点\n每10轮本地备份 · 测试图导出 · SHA256校验', C['gray'])
    save(fig, 'fig01_workflow', '复现流程：从输入数据到可核验结果',
         '来源：官方训练源码、run_metadata.json、completed.json；本次未加载发布的DTIUIE检查点，VGG编码器采用作者指定ImageNet初始化。',
         ['underwater_train.py', 'run_metadata.json', 'completed.json'])


def losses(e):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    specs = [('loss_E', 'A  增强网络总损失', C['blue']),
             ('loss_S', 'B  任务网络三项CE总损失', C['orange']),
             ('loss_SA_mean', 'C  先验网络CE损失', C['green'])]
    for ax, (column, title, color) in zip(axes.flat, specs):
        ax.plot(e.epoch, e[column], color=color, lw=1.7, label='每轮训练batch的均值')
        shade(ax); ax.set_title(title, loc='left'); ax.set_ylabel('Loss / 损失'); ax.legend(frameon=False)
    ax = axes.flat[3]
    ax.plot(e.epoch, e.loss_enc_pixel_mean, color=C['blue'], label='Pixel L1')
    ax.plot(e.epoch, e.loss_enc_feature_mean, color=C['orange'], label='C5 feature L1（第11轮起）')
    shade(ax); ax.set_title('D  增强损失的两项组成', loc='left'); ax.set_ylabel('Loss / 损失'); ax.legend(frameon=False)
    save(fig, 'fig02_losses', '三阶段损失与TDP启用过程',
         '来源：TensorBoard完整scalar与训练INFO日志；浅色区为epoch 1–10，虚线为TDP启用边界。单次运行，不绘制重复实验置信区间。',
         ['data/epochs.csv', 'TensorBoard event file'])


def validation(e):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    specs = [('psnr', 'A  PSNR', 'PSNR (dB)', 1, C['blue']),
             ('ssim', 'B  SSIM', 'SSIM (0–1)', 1, C['green']),
             ('miou', 'C  分割 mIoU', 'mIoU (%)', 100, C['orange']),
             ('dice', 'D  分割 Dice', 'Dice (%)', 100, C['gold'])]
    for ax, (col, title, ylabel, factor, color) in zip(axes.flat, specs):
        ax.plot(e.epoch, e[col]*factor, color=color, lw=1.8, label='每轮内部验证（SA prior/分割）')
        shade(ax); ax.set_title(title, loc='left'); ax.set_ylabel(ylabel)
        value = e.loc[e.epoch==71, col].iloc[0]*factor
        ax.scatter([71], [value], color=C['green'], s=35, zorder=4)
        ax.annotate('epoch 71: {:.4f}'.format(value), (71, value), xytext=(-105, -28),
                    textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color=C['gray']))
        ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig03_validation', '100轮内部验证：最佳SSIM与mIoU均在第71轮',
         '来源：每轮TensorBoard验证scalar，110张内部验证图；mIoU汇总8类，Dice排除背景并按图平均。此图不是论文表II的下游重训成绩。',
         ['data/epochs.csv', 'seg_eval.py'])


def learning_rates(e):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(e.epoch, e.lr_E, color=C['blue'], label='E：epoch结束后记录的lr')
    axes[0].set_yscale('log'); axes[0].set_ylabel('Learning rate / 学习率')
    axes[0].set_title('A  增强网 cosine restart [10,90]', loc='left')
    axes[1].plot(e.epoch, e.lr_S_derived, color=C['orange'], label='S：从官方scheduler规则恢复')
    axes[1].plot(e.epoch, e.lr_SA_code, color=C['green'], linestyle='--', label='SA：0.02恒定，官方未step')
    axes[1].set_yscale('log'); axes[1].set_ylabel('Learning rate / 学习率')
    axes[1].set_title('B  两个SGD任务网络的实际调度', loc='left')
    for ax in axes: shade(ax); ax.legend(frameon=False)
    save(fig, 'fig04_lr', '真实学习率：区分实际参数与未生效的CLI参数',
         'E来自TensorBoard；S由作者cosine规则恢复、巡检点交叉核验；SA保持官方恒定lr。AdamW实际weight decay=0.01，而日志CLI写0.001。',
         ['data/epochs.csv', 'utils/lr_scheduler.py', 'run_metadata.json'])


def segmented_plot(ax, x, y, color, label):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    # Break gaps above 7 minutes; never invent samples between inspections.
    splits = np.flatnonzero(np.diff(x) > 7/60) + 1
    for idx, group in enumerate(np.split(np.arange(len(x)), splits)):
        ax.plot(x[group], y[group], '-o', ms=3, lw=1.1, color=color,
                label=label if idx==0 else None)


def gpu(g, duration_hours):
    g = g[g.connection_ok.eq(True) & g.gpu_util_pct.notna()].sort_values('elapsed_hours')
    training = g[g.elapsed_hours.between(0, duration_hours)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    segmented_plot(axes[0], g.elapsed_hours, g.gpu_util_pct, C['blue'], 'nvidia-smi实测采样')
    axes[0].set_ylim(-2, 105); axes[0].set_ylabel('GPU利用率 (%)')
    axes[0].set_title('A  GPU计算利用率：训练波动与训练后空闲', loc='left')
    segmented_plot(axes[1], g.elapsed_hours, g.vram_used_mib/1024, C['orange'], 'nvidia-smi已使用显存')
    axes[1].axhline(g.vram_total_mib.median()/1024, ls='--', color=C['gray'], label='实例报告的显存容量')
    axes[1].set_ylabel('显存 (GiB)'); axes[1].set_title('B  使用量与容量', loc='left')
    segmented_plot(axes[2], g.elapsed_hours, g.temp_c, C['green'], 'GPU温度')
    axes[2].set_ylabel('温度 (°C)'); axes[2].set_xlabel('距训练启动时间 (h)')
    axes[2].set_title('C  温度采样', loc='left')
    for ax in axes:
        ax.axvline(duration_hours, color=C['gray'], ls=':', lw=1)
        ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, loc='upper right')
    save(fig, 'fig05_gpu', 'GPU全程巡检：利用率、显存与温度',
         '来源：本机SSH巡检history JSON；图中点是离散实测，连线仅帮助阅读，>7分钟观测缺口断开。空闲/验证点不等同训练停顿；未连续记录功率，故不计算能耗。',
         ['data/gpu_samples.csv', 'monitor/history-*.json'])


def runtime(e):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].bar(e.epoch, e.epoch_duration_seconds/60, color=C['blue'], width=.8)
    axes[0].set_ylabel('相邻epoch末验证时间间隔 (min)'); shade(axes[0])
    axes[0].set_title('A  每轮耗时（含轮间保存等操作）', loc='left')
    axes[1].plot(e.elapsed_hours, e.epoch, color=C['green'], lw=2)
    axes[1].axhline(71, color=C['gold'], ls='--', label='最佳权重：epoch 71')
    axes[1].set_xlabel('距训练启动时间 (h)'); axes[1].set_ylabel('完成epoch数量')
    axes[1].set_title('B  训练进度随时间推进', loc='left'); axes[1].legend(frameon=False)
    axes[1].grid(alpha=.2)
    save(fig, 'fig06_epoch_runtime', '耗时与进度：100轮在8小时52分钟内完成',
         '来源：TensorBoard验证wall_time；柱形间隔不能解释为纯GPU计算耗时。首轮包含初始化，最终完成时间另由completed.json记录。',
         ['data/epochs.csv', 'completed.json'])


def timeline(e, checks, backups, duration_hours):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    warmup_end = e.loc[e.epoch==10, 'elapsed_hours'].iloc[0]
    best_time = e.loc[e.epoch==71, 'elapsed_hours'].iloc[0]
    ax.broken_barh([(0, warmup_end)], (3.4,.45), facecolors=C['gold'], label='前10轮：Pixel L1')
    ax.broken_barh([(warmup_end, duration_hours-warmup_end)], (3.4,.45), facecolors=C['blue'], label='第11–100轮：L1 + TDP')
    ax.scatter(checks.elapsed_hours, np.full(len(checks), 2.8), s=15, color=C['green'], label='SSH巡检')
    for _, item in backups.iterrows():
        start = item.get('elapsed_start_hours', np.nan)
        end = item.get('elapsed_end_hours', np.nan)
        if pd.notna(start) and pd.notna(end):
            ax.broken_barh([(start, max(end-start,.015))], (1.8,.35), facecolors=C['orange'])
    ax.axvline(best_time, color=C['green'], ls=':', lw=1.2)
    ax.text(best_time+.04, 4.05, '最佳：epoch 71', color=C['green'])
    ax.axvline(duration_hours, color=C['gray'], ls=':', lw=1.2)
    ax.text(duration_hours+.04, 3.7, '训练完成', color=C['gray'])
    ax.set_yticks([3.625,2.8,1.975], ['训练阶段', 'SSH巡检采样', '备份巡检（含传输）'])
    ax.set_ylim(1.2,4.5); ax.set_xlabel('距正式训练启动时间 (h)'); ax.set_title('监控和下载与训练并行进行', loc='left')
    ax.legend(frameon=False, ncol=3, loc='lower left'); ax.grid(axis='x', alpha=.2)
    save(fig, 'fig07_timeline', '任务时间线：训练、巡检与断点回收',
         '来源：TensorBoard、巡检JSON与备份记录。橙色条是包含断点传输的整次巡检时间，非纯下载计时；此时训练继续。未记录的环境准备时长不补造。',
         ['data/epochs.csv', 'data/monitor_checks.csv', 'data/checkpoints/chronology.csv'])


def data_audit():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    categories = ['原图', 'Reference', '语义mask']
    x = np.arange(3)
    axes[0].bar(x-.18, [1525]*3, width=.36, color=C['blue'], label='trainval')
    axes[0].bar(x+.18, [110]*3, width=.36, color=C['green'], label='test / 内部验证')
    axes[0].set_xticks(x, categories); axes[0].set_ylabel('文件数量 (张)')
    axes[0].set_title('A  三件套数量和文件名配对完整', loc='left'); axes[0].legend(frameon=False)
    for i in x:axes[0].text(i-.18,1540,'1525',ha='center',fontsize=9)
    axes[1].bar(['训练尺寸一致','训练mask尺寸异常','测试尺寸异常'],[1488,37,0],color=[C['blue'],C['orange'],C['gray']])
    axes[1].set_ylabel('图像组数 (组)'); axes[1].set_title('B  官方原包保留的尺寸风险', loc='left')
    axes[1].text(1,80,'37 / 1525 = 2.43%\n原始ZIP中也存在\n本次按官方数据原样训练',ha='center',color=C['orange'],fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig, 'fig08_data_audit', '数据完整性审计与已知训练mask尺寸差异',
         '来源：data_audit.json；所有mask为L模式、ID 0–7。37张尺寸差异没有擅自裁剪；单独resize仍可能造成语义对齐偏差，是复现限制。',
         ['cloud_reproduction/data_audit.json'])


def image_comparison():
    names = ['d_r_122_.jpg', 'd_r_129_.jpg', 'd_r_144_.jpg', 'd_r_166_.jpg']
    columns = [('原始输入', PROJECT/'dataset/test/images'), ('任务Reference', PROJECT/'dataset/test/reference'),
               ('最佳：epoch 71', EXTRACTED/'outputs/trained_best_ssim_model_SA_20261003/png'),
               ('最终：epoch 100', EXTRACTED/'outputs/trained_final_model_SA_20261003/png')]
    fig, axes = plt.subplots(4, 4, figsize=(12, 11.6))
    for row, name in enumerate(names):
        for col, (title, folder) in enumerate(columns):
            filename = name if col < 2 else Path(name).stem+'.png'
            source = folder/filename
            if not source.is_file():
                portable_folders = ['raw', 'reference', 'best_ssim', 'final']
                source = ROOT/'evidence/samples'/portable_folders[col]/filename
            with Image.open(source) as image:
                axes[row,col].imshow(image.convert('RGB').resize((256,256)))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(title,fontsize=11)
            if col==0:axes[row,col].set_ylabel(name,fontsize=9)
    save(fig, 'fig09_image_comparison', '可追溯样例：输入、Reference与重新训练输出',
         '来源：明确列出的4个测试文件与已校验训练输出；样例是按文件名选取，不代表完整测试集。显示resize只供视觉对比，不由此重新计算指标。',
         ['dataset/test', 'outputs/trained_best_ssim_model_SA_20261003', 'outputs/trained_final_model_SA_20261003'])


def step_losses(iterations):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    step = iterations.global_step_one_based.to_numpy()
    for col, color, label in [('loss_E',C['blue'],'E total'),('loss_S',C['orange'],'S total'),('loss_seg_sa',C['green'],'SA CE')]:
        series = iterations[col]
        ax = axes[0] if col=='loss_E' else axes[1]
        ax.plot(step,series,color=color,alpha=.16,lw=.4,rasterized=True)
        ax.plot(step,series.rolling(381,min_periods=1).median(),color=color,lw=1.4,label=label+'（381 step滚动中位数）')
    for ax in axes:
        ax.axvline(3810,color=C['gold'],ls='--',label='TDP开始：step 3811')
        ax.set_ylabel('Loss / 损失');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('A  增强损失：保留完整原始step曲线',loc='left')
    axes[1].set_title('B  两个分割网络的损失',loc='left');axes[1].set_xlabel('完成的训练step (1–38100)')
    save(fig,'fig10_step_loss','38,100个训练step：原始波动与明确标注的平滑曲线',
         '来源：TensorBoard完整scalar；浅线为原始点，粗线为381-step滚动中位数，不是统计置信区间。未进行下采样或替换异常点。',
         ['data/iterations.csv'])


def main():
    epochs = pd.read_csv(ROOT/'data/epochs.csv')
    samples = pd.read_csv(ROOT/'data/gpu_samples.csv')
    checks = pd.read_csv(ROOT/'data/monitor_checks.csv')
    chronology = pd.read_csv(ROOT/'data/checkpoints/chronology.csv')
    backups = chronology[chronology.event_type.eq('local_checkpoint_backup')]
    iterations = pd.read_csv(ROOT/'data/iterations.csv')
    completion_path = EXTRACTED/'training_runs/from_scratch_20261003/completed.json'
    if not completion_path.exists():completion_path = ROOT/'evidence/run/completed.json'
    completion = json.loads(completion_path.read_text())
    hours = completion['elapsed_seconds']/3600
    workflow();losses(epochs);validation(epochs);learning_rates(epochs)
    gpu(samples,hours);runtime(epochs);timeline(epochs,checks,backups,hours)
    data_audit();image_comparison();step_losses(iterations)
    (ROOT/'figure_catalog.json').write_text(json.dumps(CATALOG,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved {} figures in PNG, PDF and SVG'.format(len(CATALOG)))


if __name__ == '__main__':
    main()
