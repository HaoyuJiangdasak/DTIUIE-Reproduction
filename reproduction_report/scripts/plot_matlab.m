% Rebuild editable MATLAB figures from the recorded DTIUIE training logs.
% No inference or training is performed; every curve uses the report CSVs.
% Run: matlab -batch "run('.../scripts/plot_matlab.m')"
reportRoot = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(reportRoot, 'tools', 'matlab');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fprintf('DTIUIE MATLAB visualization started: %s\n', char(datetime('now')));
fprintf('MATLAB: %s\n', version);
fprintf('MATLAB license available: %d\n', license('test', 'MATLAB'));
fprintf('Report root: %s\n', reportRoot);
E = readtable(fullfile(reportRoot, 'data', 'epochs.csv'), 'VariableNamingRule', 'preserve');
G = readtable(fullfile(reportRoot, 'data', 'gpu_samples.csv'), 'VariableNamingRule', 'preserve');
summaryText = fileread(fullfile(reportRoot, 'data', 'summary.json'));
summaryText = strrep(summaryText, char(65279), ''); % Accept an optional UTF-8 BOM.
runSummary = jsondecode(summaryText);
completedElapsedHours = runSummary.elapsed_seconds_completed_record / 3600;
assert(isfinite(completedElapsedHours) && completedElapsedHours > 0, ...
    'Summary must contain the recorded elapsed time at actual completion.');
fprintf('Training completion marker: %.10f h (summary completed record)\n', completedElapsedHours);
requiredE = {'epoch','loss_E','loss_S','ssim','miou'};
requiredG = {'elapsed_hours','gpu_util_pct','vram_used_mib','vram_total_mib'};
assert(all(ismember(requiredE, E.Properties.VariableNames)), 'Missing epoch columns.');
assert(all(ismember(requiredG, G.Properties.VariableNames)), 'Missing GPU columns.');
E = sortrows(E, 'epoch');
G = sortrows(G, 'elapsed_hours');
assert(numel(unique(E.epoch)) == height(E), 'Epoch rows must be unique.');
assert(numel(unique(G.elapsed_hours)) == height(G), 'GPU timestamps must be unique.');
fprintf('Recorded epochs: %d; GPU samples: %d\n', height(E), height(G));
availableFonts = listfonts;
fontName = 'Arial';
for candidate = {'Microsoft YaHei','SimHei','Microsoft YaHei UI'}
    if any(strcmpi(availableFonts, candidate{1})), fontName = candidate{1}; break; end
end
fprintf('Figure font: %s\n', fontName);
blue = [0.12 0.35 0.67];
teal = [0.00 0.49 0.48];
orange = [0.82 0.33 0.14];
purple = [0.43 0.29 0.64];
bestEpoch = 71; % Recorded code-selected best checkpoint; not a metric argmax claim.
warmupEpoch = 10;

fTrain = figure('Visible','off','Color','w','Position',[100 100 1400 960], ...
    'Name','DTIUIE training and validation (editable)', 'NumberTitle','off');
layout = tiledlayout(fTrain,2,2,'TileSpacing','compact','Padding','compact');
metricNames = {'loss_E','loss_S','ssim','miou'};
metricTitles = {'(a) 增强网络训练损失 E','(b) 分割网络训练损失 S', ...
    '(c) 验证集结构相似性 SSIM','(d) 验证集平均交并比 mIoU'};
yNames = {'Loss E（官方日志）','Loss S（官方日志）','SSIM（官方日志）','mIoU（官方日志）'};
colors = [blue; orange; teal; purple];
for k = 1:4
    ax = nexttile(layout);
    plot(ax,E.epoch,E.(metricNames{k}),'-','Color',colors(k,:), 'LineWidth',1.7);
    hold(ax,'on');
    xline(ax,warmupEpoch,'--','前 10 轮阶段边界','Color',[0.47 0.49 0.52], ...
        'LabelHorizontalAlignment','left','LabelVerticalAlignment','top', ...
        'FontName',fontName,'FontSize',9,'HandleVisibility','off');
    xline(ax,bestEpoch,':','代码选定最佳轮 71','Color',[0.70 0.38 0.12], ...
        'LabelHorizontalAlignment','left','LabelVerticalAlignment','bottom', ...
        'FontName',fontName,'FontSize',9,'HandleVisibility','off');
    bi = find(E.epoch == bestEpoch,1);
    if ~isempty(bi) && isfinite(E.(metricNames{k})(bi))
        plot(ax,E.epoch(bi),E.(metricNames{k})(bi),'o','MarkerSize',7, ...
            'MarkerEdgeColor','w','MarkerFaceColor',colors(k,:), 'HandleVisibility','off');
    end
    xlabel(ax,'训练轮次 / Epoch','FontName',fontName);
    ylabel(ax,yNames{k},'FontName',fontName);
    title(ax,metricTitles{k},'FontName',fontName,'FontWeight','bold');
    styleAxes(ax,fontName);
    xlim(ax,[min(E.epoch) max(E.epoch)]);
end
title(layout,'DTIUIE 完整训练与轮末验证 | 100 epochs', ...
    'FontName',fontName,'FontSize',19,'FontWeight','bold');
xlabel(layout,'来源：data/epochs.csv；保留原始逐轮数据，不作平滑；loss 为训练日志的官方统计口径。', ...
    'FontName',fontName,'FontSize',10);
exportFigure(fTrain,outDir,'matlab_training_validation');

% Break lines across observation gaps; no values are fabricated between samples.
dt = diff(G.elapsed_hours);
normalCadence = median(dt(dt > 0),'omitnan');
if isempty(normalCadence) || ~isfinite(normalCadence), normalCadence = 5/60; end
gapThresholdH = 7/60; % Shared report convention: break observations > 420 s apart.
gapAfter = find(dt > gapThresholdH);
fprintf('GPU median sampling cadence: %.2f min; gaps > %.2f min: %d\n', ...
    normalCadence*60, gapThresholdH*60, numel(gapAfter));
fGPU = figure('Visible','off','Color','w','Position',[100 100 1440 960], ...
    'Name','DTIUIE observed GPU history (editable)','NumberTitle','off');
gpuLayout = tiledlayout(fGPU,2,1,'TileSpacing','compact','Padding','compact');
ax = nexttile(gpuLayout);
plotObserved(ax,G.elapsed_hours,G.gpu_util_pct,gapAfter,blue);
ylim(ax,[0 105]);
ylabel(ax,'GPU 利用率 / %','FontName',fontName);
title(ax,'(a) GPU 利用率：每次 SSH 巡检的实际采样','FontName',fontName,'FontWeight','bold');
styleAxes(ax,fontName);
ax = nexttile(gpuLayout);
plotObserved(ax,G.elapsed_hours,G.vram_used_mib/1024,gapAfter,teal);
hold(ax,'on');
plot(ax,G.elapsed_hours,G.vram_total_mib/1024,'--','Color',[0.46 0.47 0.50], ...
    'LineWidth',1.2,'DisplayName','显存容量');
ylabel(ax,'GPU 显存 / GiB','FontName',fontName);
xlabel(ax,'距训练开始的时间 / h','FontName',fontName);
title(ax,'(b) 实际显存占用与显存容量','FontName',fontName,'FontWeight','bold');
legend(ax,{'实际显存采样','显存容量'},'Location','southeast','FontName',fontName);
styleAxes(ax,fontName);
if any(isfinite(G.vram_total_mib)), ylim(ax,[0 max(G.vram_total_mib/1024)*1.08]); end
allGPUAxes = findall(fGPU,'Type','axes');
for k = 1:numel(allGPUAxes)
    xlim(allGPUAxes(k),[0 max(G.elapsed_hours)]);
    if ismember('elapsed_hours',E.Properties.VariableNames)
        ei = find(E.epoch == warmupEpoch,1);
        if ~isempty(ei) && isfinite(E.elapsed_hours(ei))
            xline(allGPUAxes(k),E.elapsed_hours(ei),'--','第 10 轮完成', ...
                'Color',[0.47 0.49 0.52],'FontName',fontName,'FontSize',9, ...
                'LabelHorizontalAlignment','left','HandleVisibility','off');
        end
    end
    xline(allGPUAxes(k),completedElapsedHours,':','训练结束', ...
        'Color',[0.65 0.28 0.18],'FontName',fontName,'FontSize',9, ...
        'LabelHorizontalAlignment','left','HandleVisibility','off');
end
gpuName = 'GPU';
if ismember('gpu_name',G.Properties.VariableNames)
    nameValues = string(G.gpu_name);
    ni = find(~ismissing(nameValues) & strlength(nameValues)>0,1);
    if ~isempty(ni), gpuName = char(nameValues(ni)); end
end
title(gpuLayout,['DTIUIE 云端资源记录 | ' gpuName], ...
    'FontName',fontName,'FontSize',19,'FontWeight','bold');
xlabel(gpuLayout,sprintf('来源：data/gpu_samples.csv；%d 个离散样本；%d 处 >7 min 观测缺口断线；训练结束线右侧含导出/空闲。',height(G),numel(gapAfter)), ...
    'FontName',fontName,'FontSize',10);
exportFigure(fGPU,outDir,'matlab_gpu_utilization_memory');

if ismember('temp_c',G.Properties.VariableNames) && any(isfinite(G.temp_c))
    fTemp = figure('Visible','off','Color','w','Position',[100 100 1300 650], ...
        'Name','DTIUIE observed GPU temperature (editable)','NumberTitle','off');
    ax = axes(fTemp);
    plotObserved(ax,G.elapsed_hours,G.temp_c,gapAfter,orange);
    xlabel(ax,'距训练开始的时间 / h','FontName',fontName);
    ylabel(ax,'GPU 温度 / °C','FontName',fontName);
    title(ax,'DTIUIE GPU 温度：真实 SSH 巡检采样','FontName',fontName,'FontSize',17);
    styleAxes(ax,fontName);
    xlim(ax,[0 max(G.elapsed_hours)]);
    xline(ax,completedElapsedHours,':','训练结束；后续为空闲采样', ...
        'Color',[0.45 0.28 0.18],'FontName',fontName,'FontSize',9, ...
        'LabelHorizontalAlignment','left','HandleVisibility','off');
    subtitle(ax,sprintf('%d 个真实样本；%d 处 >7 min 观测缺口断线；不补造缺口数据',height(G),numel(gapAfter)), ...
        'FontName',fontName,'FontSize',10);
    exportFigure(fTemp,outDir,'matlab_gpu_temperature');
end

if ismember('power_w',G.Properties.VariableNames) && ~any(isfinite(G.power_w))
    fprintf('Power data absent: no power curve produced.\n');
end
save(fullfile(outDir,'matlab_recorded_tables.mat'),'E','G','bestEpoch','warmupEpoch', ...
    'normalCadence','gapThresholdH','gapAfter','fontName','completedElapsedHours');
metadata = struct('matlab_version',version,'epoch_rows',height(E),'gpu_rows',height(G), ...
    'best_epoch_marker',bestEpoch,'warmup_epoch_marker',warmupEpoch, ...
    'font_name',fontName,'median_gpu_sample_interval_minutes',normalCadence*60, ...
    'gpu_observation_gap_threshold_minutes',gapThresholdH*60, ...
    'gpu_observation_gap_count',numel(gapAfter), ...
    'training_completion_marker_elapsed_hours',completedElapsedHours, ...
    'training_completion_marker_source','data/summary.json:elapsed_seconds_completed_record / 3600', ...
    'training_completion_marker_utc',runSummary.completed_at_utc, ...
    'figure_data_source','CSV extracted from recorded run logs; no new model evaluation', ...
    'png_dpi',300,'pdf_content_type','vector','finished_local',char(datetime('now')));
fid = fopen(fullfile(outDir,'matlab_execution_metadata.json'),'w','n','UTF-8');
assert(fid ~= -1,'Cannot write MATLAB metadata.');
fprintf(fid,'%s\n',jsonencode(metadata,'PrettyPrint',true));
fclose(fid);
close(fTrain); close(fGPU);
if exist('fTemp','var'), close(fTemp); end
fprintf('DTIUIE MATLAB visualization finished successfully.\n');

function styleAxes(ax,fontName)
    set(ax,'FontName',fontName,'FontSize',11,'LineWidth',0.8, ...
        'Box','off','Color','w','GridColor',[0.53 0.56 0.60], ...
        'GridAlpha',0.18,'XColor',[0.20 0.22 0.25],'YColor',[0.20 0.22 0.25]);
    grid(ax,'on');
end

function plotObserved(ax,x,y,gapAfter,color)
    % The point markers are real measurements, and gaps are broken with NaN.
    xp = x(:); yp = y(:);
    for j = numel(gapAfter):-1:1
        gi = gapAfter(j);
        xp = [xp(1:gi); NaN; xp(gi+1:end)];
        yp = [yp(1:gi); NaN; yp(gi+1:end)];
    end
    plot(ax,xp,yp,'-o','Color',color,'LineWidth',1.3,'MarkerSize',3, ...
        'MarkerFaceColor',color,'MarkerEdgeColor',color);
end

function exportFigure(fig,outDir,baseName)
    % Explicit colors override a user's desktop dark theme for readable exports.
    textObjects = findall(fig,'Type','text');
    set(textObjects,'Color',[0.15 0.19 0.23]);
    legendObjects = findall(fig,'Type','legend');
    set(legendObjects,'Color','w','TextColor',[0.15 0.19 0.23], ...
        'EdgeColor',[0.65 0.68 0.72]);
    layouts = findall(fig,'Type','tiledlayout');
    for i = 1:numel(layouts)
        layouts(i).Title.Color = [0.15 0.19 0.23];
        layouts(i).XLabel.Color = [0.25 0.29 0.33];
    end
    drawnow;
    savefig(fig,fullfile(outDir,[baseName '.fig']));
    exportgraphics(fig,fullfile(outDir,[baseName '.png']),'Resolution',300);
    exportgraphics(fig,fullfile(outDir,[baseName '.pdf']),'ContentType','vector');
    fprintf('Exported %s.{fig,png,pdf}\n',baseName);
end
