% Reopen the saved native FIGs and confirm object-level editability.
outDir = fileparts(mfilename('fullpath'));
files = dir(fullfile(outDir,'matlab_*.fig'));
assert(numel(files) == 3, 'Three saved native MATLAB figures are required.');
checks = struct([]);
for i = 1:numel(files)
    fig = openfig(fullfile(files(i).folder,files(i).name),'invisible');
    ax = findall(fig,'Type','axes');
    lines = findall(fig,'Type','line');
    assert(~isempty(ax) && ~isempty(lines),'Saved FIG lacks editable axes/line objects.');
    maxPoints = 0;
    for j = 1:numel(lines)
        maxPoints = max(maxPoints, sum(isfinite(lines(j).XData) & isfinite(lines(j).YData)));
    end
    if strcmp(files(i).name,'matlab_training_validation.fig')
        assert(numel(ax) == 4 && maxPoints == 100,'Training FIG must retain four axes and 100 epoch points.');
    end
    checks(i).file = files(i).name;
    checks(i).editable_axes = numel(ax);
    checks(i).editable_line_objects = numel(lines);
    checks(i).largest_observed_point_count = maxPoints;
    checks(i).reopen_success = true;
    fprintf('REOPEN_OK %s; axes=%d; line_objects=%d; max_finite_points=%d\n', ...
        files(i).name,numel(ax),numel(lines),maxPoints);
    close(fig);
end
fid = fopen(fullfile(outDir,'matlab_fig_verification.json'),'w','n','UTF-8');
fprintf(fid,'%s\n',jsonencode(checks,'PrettyPrint',true));
fclose(fid);
fprintf('All saved FIGs reopen successfully with editable native objects.\n');
