% 用途：导出训练结束时E神经元的朝向选择性、感受野、放电率和逐刺激响应数据及图像。
function files = save_pcb_final_diagnostics(result,run_dir,stem)

d = result.final_diagnostics;
csv_dir = fullfile(run_dir,'csv','final_diagnostics');
png_dir = fullfile(run_dir,'png','final_diagnostics');
osi_dir = fullfile(png_dir,'OSI');
rf_dir = fullfile(png_dir,'RF');
firing_dir = fullfile(png_dir,'Firing');
ensure_directories({csv_dir,osi_dir,rf_dir,firing_dir});

n_E = numel(d.OSI_E);
cell_table = table((1:n_E)',double(d.OSI_E), ...
    double(d.preferred_orientation_deg_E), ...
    double(d.preferred_spatial_frequency_E), ...
    double(d.mean_rate_E_Hz), ...
    double(d.active_stimulus_fraction_E), ...
    'VariableNames',{'E_neuron','OSI','preferred_orientation_deg', ...
    'preferred_spatial_frequency','mean_firing_rate_Hz', ...
    'active_stimulus_fraction'});
files.cell_csv = fullfile(csv_dir,[stem '_final_E_cells.csv']);
files.response_csv = fullfile(csv_dir, ...
    [stem '_final_E_response_counts.csv']);
writetable(cell_table,files.cell_csv);
writematrix(d.response_counts_E,files.response_csv);

files.osi_png = fullfile(osi_dir,[stem '_final_E_OSI.png']);
plot_osi(d,result.condition.id,files.osi_png);
files.rf_png = fullfile(rf_dir,[stem '_final_E_RF.png']);
plot_rf(d.RF_E,result.cfg,result.condition.id,files.rf_png);
files.firing_png = fullfile(firing_dir,[stem '_final_E_firing.png']);
plot_firing(d,result.cfg,result.condition.id,files.firing_png);
end

function plot_osi(d,condition_id,output_file)
dark = [0.04 0.14 0.25];
valid = double(d.OSI_E(isfinite(d.OSI_E)));
fig = figure('Color','w','Visible','off','Position',[100 100 900 560]);
ax = axes(fig);
if isempty(valid)
    axis(ax,[0 1 0 1]);
    text(ax,0.5,0.5,'No valid OSI samples','HorizontalAlignment','center', ...
        'FontSize',14,'Color',dark);
else
    histogram(ax,valid,0:0.05:1,'Normalization','probability', ...
        'FaceColor',[0.20 0.48 0.72],'EdgeColor','white');
    xlim(ax,[0 1]);
    text(ax,0.03,0.96,sprintf(['Mean %.3f   Median %.3f\n' ...
        'OSI >= 0.5: %.1f%%   Valid: %.1f%%'],d.mean_OSI_E, ...
        d.median_OSI_E,100*d.fraction_OSI_ge_0_5, ...
        100*d.valid_OSI_fraction),'Units','normalized', ...
        'VerticalAlignment','top','BackgroundColor','white', ...
        'EdgeColor',[0.75 0.75 0.75],'Color',dark);
end
xlabel(ax,'Orientation selectivity index');
ylabel(ax,'Fraction of E neurons');
title(ax,sprintf('Final E-cell OSI - condition %s',condition_id));
style_axis(ax);
exportgraphics(fig,output_file,'Resolution',240,'BackgroundColor','white');
close(fig);
end

function plot_rf(RF,cfg,condition_id,output_file)
RF = double(RF);
scale = max(abs(RF),[],2);
scale(scale == 0) = 1;
RF = RF./scale;
mosaic = make_rf_mosaic(RF,round(sqrt(cfg.N_pixels)));
fig = figure('Color','w','Visible','off','Position',[80 60 1250 1050]);
ax = axes(fig);
image_handle = imagesc(ax,mosaic);
image_handle.AlphaData = ~isnan(mosaic);
axis(ax,'image','off'); clim(ax,[-1 1]); ax.Color = 'w';
colormap(fig,blue_white_red(256));
title(ax,sprintf(['Final E receptive fields - condition %s\n' ...
    'Each RF normalized independently'],condition_id), ...
    'Color',[0.04 0.14 0.25]);
exportgraphics(fig,output_file,'Resolution',260,'BackgroundColor','white');
close(fig);
end

function plot_firing(d,cfg,condition_id,output_file)
rates = double(d.response_counts_E)/cfg.T;
mean_rates = double(d.mean_rate_E_Hz);
[~,order] = sort(mean_rates,'descend');
fig = figure('Color','w','Visible','off','Position',[80 80 1250 560]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax = nexttile(tl);
imagesc(ax,rates(order,:));
xlabel(ax,'Evaluation image patch'); ylabel(ax,'E neuron (sorted)');
title(ax,'Firing-rate response (Hz)'); colorbar(ax);
style_axis(ax);
ax = nexttile(tl);
histogram(ax,mean_rates,'BinMethod','fd','FaceColor',[0.20 0.48 0.72], ...
    'EdgeColor','white');
xlabel(ax,'Mean firing rate per E neuron (Hz)'); ylabel(ax,'E-neuron count');
title(ax,sprintf('Mean %.3f Hz; silent %.1f%%',mean(mean_rates), ...
    100*d.silent_neuron_fraction_E));
style_axis(ax);
title(tl,sprintf('Final E-cell firing - condition %s',condition_id), ...
    'FontWeight','bold','Color',[0.04 0.14 0.25]);
exportgraphics(fig,output_file,'Resolution',240,'BackgroundColor','white');
close(fig);
end

function mosaic = make_rf_mosaic(RF,input_side)
n_cells = size(RF,1);
n_cols = ceil(sqrt(n_cells));
n_rows = ceil(n_cells/n_cols);
gap = 1;
mosaic = nan(n_rows*(input_side+gap)-gap, ...
    n_cols*(input_side+gap)-gap);
for i = 1:n_cells
    row = floor((i-1)/n_cols);
    col = mod(i-1,n_cols);
    rows = row*(input_side+gap)+(1:input_side);
    cols = col*(input_side+gap)+(1:input_side);
    mosaic(rows,cols) = reshape(RF(i,:),input_side,input_side);
end
end

function cmap = blue_white_red(n)
half = floor(n/2);
blue = [linspace(0,1,half)' linspace(0,1,half)' ones(half,1)];
red_count = n-half;
red = [ones(red_count,1) linspace(1,0,red_count)' ...
    linspace(1,0,red_count)'];
cmap = [blue;red];
end

function style_axis(ax)
dark = [0.04 0.14 0.25];
set(ax,'FontName','Arial','FontSize',10,'LineWidth',1,'Color','w', ...
    'XColor',[0.08 0.08 0.08],'YColor',[0.08 0.08 0.08]);
grid(ax,'on'); ax.GridAlpha = 0.16;
ax.Title.Color = dark;
ax.XLabel.Color = dark;
ax.YLabel.Color = dark;
end

function ensure_directories(directories)
for i = 1:numel(directories)
    if ~exist(directories{i},'dir')
        mkdir(directories{i});
    end
end
end
