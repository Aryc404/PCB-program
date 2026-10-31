% 用途：汇总训练后的宽松与紧密E/I平衡指标，导出数值表、输入电流示例和各条件对比图。
function outputs = save_pcb_ei_balance_summary(all_task_outputs, ...
        network_scales,condition_ids,runs_per_scale,csv_dir,png_dir, ...
        stem,mode)

records = collect_records(all_task_outputs,network_scales, ...
    condition_ids,runs_per_scale);
if isempty(records)
    error('No E/I balance records were available for plotting.');
end

loose_table = make_loose_table(records);
scatter_table = make_scatter_table(records);
tight_table = make_tight_table(records);
outputs.loose_csv = string(fullfile(csv_dir, ...
    [stem '_ei_balance_loose.csv']));
outputs.tight_csv = string(fullfile(csv_dir, ...
    [stem '_ei_balance_tight.csv']));
outputs.scatter_csv = string(fullfile(csv_dir, ...
    [stem '_ei_balance_loose_scatter.csv']));
writetable(loose_table,char(outputs.loose_csv));
writetable(scatter_table,char(outputs.scatter_csv));
writetable(tight_table,char(outputs.tight_csv));

plot_dir = fullfile(png_dir,'ei_balance');
ensure_directory(plot_dir);
outputs.plot_files = strings(0,1);
for condition_index = 1:numel(condition_ids)
    condition_id = char(condition_ids(condition_index));
    selected = records(strcmp({records.condition},condition_id));
    outputs.plot_files(end+1,1) = plot_example_traces( ...
        selected,plot_dir,mode);
    outputs.plot_files(end+1,1) = plot_loose_errorbars( ...
        selected,plot_dir,mode);
    outputs.plot_files(end+1,1) = plot_loose_scatter( ...
        selected,plot_dir,mode);
    outputs.plot_files(end+1,1) = plot_tight_curves( ...
        selected,plot_dir,mode);
    outputs.plot_files(end+1,1) = plot_tight_population( ...
        selected,plot_dir,mode);
end
end

function records = collect_records(all_task_outputs,network_scales, ...
        condition_ids,runs_per_scale)
template = struct('condition','','display_name','','network_scale',NaN, ...
    'run_index',NaN,'balance',struct());
records = repmat(template,0,1);
for scale_index = 1:numel(network_scales)
    outputs = all_task_outputs{scale_index};
    expected_count = numel(condition_ids)*runs_per_scale(scale_index);
    if numel(outputs) ~= expected_count
        error('Unexpected number of task outputs for cn=%d.', ...
            network_scales(scale_index));
    end
    for output_index = 1:numel(outputs)
        result = outputs{output_index}.result;
        if isempty(result) || ~isfield(result,'ei_balance')
            error('A completed task lacks result.ei_balance.');
        end
        record = template;
        record.condition = char(result.condition.id);
        record.display_name = char(result.condition.display_name);
        record.network_scale = result.cfg.network_scale;
        record.run_index = outputs{output_index}.run_index;
        record.balance = result.ei_balance;
        records(end+1,1) = record;
    end
end
[~,order] = sortrows([[records.network_scale]' [records.run_index]']);
records = records(order);
end

function T = make_loose_table(records)
rows = numel(records)*3;
condition = strings(rows,1);
network_scale = zeros(rows,1);
run_index = zeros(rows,1);
input_class = strings(rows,1);
n_neurons = zeros(rows,1);
mean_exc = nan(rows,1); sd_exc = nan(rows,1);
mean_inh_signed = nan(rows,1); sd_inh_signed = nan(rows,1);
mean_inh_magnitude = nan(rows,1); sd_inh_magnitude = nan(rows,1);
mean_total = nan(rows,1); sd_total = nan(rows,1);
row = 0;
for record_index = 1:numel(records)
    b = records(record_index).balance;
    for class_index = 1:3
        row = row + 1;
        condition(row) = string(records(record_index).condition);
        network_scale(row) = records(record_index).network_scale;
        run_index(row) = records(record_index).run_index;
        input_class(row) = b.class_names(class_index);
        n_neurons(row) = numel(b.loose.neuron_ids);
        mean_exc(row) = b.loose.group_mean_V_exc(class_index);
        sd_exc(row) = b.loose.group_sd_V_exc(class_index);
        mean_inh_signed(row) = ...
            b.loose.group_mean_V_inh_signed(class_index);
        sd_inh_signed(row) = ...
            b.loose.group_sd_V_inh_signed(class_index);
        mean_inh_magnitude(row) = ...
            b.loose.group_mean_V_inh_magnitude(class_index);
        sd_inh_magnitude(row) = ...
            b.loose.group_sd_V_inh_magnitude(class_index);
        mean_total(row) = b.loose.group_mean_V_total(class_index);
        sd_total(row) = b.loose.group_sd_V_total(class_index);
    end
end
T = table(condition,network_scale,run_index,input_class,n_neurons, ...
    mean_exc,sd_exc,mean_inh_signed,sd_inh_signed, ...
    mean_inh_magnitude,sd_inh_magnitude,mean_total,sd_total);
T = sortrows(T,{'condition','network_scale','run_index'});
end

function T = make_scatter_table(records)
rows = 0;
for i = 1:numel(records)
    rows = rows + 3*numel(records(i).balance.loose_scatter.neuron_ids);
end
condition = strings(rows,1);
network_scale = zeros(rows,1);
run_index = zeros(rows,1);
neuron_id = zeros(rows,1);
input_class = strings(rows,1);
patch_index = zeros(rows,1);
patch_correlation = nan(rows,1);
mean_exc = nan(rows,1);
mean_inh_magnitude = nan(rows,1);
mean_total = nan(rows,1);
row = 0;
for record_index = 1:numel(records)
    b = records(record_index).balance;
    s = b.loose_scatter;
    n_neurons = numel(s.neuron_ids);
    for class_index = 1:3
        target = row + (1:n_neurons);
        condition(target) = string(records(record_index).condition);
        network_scale(target) = records(record_index).network_scale;
        run_index(target) = records(record_index).run_index;
        neuron_id(target) = s.neuron_ids;
        input_class(target) = b.class_names(class_index);
        patch_index(target) = s.patch_indices(:,class_index);
        patch_correlation(target) = s.patch_correlations(:,class_index);
        mean_exc(target) = s.time_mean_V_exc(:,class_index);
        mean_inh_magnitude(target) = ...
            s.time_mean_V_inh_magnitude(:,class_index);
        mean_total(target) = s.time_mean_V_total(:,class_index);
        row = row + n_neurons;
    end
end
T = table(condition,network_scale,run_index,neuron_id,input_class, ...
    patch_index,patch_correlation,mean_exc,mean_inh_magnitude,mean_total);
T = sortrows(T,{'condition','network_scale','run_index','input_class', ...
    'neuron_id'});
end

function T = make_tight_table(records)
n = numel(records);
condition = strings(n,1);
network_scale = zeros(n,1); run_index = zeros(n,1);
example_neuron = zeros(n,1); n_images = zeros(n,1);
zero_lag_mean = nan(n,1); zero_lag_ci95 = nan(n,1);
peak_mean_correlation = nan(n,1); peak_lag_ms = nan(n,1);
population_n = zeros(n,1);
population_zero_lag_mean = nan(n,1); population_zero_lag_sd = nan(n,1);
population_peak_correlation_mean = nan(n,1);
population_peak_correlation_sd = nan(n,1);
population_peak_lag_ms_mean = nan(n,1);
population_peak_lag_ms_sd = nan(n,1);
for i = 1:n
    b = records(i).balance;
    condition(i) = string(records(i).condition);
    network_scale(i) = records(i).network_scale;
    run_index(i) = records(i).run_index;
    example_neuron(i) = b.tight.neuron_id;
    n_images(i) = numel(b.tight.patch_indices);
    zero_lag_mean(i) = b.tight.zero_lag_mean;
    zero_lag_ci95(i) = b.tight.zero_lag_ci95;
    peak_mean_correlation(i) = b.tight.peak_mean_correlation;
    peak_lag_ms(i) = b.tight.peak_lag_ms;
    population_n(i) = numel(b.tight_population.neuron_ids);
    population_zero_lag_mean(i) = b.tight_population.zero_lag_mean;
    population_zero_lag_sd(i) = b.tight_population.zero_lag_sd;
    population_peak_correlation_mean(i) = ...
        b.tight_population.peak_correlation_mean;
    population_peak_correlation_sd(i) = ...
        b.tight_population.peak_correlation_sd;
    population_peak_lag_ms_mean(i) = ...
        b.tight_population.peak_lag_ms_mean;
    population_peak_lag_ms_sd(i) = ...
        b.tight_population.peak_lag_ms_sd;
end
T = table(condition,network_scale,run_index,example_neuron,n_images, ...
    zero_lag_mean,zero_lag_ci95,peak_mean_correlation,peak_lag_ms, ...
    population_n,population_zero_lag_mean,population_zero_lag_sd, ...
    population_peak_correlation_mean,population_peak_correlation_sd, ...
    population_peak_lag_ms_mean,population_peak_lag_ms_sd);
T = sortrows(T,{'condition','network_scale','run_index'});
end

function output_file = plot_example_traces(records,plot_dir,mode)
records = first_run_per_scale(records);
n = numel(records);
fig = figure('Visible','off','Color','white','Position',[50 50 1500 350*n]);
tl = tiledlayout(n,3,'TileSpacing','compact','Padding','compact');
colors = struct('exc',[0.10 0.42 0.72],'inh',[0.82 0.25 0.20], ...
    'total',[0.08 0.08 0.08]);
for r = 1:n
    b = records(r).balance;
    for class_index = 1:3
        ax = nexttile(tl); hold(ax,'on');
        plot(ax,b.time_ms,b.example.V_exc(class_index,:), ...
            'Color',colors.exc,'LineWidth',1.1,'DisplayName','V_{exc}');
        plot(ax,b.time_ms,b.example.V_inh_signed(class_index,:), ...
            'Color',colors.inh,'LineWidth',1.1,'DisplayName','V_{inh}');
        plot(ax,b.time_ms,b.example.V_total(class_index,:), ...
            'Color',colors.total,'LineWidth',1.25,'DisplayName','V_{total}');
        yline(ax,b.example.threshold,'--','Color',[0.55 0.55 0.55], ...
            'LineWidth',1,'DisplayName','Learned threshold');
        title(ax,sprintf('c_n=%d | %s | r=%.3f', ...
            records(r).network_scale,char(b.class_names(class_index)), ...
            b.example.patch_correlations(class_index)), ...
            'Color',[0.05 0.05 0.05]);
        xlabel(ax,'Time (ms)'); ylabel(ax,'Voltage contribution');
        style_axis(ax);
        if r == 1 && class_index == 1
            legend(ax,'Location','best','Color','white', ...
                'TextColor',[0.05 0.05 0.05]);
        end
    end
end
title(tl,sprintf('%s: loose-balance example traces after batch 1200', ...
    condition_label(records(1))),'Color',[0.02 0.02 0.02], ...
    'FontWeight','bold');
output_file = make_plot_path(plot_dir,records(1).condition, ...
    'loose_example_traces',mode);
exportgraphics(fig,char(output_file),'Resolution',240, ...
    'BackgroundColor','white'); close(fig);
end

function output_file = plot_loose_errorbars(records,plot_dir,mode)
records = first_run_per_scale(records);
n = numel(records);
fig = figure('Visible','off','Color','white','Position',[60 60 480*n 520]);
tl = tiledlayout(1,n,'TileSpacing','compact','Padding','compact');
colors = {[0.10 0.42 0.72],[0.82 0.25 0.20],[0.08 0.08 0.08]};
for r = 1:n
    b = records(r).balance;
    ax = nexttile(tl); hold(ax,'on');
    values = [b.loose.group_mean_V_exc; ...
        b.loose.group_mean_V_inh_signed;b.loose.group_mean_V_total];
    errors = [b.loose.group_sd_V_exc; ...
        b.loose.group_sd_V_inh_signed;b.loose.group_sd_V_total];
    offsets = [-0.08 0 0.08];
    labels = {'V_{exc}','V_{inh}','V_{total}'};
    for series = 1:3
        errorbar(ax,(1:3)+offsets(series),values(series,:), ...
            errors(series,:),'-o','Color',colors{series}, ...
            'LineWidth',1.5,'MarkerFaceColor','white', ...
            'DisplayName',labels{series});
    end
    xlim(ax,[0.65 3.35]); xticks(ax,1:3);
    xticklabels(ax,{'Low','Middle','High'});
    xlabel(ax,'RF-patch correlation class');
    ylabel(ax,'Time-mean voltage contribution');
    title(ax,sprintf('c_n=%d | mean +/- SD, n=%d', ...
        records(r).network_scale,numel(b.loose.neuron_ids)), ...
        'Color',[0.05 0.05 0.05]);
    style_axis(ax);
    if r == 1
        legend(ax,'Location','best','Color','white', ...
            'TextColor',[0.05 0.05 0.05]);
    end
end
title(tl,sprintf('%s: loose balance after batch 1200', ...
    condition_label(records(1))),'Color',[0.02 0.02 0.02], ...
    'FontWeight','bold');
output_file = make_plot_path(plot_dir,records(1).condition, ...
    'loose_errorbar',mode);
exportgraphics(fig,char(output_file),'Resolution',240, ...
    'BackgroundColor','white'); close(fig);
end

function output_file = plot_loose_scatter(records,plot_dir,mode)
records = first_run_per_scale(records);
n = numel(records);
fig = figure('Visible','off','Color','white','Position',[50 50 1500 350*n]);
tl = tiledlayout(n,3,'TileSpacing','compact','Padding','compact');
for r = 1:n
    b = records(r).balance;
    s = b.loose_scatter;
    for class_index = 1:3
        ax = nexttile(tl); hold(ax,'on');
        x = double(s.time_mean_V_inh_magnitude(:,class_index));
        y_exc = double(s.time_mean_V_exc(:,class_index));
        y_total = double(s.time_mean_V_total(:,class_index));
        scatter(ax,x,y_exc,24,[0.10 0.42 0.72],'o','filled', ...
            'MarkerFaceAlpha',0.65,'DisplayName','Excitation');
        scatter(ax,x,y_total,24,[0.10 0.10 0.10],'s','filled', ...
            'MarkerFaceAlpha',0.65,'DisplayName','Total');
        add_fit_line(ax,x,y_exc,[0.10 0.42 0.72]);
        add_fit_line(ax,x,y_total,[0.10 0.10 0.10]);
        xlabel(ax,'Mean inhibitory magnitude (-V_{inh})');
        ylabel(ax,'Mean voltage contribution');
        title(ax,sprintf('c_n=%d | %s',records(r).network_scale, ...
            char(b.class_names(class_index))),'Color',[0.05 0.05 0.05]);
        style_axis(ax);
        if r == 1 && class_index == 1
            legend(ax,'Location','best','Color','white', ...
                'TextColor',[0.05 0.05 0.05]);
        end
    end
end
title(tl,sprintf('%s: loose-balance population relationships', ...
    condition_label(records(1))),'Color',[0.02 0.02 0.02], ...
    'FontWeight','bold');
output_file = make_plot_path(plot_dir,records(1).condition, ...
    'loose_population_scatter',mode);
exportgraphics(fig,char(output_file),'Resolution',240, ...
    'BackgroundColor','white'); close(fig);
end

function output_file = plot_tight_curves(records,plot_dir,mode)
records = first_run_per_scale(records);
n = numel(records);
fig = figure('Visible','off','Color','white','Position',[60 60 480*n 500]);
tl = tiledlayout(1,n,'TileSpacing','compact','Padding','compact');
for r = 1:n
    b = records(r).balance;
    ax = nexttile(tl); hold(ax,'on');
    curves = double(b.tight.correlation_curves);
    lags = double(b.tight.lag_ms);
    if any(isfinite(curves(:)))
        for image_index = 1:size(curves,1)
            plot(ax,lags,curves(image_index,:),'-', ...
                'Color',[0.78 0.78 0.78],'LineWidth',0.8, ...
                'HandleVisibility','off');
        end
        plot(ax,lags,b.tight.mean_correlation,'k-', ...
            'LineWidth',2.1,'DisplayName','Mean of high-input images');
        errorbar(ax,0,b.tight.zero_lag_mean,b.tight.zero_lag_ci95, ...
            'o','Color',[0.82 0.25 0.20],'MarkerFaceColor','white', ...
            'LineWidth',1.4,'DisplayName','Lag 0, 95% CI');
    else
        text(ax,0.5,0.5,'Undefined: inhibitory input is zero', ...
            'Units','normalized','HorizontalAlignment','center', ...
            'Color',[0.2 0.2 0.2]);
    end
    xline(ax,0,':','Color',[0.5 0.5 0.5], ...
        'HandleVisibility','off');
    xlabel(ax,'Lag (ms): positive means inhibition lags excitation');
    ylabel(ax,'corr(E(t), I(t+lag))'); ylim(ax,[-1 1]);
    title(ax,sprintf('c_n=%d | neuron %d | n=%d', ...
        records(r).network_scale,b.tight.neuron_id, ...
        size(curves,1)),'Color',[0.05 0.05 0.05]);
    style_axis(ax);
    if r == 1 && any(isfinite(curves(:)))
        legend(ax,'Location','best','Color','white', ...
            'TextColor',[0.05 0.05 0.05]);
    end
end
title(tl,sprintf('%s: tight E/I balance after batch 1200', ...
    condition_label(records(1))),'Color',[0.02 0.02 0.02], ...
    'FontWeight','bold');
output_file = make_plot_path(plot_dir,records(1).condition, ...
    'tight_xcorr',mode);
exportgraphics(fig,char(output_file),'Resolution',240, ...
    'BackgroundColor','white'); close(fig);
end

function output_file = plot_tight_population(records,plot_dir,mode)
records = first_run_per_scale(records);
cn = [records.network_scale];
metrics = {'zero_lag','peak_correlation','peak_lag_ms'};
titles = {'Correlation at lag 0','Peak correlation','Peak lag (ms)'};
fig = figure('Visible','off','Color','white','Position',[60 60 1350 450]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
for metric_index = 1:3
    ax = nexttile(tl); hold(ax,'on');
    means = nan(size(cn)); deviations = nan(size(cn));
    for r = 1:numel(records)
        p = records(r).balance.tight_population;
        means(r) = p.([metrics{metric_index} '_mean']);
        deviations(r) = p.([metrics{metric_index} '_sd']);
    end
    if any(isfinite(means))
        errorbar(ax,cn,means,deviations,'-o', ...
            'Color',[0.20 0.48 0.72],'MarkerFaceColor','white', ...
            'LineWidth',1.7,'MarkerSize',6);
    else
        text(ax,0.5,0.5,'Undefined: inhibitory input is zero', ...
            'Units','normalized','HorizontalAlignment','center', ...
            'Color',[0.2 0.2 0.2]);
    end
    xticks(ax,cn); xlabel(ax,'Network scale c_n');
    ylabel(ax,titles{metric_index}); title(ax,titles{metric_index}, ...
        'Color',[0.05 0.05 0.05]); style_axis(ax);
    if metric_index < 3
        ylim(ax,[-1 1]);
    end
end
title(tl,sprintf('%s: tight-balance population summary (mean +/- SD)', ...
    condition_label(records(1))),'Color',[0.02 0.02 0.02], ...
    'FontWeight','bold');
output_file = make_plot_path(plot_dir,records(1).condition, ...
    'tight_population_errorbar',mode);
exportgraphics(fig,char(output_file),'Resolution',240, ...
    'BackgroundColor','white'); close(fig);
end

function records = first_run_per_scale(records)
scales = unique([records.network_scale],'stable');
selected = false(size(records));
for i = 1:numel(scales)
    candidates = find([records.network_scale] == scales(i));
    [~,local] = min([records(candidates).run_index]);
    selected(candidates(local)) = true;
end
records = records(selected);
[~,order] = sort([records.network_scale]);
records = records(order);
end

function add_fit_line(ax,x,y,color)
valid = isfinite(x) & isfinite(y);
if nnz(valid) < 2 || max(x(valid))-min(x(valid)) == 0
    return;
end
coefficients = polyfit(x(valid),y(valid),1);
x_fit = linspace(min(x(valid)),max(x(valid)),100);
plot(ax,x_fit,polyval(coefficients,x_fit),'-','Color',color, ...
    'LineWidth',1.2,'HandleVisibility','off');
end

function label = condition_label(record)
label = sprintf('Condition %s',record.condition);
end

function output_file = make_plot_path(plot_dir,condition,suffix,mode)
output_file = string(fullfile(plot_dir,sprintf( ...
    'pcb_shuffle_%s_%s_%s.png',lower(condition),mode,suffix)));
end

function style_axis(ax)
set(ax,'FontName','Arial','FontSize',10,'LineWidth',1, ...
    'Color','white','XColor',[0.08 0.08 0.08], ...
    'YColor',[0.08 0.08 0.08]);
grid(ax,'on'); ax.GridAlpha = 0.16; box(ax,'off');
end

function ensure_directory(directory)
if ~exist(directory,'dir')
    mkdir(directory);
end
end
