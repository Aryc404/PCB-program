function files = plot_pcb_shuffle_existing_results(language)
%PLOT_PCB_SHUFFLE_EXISTING_RESULTS Create the 12 requested cn=1--3 plots.
% cn=1 and repeated cn=2 runs use scatter, mean and sample SD.
% cn=2 single_run is used only when no runXX file exists; cn=3 uses the
% latest available result.

if nargin < 1, language = 'english'; end
chinese = strcmpi(language,'chinese');

code_dir = fileparts(mfilename('fullpath'));
% 优先读取当前整理文件夹内的结果；如果未复制历史结果，
% 则直接读取原工程，避免在C盘重复保存数百MB的CSV/MAT和检查点。
candidate_results_dirs = {
    fullfile(code_dir,'pcb_shuffle_study','results')
    'C:\Users\吉毅诚\Nutstore\1\bio-spiking-v1-learning-main\population_sparseness_study\pcb_shuffle_study\results'
    'D:\bio-spiking-v1-learning-main\population_sparseness_study\pcb_shuffle_study\results'
    };
required_relative_file = fullfile('archive','legacy_cn1_10seeds', ...
    'final_statistics','csv', ...
    'pcb_shuffle_reliability_cn1_10seeds_paper_final_raw.csv');
organized_relative_file = fullfile('01_实验数据','cn1_十种子旧版', ...
    'final_statistics','csv', ...
    'pcb_shuffle_reliability_cn1_10seeds_paper_final_raw.csv');
available = find(cellfun(@(folder) ...
    isfile(fullfile(folder,required_relative_file)) || ...
    isfile(fullfile(folder,organized_relative_file)), ...
    candidate_results_dirs),1,'first');
if isempty(available)
    error(['Cannot find the PCB result directory. Keep the original project ' ...
        'available or copy its pcb_shuffle_study/results directory locally.']);
end
results_dir = candidate_results_dirs{available};
% 可读取整理后的原项目，也可读取仍采用原布局的独立复现数据。
organized = isfile(fullfile(results_dir,organized_relative_file));
if organized
    reliability_file = fullfile(results_dir,organized_relative_file);
    scale_results_dir = fullfile(results_dir,'01_实验数据','网络规模实验');
    archive_dir = fullfile(results_dir,'04_历史归档');
    figures_zh = fullfile(results_dir,'03_汇总图表','网络规模对比12图');
    figures_en = fullfile(results_dir,'03_汇总图表','网络规模对比12图_英文');
else
    reliability_file = fullfile(results_dir,required_relative_file);
    scale_results_dir = fullfile(results_dir,'不同cn');
    archive_dir = fullfile(results_dir,'archive');
    figures_zh = fullfile(results_dir,'shuffle 实验12张图');
    figures_en = fullfile(results_dir,'png');
end
definitions = pcb_shuffle_conditions();
groups = definitions.figure_groups;
conditions = unique([groups{:}],'stable');

metrics = {'RF_pairwise_correlation_RMS_E', ...
    'lifetime_sparseness_E','population_sparseness_E'};
metric_titles = {'E RF pairwise correlation RMS', ...
    'Lifetime sparseness','Population sparseness'};
metric_titles_zh = {'E神经元感受野相关性系数RMS', ...
    'E神经元时间稀疏性','E神经元群体稀疏性'};
metric_files = {'rf_pairwise_correlation','lifetime_sparseness', ...
    'population_sparseness'};
group_titles_zh = {'AD AE AF AG','AD BD CD','BE BF BG','CE CF CG'};

reliability = readtable(reliability_file);
reliability.condition = string(reliability.condition);
assert(height(reliability) == 120, ...
    'Expected 12 conditions x 10 cn=1 seeds.');

[cn2_values,cn2_sd,cn2_raw,cn2_counts,cn2_batches] = ...
    load_cn2_results(scale_results_dir,conditions,metrics);
recovered = readtable(fullfile(archive_dir, ...
    'recovered_terminal_20260812', ...
    'recovered_latest_metrics.csv'));
recovered.condition = string(recovered.condition);
[cn3_values,cn3_batches] = load_last_results( ...
    scale_results_dir,3,conditions,metrics,recovered);

if chinese
    png_dir = figures_zh;
    file_suffix = '';
    font_name = 'Microsoft YaHei';
else
    png_dir = figures_en;
    file_suffix = '';
    font_name = 'Arial';
end
if ~exist(png_dir,'dir')
    mkdir(png_dir);
end

files = strings(12,1);
file_index = 0;
palette = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; ...
    0.4660 0.6740 0.1880; 0.4940 0.1840 0.5560];
markers = {'o','s','^','d'};
line_styles = {'-','--',':','-.'};
for g = 1:numel(groups)
    group_conditions = string(groups{g});
    colors = palette(1:numel(group_conditions),:);
    for m = 1:numel(metrics)
        fig = figure('Visible','off','Color','white', ...
            'Position',[100 100 900 650]);
        ax = axes(fig); %#ok<LAXES>
        hold(ax,'on');

        for c = 1:numel(group_conditions)
            condition = group_conditions(c);
            condition_index = find(conditions == condition,1);
            selected = reliability.condition == condition;
            seed_values = reliability.(metrics{m})(selected);
            assert(numel(seed_values) == 10 && all(isfinite(seed_values)), ...
                'Condition %s must have 10 finite cn=1 seeds.',condition);

            cn1_mean = mean(seed_values);
            cn1_sd = std(seed_values,0);
            cn2_seed_values = cn2_raw{condition_index,m};
            y = [cn1_mean,cn2_values(condition_index,m), ...
                cn3_values(condition_index,m)];
            color = colors(c,:);

            plot(ax,1:3,y,'LineWidth',1.8,'MarkerSize',6, ...
                'LineStyle',line_styles{c},'Marker',markers{c}, ...
                'Color',color,'MarkerFaceColor','white', ...
                'DisplayName',char(condition));

            jitter = linspace(-0.055,0.055,numel(seed_values));
            scatter(ax,1+jitter,seed_values,22,color,'filled', ...
                'MarkerFaceAlpha',0.65,'HandleVisibility','off');
            errorbar(ax,1,cn1_mean,cn1_sd,'Color',color, ...
                'LineWidth',1.5,'CapSize',10,'LineStyle','none', ...
                'HandleVisibility','off');

            if numel(cn2_seed_values) > 1
                jitter = linspace(-0.045,0.045,numel(cn2_seed_values));
                scatter(ax,2+jitter,cn2_seed_values,24,color,'filled', ...
                    'MarkerFaceAlpha',0.65,'HandleVisibility','off');
                errorbar(ax,2,cn2_values(condition_index,m), ...
                    cn2_sd(condition_index,m),'Color',color, ...
                    'LineWidth',1.5,'CapSize',10,'LineStyle','none', ...
                    'HandleVisibility','off');
            end
        end

        if chinese
            xlabel(ax,'网络规模 c_n','Color','black');
            ylabel(ax,metric_titles_zh{m},'Color','black');
            title(ax,sprintf('%s（%s）',metric_titles_zh{m}, ...
                group_titles_zh{g}),'Color','black');
        else
            xlabel(ax,'Network scale c_n','Color','black');
            ylabel(ax,metric_titles{m},'Color','black');
            title(ax,{sprintf('%s - %s',metric_titles{m}, ...
                strrep(definitions.figure_group_names{g},'_',' ')), ...
                'cn=1: 10 seeds; cn=2: available runs; dots, mean \pm SD'}, ...
                'Color','black');
        end
        xticks(ax,1:3); xlim(ax,[0.85 3.15]); ylim(ax,[-0.02 1.02]);
        grid(ax,'on'); box(ax,'off');
        legend(ax,'Location','best','Color','white','TextColor','black', ...
            'EdgeColor',[0.65 0.65 0.65]);
        set(ax,'Color','white','XColor','black','YColor','black', ...
            'GridColor',[0.8 0.8 0.8],'GridAlpha',0.6, ...
            'FontName',font_name,'FontSize',11,'LineWidth',1);

        file_index = file_index + 1;
        files(file_index) = fullfile(png_dir,sprintf( ...
            '%02d_%s_%s_cn1_cn3_errorbars%s.png',file_index, ...
            definitions.figure_group_names{g},metric_files{m},file_suffix));
        exportgraphics(fig,files(file_index),'Resolution',300, ...
            'BackgroundColor','white');
        close(fig);
    end
end

assert(file_index == 12 && all(isfile(files)), ...
    'Twelve plots were not created.');
fprintf('Created 12 figures in:\n%s\n',png_dir);
fprintf('cn=2 batches: %s\n',mat2str(cn2_batches'));
fprintf('cn=2 seed counts: %s\n',mat2str(cn2_counts'));
fprintf('cn=3 batches: %s\n',mat2str(cn3_batches'));
end

function [means,sds,raw,counts,batches] = load_cn2_results( ...
        results_dir,conditions,metrics)
means = nan(numel(conditions),numel(metrics));
sds = zeros(numel(conditions),numel(metrics));
raw = cell(numel(conditions),numel(metrics));
counts = zeros(numel(conditions),1);
batches = nan(numel(conditions),1);
csv_dir = fullfile(results_dir,'cn_2','csv');
mat_dir = fullfile(results_dir,'cn_2','mat');
for c = 1:numel(conditions)
    condition = lower(char(conditions(c)));
    files = dir(fullfile(csv_dir,sprintf( ...
        'pcb_shuffle_%s_paper_cn2_run*.csv',condition)));
    if ~isempty(files)
        paired = arrayfun(@(f)isfile(fullfile(mat_dir, ...
            strrep(f.name,'.csv','.mat'))),files);
        files = files(paired);
    end
    if isempty(files)
        files = dir(fullfile(csv_dir,sprintf( ...
            'pcb_shuffle_%s_paper_cn2_single_run.csv',condition)));
        if ~isempty(files)
            paired = arrayfun(@(f)isfile(fullfile(mat_dir, ...
                strrep(f.name,'.csv','.mat'))),files);
            files = files(paired);
        end
    end
    assert(~isempty(files),'Missing cn=2 result for %s.',conditions(c));
    counts(c) = numel(files);
    values = nan(numel(files),numel(metrics));
    file_batches = nan(numel(files),1);
    for f = 1:numel(files)
        data = readtable(fullfile(files(f).folder,files(f).name));
        [file_batches(f),row] = max(data.batch);
        for m = 1:numel(metrics)
            values(f,m) = data.(metrics{m})(row);
        end
    end
    assert(all(file_batches == 1200), ...
        'Condition %s has an incomplete cn=2 run.',conditions(c));
    assert(all(isfinite(values),'all'), ...
        'Condition %s has a non-finite cn=2 metric.',conditions(c));
    batches(c) = max(file_batches);
    means(c,:) = mean(values,1);
    if numel(files) > 1
        sds(c,:) = std(values,0,1);
    end
    for m = 1:numel(metrics)
        raw{c,m} = values(:,m);
    end
end
end

function [values,batches] = load_last_results( ...
        results_dir,cn,conditions,metrics,fallback)
values = nan(numel(conditions),numel(metrics));
batches = nan(numel(conditions),1);
for c = 1:numel(conditions)
    condition = conditions(c);
    filename = sprintf('pcb_shuffle_%s_paper_cn%d_single_run.csv', ...
        lower(char(condition)),cn);
    csv_file = fullfile(results_dir,sprintf('cn_%d',cn), ...
        'csv',filename);
    if ~isfile(csv_file)
        csv_file = fullfile(results_dir,sprintf('cn_%d',cn), ...
            'single_run','csv',filename);
    end
    if isfile(csv_file)
        data = readtable(csv_file);
        row = height(data);
    else
        assert(~isempty(fallback), ...
            'Missing cn=%d result for %s.',cn,condition);
        data = fallback(fallback.condition == condition,:);
        assert(height(data) == 1, ...
            'Fallback must contain one result for %s.',condition);
        row = 1;
    end
    batches(c) = data.batch(row);
    for m = 1:numel(metrics)
        values(c,m) = data.(metrics{m})(row);
    end
end
assert(all(isfinite(values),'all'),'All plotted values must be finite.');
end
