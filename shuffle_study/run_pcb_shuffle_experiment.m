% 用途：运行指定的消融条件，配置训练与评估参数，并保存CSV和MAT结果。
function summary = run_pcb_shuffle_experiment(mode, inhibitory_selection, ...
        homeostatic_selection, user_options)

if nargin < 1 || isempty(mode)
    mode = 'paper';
end
if nargin < 2 || isempty(inhibitory_selection)
    inhibitory_selection = 'A';
end
if nargin < 3 || isempty(homeostatic_selection)
    homeostatic_selection = 'D';
end
if nargin < 4
    user_options = struct();
end

mode = lower(char(mode));
if ~strcmp(mode,'paper')
    error('PCB:UnsupportedMode','Only the paper experiment is supported.');
end

definitions = pcb_shuffle_conditions();
inhibitory_codes = expand_selection(inhibitory_selection, ...
    definitions.valid_inhibitory_codes, 'A/B/C');
homeostatic_codes = expand_selection(homeostatic_selection, ...
    definitions.valid_homeostatic_codes, 'D/E/F/G');
options = fill_runner_options(user_options, definitions, mode);

if isempty(options.output_root)
    paths = pcb_results_paths();
    output_root = paths.scaling;
else
    [~,output_root] = pcb_results_paths(options.output_root);
end
scale_dir = fullfile(output_root,sprintf('cn_%d',options.network_scale));
run_dir = scale_dir;
csv_dir = fullfile(run_dir, 'csv');
mat_dir = fullfile(run_dir, 'mat');
ensure_directories({csv_dir,mat_dir});

run_options.run_window_experiment = false;
run_options.measure_standard_metrics = ~options.balance_only;
run_options.measure_extended_metrics = false;
run_options.measure_final_diagnostics = options.final_diagnostics && ...
    ~options.balance_only;
run_options.measure_ei_balance = options.measure_ei_balance;
run_options.balance_only = options.balance_only;
run_options.finite_sparseness_correction = true;
run_options.seed_override = options.training_seed;
run_options.eval_patch_seed_override = options.eval_patch_seed;
run_options.eval_dynamics_seed_override = options.eval_dynamics_seed;
run_options.eval_grating_seed_override = options.eval_grating_seed;
run_options.balance_seed_override = options.balance_seed;
run_options.balance_neuron_count = options.balance_neuron_count;
run_options.balance_example_candidates = options.balance_example_candidates;
run_options.balance_population_candidates = options.balance_population_candidates;
run_options.balance_scatter_candidates = options.balance_scatter_candidates;
run_options.balance_tight_candidates = options.balance_tight_candidates;
run_options.balance_tight_top_k = options.balance_tight_top_k;
run_options.balance_max_lag_ms = options.balance_max_lag_ms;
run_options.balance_transient_ms = options.balance_transient_ms;
run_options.training_checkpoint_batches = options.training_checkpoint_batches;
run_options.training_progress_every = options.training_progress_every;
run_options.training_checkpoint_dir = options.training_checkpoint_dir;
run_options.training_checkpoint_stem = options.training_checkpoint_stem;
run_options.balance_checkpoint_file = options.balance_checkpoint_file;
run_options.resume_training = options.resume_training;
run_options.network_scale_override = options.network_scale;
run_options.checkpoints_override = checkpoint_batches();
if options.balance_only
    run_options.checkpoints_override = ...
        run_options.checkpoints_override(end);
end

n_conditions = numel(inhibitory_codes)*numel(homeostatic_codes);
results = cell(n_conditions,1);
conditions = repmat(pcb_shuffle_conditions('A','D'),n_conditions,1);
tables = cell(n_conditions,1);
index = 0;

fprintf(['\nPCB A--G experiment | %s | cn=%d | run %02d | ' ...
    '%d checkpoints\n'],upper(mode),options.network_scale, ...
    options.run_index,numel(run_options.checkpoints_override));

for i = 1:numel(inhibitory_codes)
    for h = 1:numel(homeostatic_codes)
        index = index + 1;
        condition = pcb_shuffle_conditions( ...
            inhibitory_codes(i), homeostatic_codes(h));
        conditions(index) = condition;
        fprintf('\n[%d/%d] %s: %s\n', ...
            index,n_conditions,condition.id,condition.display_name);
        result = run_ps_factorial_condition(mode,condition,run_options);
        validate_result(result,condition);
        results{index} = result;
        tables{index} = checkpoint_table(result,options);
        if ~options.defer_save
            save_condition(result,tables{index},mode,options,csv_dir,mat_dir);
        end
    end
end

nonempty_tables = tables(~cellfun(@isempty,tables));
if isempty(nonempty_tables)
    summary_table = table();
else
    summary_table = vertcat(nonempty_tables{:});
end
if n_conditions > 1 && ~isempty(summary_table)
    summary_stem = sprintf('pcb_shuffle_summary_all_%s_cn%d_run%02d', ...
        mode,options.network_scale,options.run_index);
    writetable(summary_table,fullfile(csv_dir,[summary_stem '.csv']));
end

summary.mode = mode;
summary.options = options;
summary.conditions = conditions;
summary.results = results;
summary.table = summary_table;
if n_conditions > 1 && ~isempty(summary_table)
    save(fullfile(mat_dir,[summary_stem '.mat']),'summary','-v7.3');
end

fprintf('\nCompleted %d condition(s). Outputs:\n%s\n', ...
    n_conditions,run_dir);
end

function batches = checkpoint_batches()
batches = 0:100:1200;
end

function codes = expand_selection(selection, valid_codes, label)
selection = upper(string(selection));
if isscalar(selection) && selection == "ALL"
    codes = valid_codes;
    return;
end
selection = selection(:)';
if isempty(selection) || any(~ismember(selection,valid_codes))
    error('Selection must contain %s codes or ALL.',label);
end
codes = unique(selection,'stable');
end

function options = fill_runner_options(options,definitions,mode)
if ~isfield(options,'output_root')
    options.output_root = '';
end
if ~isfield(options,'defer_save')
    options.defer_save = false;
end
if ~isfield(options,'final_diagnostics')
    options.final_diagnostics = true;
end
if ~isfield(options,'run_index') || isempty(options.run_index)
    options.run_index = 1;
end
if ~isfield(options,'network_scale') || isempty(options.network_scale)
    options.network_scale = 3;
end
if ~isscalar(options.network_scale) || ...
        ~ismember(options.network_scale,definitions.network_scales)
    error('network_scale must be one of 1, 2, 3 or 4.');
end

if ~isfield(options,'training_seed')
    options.training_seed = definitions.seeds.training;
end
if ~isfield(options,'eval_patch_seed')
    options.eval_patch_seed = definitions.seeds.eval_patch;
end
if ~isfield(options,'eval_dynamics_seed')
    options.eval_dynamics_seed = definitions.seeds.eval_dynamics;
end
if ~isfield(options,'eval_grating_seed')
    options.eval_grating_seed = definitions.seeds.eval_grating;
end
if ~isfield(options,'measure_ei_balance')
    options.measure_ei_balance = false;
end
if ~isfield(options,'balance_only')
    options.balance_only = false;
end
if ~isfield(options,'training_checkpoint_batches')
    options.training_checkpoint_batches = zeros(1,0);
end
if ~isfield(options,'training_progress_every')
    options.training_progress_every = 100;
end
if ~isfield(options,'training_checkpoint_dir')
    options.training_checkpoint_dir = '';
end
if ~isfield(options,'training_checkpoint_stem')
    options.training_checkpoint_stem = '';
end
if ~isfield(options,'balance_checkpoint_file')
    options.balance_checkpoint_file = '';
end
if ~isfield(options,'resume_training')
    options.resume_training = false;
end
if ~isfield(options,'balance_seed')
    options.balance_seed = 3202603;
end
balance_defaults = struct('neuron_count',40, ...
    'example_candidates',1000,'population_candidates',100, ...
    'scatter_candidates',100, ...
    'tight_candidates',1000,'tight_top_k',10, ...
    'max_lag_ms',50,'transient_ms',100);
balance_fields = fieldnames(balance_defaults);
for field_index = 1:numel(balance_fields)
    name = balance_fields{field_index};
    option_name = ['balance_' name];
    if ~isfield(options,option_name)
        options.(option_name) = balance_defaults.(name);
    end
end
if ~isscalar(options.run_index) || options.run_index < 1 || ...
        options.run_index ~= floor(options.run_index)
    error('run_index must be a positive integer.');
end
options.mode = mode;
options.defer_save = logical(options.defer_save);
options.final_diagnostics = logical(options.final_diagnostics);
options.measure_ei_balance = logical(options.measure_ei_balance);
options.balance_only = logical(options.balance_only);
options.resume_training = logical(options.resume_training);
end

function T = checkpoint_table(result,options)
if options.balance_only
    T = table();
    return;
end
n = numel(result.batch);
condition = repmat(string(result.condition.id),n,1);
inhibitory_code = repmat(string(result.condition.inhibitory_code),n,1);
homeostatic_code = repmat(string(result.condition.homeostatic_code),n,1);
network_scale = repmat(options.network_scale,n,1);
run_index = repmat(options.run_index,n,1);
training_seed = repmat(options.training_seed,n,1);
eval_patch_seed = repmat(options.eval_patch_seed,n,1);
eval_dynamics_seed = repmat(options.eval_dynamics_seed,n,1);
eval_grating_seed = repmat(options.eval_grating_seed,n,1);
inhibitory_transmission = repmat( ...
    result.condition.inhibitory_transmission_enabled,n,1);
inhibitory_plasticity = repmat( ...
    result.condition.inhibitory_plasticity_enabled,n,1);
adaptive_threshold = repmat(result.condition.threshold_enabled,n,1);
l1_normalization = repmat( ...
    result.condition.l1_normalization_enabled,n,1);
phase = result.phase;
mean_OSI_E = nan(n,1);
median_OSI_E = nan(n,1);
fraction_OSI_ge_0_5 = nan(n,1);
valid_OSI_fraction = nan(n,1);
mean_rate_E_Hz = nan(n,1);
silent_neuron_fraction_E = nan(n,1);
if isfield(result,'final_diagnostics')
    d = result.final_diagnostics;
    mean_OSI_E(end) = d.mean_OSI_E;
    median_OSI_E(end) = d.median_OSI_E;
    fraction_OSI_ge_0_5(end) = d.fraction_OSI_ge_0_5;
    valid_OSI_fraction(end) = d.valid_OSI_fraction;
    mean_rate_E_Hz(end) = mean(d.mean_rate_E_Hz);
    silent_neuron_fraction_E(end) = d.silent_neuron_fraction_E;
end

T = table(condition,inhibitory_code,homeostatic_code,network_scale, ...
    run_index,training_seed,eval_patch_seed,eval_dynamics_seed, ...
    eval_grating_seed, ...
    inhibitory_transmission,inhibitory_plasticity,adaptive_threshold, ...
    l1_normalization,phase, ...
    result.batch,result.PS_E,result.lifetime_sparseness_E, ...
    result.RF_correlation_RMS_E,mean_OSI_E,median_OSI_E, ...
    fraction_OSI_ge_0_5,valid_OSI_fraction,mean_rate_E_Hz, ...
    silent_neuron_fraction_E, ...
    'VariableNames',{'condition','inhibitory_code','homeostatic_code', ...
    'network_scale','run_index','training_seed','eval_patch_seed', ...
    'eval_dynamics_seed','eval_grating_seed','inhibitory_transmission', ...
    'inhibitory_plasticity','adaptive_threshold','l1_normalization', ...
    'phase','batch','population_sparseness_E','lifetime_sparseness_E', ...
    'RF_pairwise_correlation_RMS_E','mean_OSI_E','median_OSI_E', ...
    'fraction_OSI_ge_0_5','valid_OSI_fraction','mean_rate_E_Hz', ...
    'silent_neuron_fraction_E'});
end

function save_condition(result,T,mode,options,csv_dir,mat_dir)
stem = sprintf('pcb_shuffle_%s_%s_cn%d_run%02d', ...
    lower(result.condition.id),mode,options.network_scale,options.run_index);
if ~isempty(T)
    writetable(T,fullfile(csv_dir,[stem '.csv']));
end

save(fullfile(mat_dir,[stem '.mat']),'result','-v7');
if isfield(result,'final_diagnostics')
    save_pcb_final_diagnostics(result,fileparts(csv_dir),stem);
end
end

function validate_result(result,condition)
if result.options.measure_standard_metrics
    expected_batches = checkpoint_batches()';
    if ~isequal(result.batch(:),expected_batches)
        error('Condition %s did not record the expected checkpoints.', ...
            condition.id);
    end
    core_fields = {'PS_E','lifetime_sparseness_E', ...
        'RF_correlation_RMS_E'};
    for i = 1:numel(core_fields)
        values = result.(core_fields{i});
        if any(~isfinite(values))
            error('%s contains NaN or Inf for condition %s.', ...
                core_fields{i},condition.id);
        end
    end
    if any(result.PS_E < -1e-10 | result.PS_E > 1+1e-10) || ...
            any(result.lifetime_sparseness_E < -1e-10 | ...
            result.lifetime_sparseness_E > 1+1e-10)
        error('Corrected sparseness is outside [0,1] for %s.',condition.id);
    end
else
    expected_final_batch = checkpoint_batches();
    if ~isscalar(result.batch) || ...
            result.batch ~= expected_final_batch(end)
        error('Balance-only condition %s must retain only the final batch.', ...
            condition.id);
    end
end
if result.options.measure_final_diagnostics
    if ~isfield(result,'final_diagnostics')
        error('Final diagnostics are missing for %s.',condition.id);
    end
    d = result.final_diagnostics;
    if any(~isfinite(d.mean_rate_E_Hz)) || ...
            any(d.OSI_E(isfinite(d.OSI_E)) < 0 | ...
            d.OSI_E(isfinite(d.OSI_E)) > 1)
        error('Final E-cell diagnostics are invalid for %s.',condition.id);
    end
end
if result.options.measure_ei_balance && ~isfield(result,'ei_balance')
    error('Final E/I balance diagnostics are missing for %s.',condition.id);
end

end

function plot_condition_trajectory(result,output_file)
dark = [0.04 0.14 0.25];
colors = lines(4);
fig = figure('Color','w','Visible','off','Position',[80 80 1350 760]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

ax = nexttile; hold(ax,'on');
plot(ax,result.batch,result.PS_E,'-o','Color',colors(1,:), ...
    'LineWidth',1.8,'MarkerSize',4);
plot(ax,result.batch,result.lifetime_sparseness_E,'-s', ...
    'Color',colors(2,:),'LineWidth',1.8,'MarkerSize',4);
xlabel(ax,'Batch'); ylabel(ax,'Sparseness'); ylim(ax,[0 1]);
title(ax,'Population and lifetime sparseness');
lg = legend(ax,{'Population','Lifetime'},'Location','best','Color','w');
lg.TextColor = dark;

ax = nexttile;
plot(ax,result.batch,result.RF_correlation_RMS_E,'-o', ...
    'Color',colors(3,:),'LineWidth',1.8,'MarkerSize',4);
xlabel(ax,'Batch'); ylabel(ax,'RMS correlation'); ylim(ax,[0 1]);
title(ax,'E RF pairwise correlation RMS');

ax = nexttile; hold(ax,'on');
plot(ax,result.batch,result.mean_rate_E_Hz,'-o', ...
    'Color',colors(1,:),'LineWidth',1.8,'MarkerSize',4);
plot(ax,result.batch,result.mean_rate_I_Hz,'-s', ...
    'Color',colors(2,:),'LineWidth',1.8,'MarkerSize',4);
xlabel(ax,'Batch'); ylabel(ax,'Mean rate (Hz)');
title(ax,'Population firing rates');
lg = legend(ax,{'E','I'},'Location','best','Color','w');
lg.TextColor = dark;

ax = nexttile; hold(ax,'on');
plot(ax,result.batch,result.mean_theta_E,'-o', ...
    'Color',colors(1,:),'LineWidth',1.8,'MarkerSize',4);
plot(ax,result.batch,result.mean_theta_I,'-s', ...
    'Color',colors(2,:),'LineWidth',1.8,'MarkerSize',4);
xlabel(ax,'Batch'); ylabel(ax,'Mean threshold');
title(ax,'Adaptive thresholds');
lg = legend(ax,{'E','I'},'Location','best','Color','w');
lg.TextColor = dark;

ax = nexttile;
plot(ax,result.batch,result.mean_OSI_E,'-o', ...
    'Color',colors(4,:),'LineWidth',1.8,'MarkerSize',4);
xlabel(ax,'Batch'); ylabel(ax,'Mean OSI'); ylim(ax,[0 1]);
title(ax,'E-cell orientation selectivity');

ax = nexttile;
axis(ax,'off');
text(ax,0,0.92,sprintf('Condition %s',result.condition.id), ...
    'FontWeight','bold','FontSize',14,'Color',dark);
description_lines = split(string(result.condition.display_name)," | ");
text(ax,0,0.72,cellstr(description_lines), ...
    'FontSize',11,'Color',dark,'Interpreter','none');
text(ax,0,0.48,sprintf('Training seed: %d',result.cfg.seed), ...
    'FontSize',11,'Color',dark);
text(ax,0,0.32,'Sparseness uses finite-population correction.', ...
    'FontSize',10,'Color',dark);

for k = 1:5
    style_axis(nexttile(tl,k));
end
sg = title(tl,['PCB condition ' result.condition.id]);
sg.Color = dark; sg.FontWeight = 'bold';
exportgraphics(fig,output_file,'Resolution',240,'BackgroundColor','white');
close(fig);
end

function plot_rf_endpoints(result,output_file)
input_side = round(sqrt(result.cfg.N_pixels));
initial_rf = make_rf_mosaic(double(result.RF_E(:,:,1)),input_side);
final_rf = make_rf_mosaic(double(result.RF_E(:,:,end)),input_side);
limit = max(abs([initial_rf(:); final_rf(:)]));
if ~isfinite(limit) || limit == 0
    limit = 1;
end
fig = figure('Color','w','Visible','off','Position',[80 80 1200 560]);
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
ax = nexttile; imagesc(ax,initial_rf); axis(ax,'image','off');
clim(ax,[-limit limit]); title(ax,'E RFs: batch 0');
ax = nexttile; imagesc(ax,final_rf); axis(ax,'image','off');
clim(ax,[-limit limit]); title(ax,sprintf('E RFs: batch %d', ...
    result.batch(end)));
colormap(fig,blue_white_red(256));
sg = title(tl,['RF endpoints - condition ' result.condition.id]);
sg.Color = [0.04 0.14 0.25]; sg.FontWeight = 'bold';
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

function plot_final_comparison(results,conditions,output_file)
n = numel(results);
labels = strings(n,1);
ps = nan(n,1); ls = nan(n,1); rf = nan(n,1);
for i = 1:n
    labels(i) = string(conditions(i).id);
    ps(i) = results{i}.PS_E(end);
    ls(i) = results{i}.lifetime_sparseness_E(end);
    rf(i) = results{i}.RF_correlation_RMS_E(end);
end
fig = figure('Color','w','Visible','off','Position',[80 80 1300 430]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
metrics = {ps,ls,rf};
titles = {'Population sparseness','Lifetime sparseness', ...
    'RF pairwise correlation RMS'};
for panel = 1:3
    ax = nexttile;
    bar(ax,metrics{panel},'FaceColor',[0.20 0.48 0.72]);
    set(ax,'XTick',1:n,'XTickLabel',labels);
    xtickangle(ax,45); ylim(ax,[0 1]); title(ax,titles{panel});
    style_axis(ax);
end
sg = title(tl,'Final metrics across PCB A--G combinations');
sg.Color = [0.04 0.14 0.25]; sg.FontWeight = 'bold';
exportgraphics(fig,output_file,'Resolution',240,'BackgroundColor','white');
close(fig);
end

function style_axis(ax)
set(ax,'FontName','Arial','FontSize',10,'LineWidth',1, ...
    'Color','w','XColor',[0.08 0.08 0.08], ...
    'YColor',[0.08 0.08 0.08]);
grid(ax,'on'); ax.GridAlpha = 0.16;
ax.Title.Color = [0.04 0.14 0.25];
ax.Title.FontWeight = 'bold';
end

function ensure_directories(directories)
for i = 1:numel(directories)
    if ~exist(directories{i},'dir')
        mkdir(directories{i});
    end
end
end
