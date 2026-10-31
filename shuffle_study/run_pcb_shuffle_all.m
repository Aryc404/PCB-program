function aggregate_summary = run_pcb_shuffle_all(mode,user_options)
%RUN_PCB_SHUFFLE_ALL Run each selected condition for the requested cn values.
% Supports a separate repetition count for each cn value and records final
% E-cell diagnostics in addition to batch 0:100:1200 metrics.

if nargin < 1 || isempty(mode)
    mode = 'paper';
end
if nargin < 2
    user_options = struct();
end

mode = lower(char(mode));
if ~ismember(mode,{'quick','paper'})
    error('Mode must be ''quick'' or ''paper''.');
end

definitions = pcb_shuffle_conditions();
options = fill_all_options(user_options,definitions);
ensure_repo_path_on_client();
output_root = resolve_output_root(options,mode);
selection_tag = make_selection_tag(options);
scope_tag = make_scope_tag(selection_tag,options.network_scales, ...
    definitions.network_scales);
condition_ids = selected_condition_ids(options,definitions);
n_scales = numel(options.network_scales);
n_conditions = numel(condition_ids);
all_tables = cell(n_scales,1);
all_task_outputs = cell(n_scales,1);
max_tasks = n_conditions*max(options.runs_per_scale);

[parallel_enabled,pool_workers] = prepare_parallel_pool(options,max_tasks);
fprintf(['\nPCB checkpoint experiment | %s | %s | ' ...
    '%d cn values x %d conditions | runs per cn: %s\n'], ...
    upper(mode),upper(selection_tag),n_scales,n_conditions, ...
    mat2str(options.runs_per_scale));

resume_enabled = options.resume;
seeds = definitions.reliability_seeds;

for c = 1:n_scales
    network_scale = options.network_scales(c);
    n_runs = options.runs_per_scale(c);
    n_tasks = n_conditions*n_runs;
    task_conditions = repelem(condition_ids(:),n_runs);
    task_run_indices = repmat((1:n_runs)',n_conditions,1);
    task_outputs = cell(n_tasks,1);
    max_workers = workers_for_scale( ...
        network_scale,pool_workers,options,n_tasks);
    prepare_scale_output_directories(output_root,network_scale);
    fprintf(['\n===== cn=%d | conditions=%d | tasks=%d | ' ...
        'workers=%d =====\n'],network_scale,n_conditions,n_tasks,max_workers);

    if parallel_enabled && max_workers > 1
        parfor (task_index = 1:n_tasks,max_workers)
            task_outputs{task_index} = run_one_condition( ...
                mode,output_root,task_conditions(task_index), ...
                network_scale,task_run_indices(task_index),seeds, ...
                resume_enabled,options);
        end
    else
        for task_index = 1:n_tasks
            task_outputs{task_index} = run_one_condition( ...
                mode,output_root,task_conditions(task_index), ...
                network_scale,task_run_indices(task_index),seeds, ...
                resume_enabled,options);
        end
    end
    task_tables = cell(n_tasks,1);
    for task_index = 1:n_tasks
        persist_task_output(task_outputs{task_index},mode,output_root, ...
            network_scale,task_run_indices(task_index));
        task_tables{task_index} = task_outputs{task_index}.table;
    end
    task_tables = task_tables(~cellfun(@isempty,task_tables));
    if isempty(task_tables)
        all_tables{c} = table();
    else
        all_tables{c} = vertcat(task_tables{:});
    end
    all_task_outputs{c} = task_outputs;
end

recorded_tables = all_tables(~cellfun(@isempty,all_tables));
if isempty(recorded_tables)
    checkpoint_table = table();
    raw_table = table();
    summary_table = table();
else
    checkpoint_table = vertcat(recorded_tables{:});
    checkpoint_table = sortrows(checkpoint_table, ...
        {'condition','network_scale','run_index','batch'});
    final_batch = max(checkpoint_table.batch);
    raw_table = checkpoint_table(checkpoint_table.batch == final_batch,:);
    raw_table = sortrows(raw_table, ...
        {'condition','network_scale','run_index'});
    summary_table = summarize_runs(raw_table);
end

result_dir = fullfile(output_root,'final_statistics', ...
    sprintf('%s_%s_runs',lower(scope_tag),mode));
csv_dir = fullfile(result_dir,'csv');
mat_dir = fullfile(result_dir,'mat');
png_dir = fullfile(result_dir,'png');
ensure_directories({csv_dir,mat_dir,png_dir});
remove_stale_aggregate_plots(png_dir,mode);

stem = sprintf('pcb_shuffle_%s_%s_runs', ...
    lower(scope_tag),mode);
checkpoint_csv = fullfile(csv_dir,[stem '_all_checkpoints.csv']);
raw_csv = fullfile(csv_dir,[stem '_raw.csv']);
summary_csv = fullfile(csv_dir,[stem '_values.csv']);
if isempty(checkpoint_table)
    checkpoint_csv = '';
    raw_csv = '';
    summary_csv = '';
    plot_files = strings(0,1);
else
    writetable(checkpoint_table,checkpoint_csv);
    writetable(raw_table,raw_csv);
    writetable(summary_table,summary_csv);
    recorded_scales = unique(checkpoint_table.network_scale)';
    if numel(recorded_scales) == 1 && ...
            max(checkpoint_table.run_index) == 1
        plot_files = plot_checkpoint_trajectory_figures( ...
            checkpoint_table,definitions,selection_tag,png_dir,mode, ...
            recorded_scales);
    else
        plot_files = plot_final_size_figures( ...
            summary_table,definitions,selection_tag,png_dir,mode);
    end
end

ei_balance_outputs = struct('loose_csv',"",'tight_csv',"", ...
    'plot_files',strings(0,1));
if options.measure_ei_balance
    ei_balance_outputs = save_pcb_ei_balance_summary( ...
        all_task_outputs,options.network_scales,condition_ids, ...
        options.runs_per_scale,csv_dir,png_dir,stem,mode);
end

aggregate_summary.mode = mode;
aggregate_summary.condition_selection = selection_tag;
aggregate_summary.network_scales = options.network_scales;
aggregate_summary.runs_per_scale = options.runs_per_scale;
aggregate_summary.checkpoint_table = checkpoint_table;
aggregate_summary.raw_table = raw_table;
aggregate_summary.summary_table = summary_table;
aggregate_summary.checkpoint_csv = checkpoint_csv;
aggregate_summary.raw_csv = raw_csv;
aggregate_summary.summary_csv = summary_csv;
aggregate_summary.plot_files = plot_files;
aggregate_summary.ei_balance = ei_balance_outputs;
aggregate_summary.output_directory = result_dir;
save(fullfile(mat_dir,[stem '.mat']),'aggregate_summary','-v7.3');

fprintf('\nCompleted checkpoint experiment. Outputs:\n%s\n',result_dir);
end

function output = run_one_condition(mode,output_root,condition_id,network_scale, ...
        run_index,seeds,resume_enabled,options)
run_options.network_scale = network_scale;
run_options.run_index = run_index;
run_options.output_root = output_root;
run_options.defer_save = true;
run_options.training_seed = seeds.training(run_index);
run_options.eval_patch_seed = seeds.eval_patch(run_index);
run_options.eval_dynamics_seed = seeds.eval_dynamics(run_index);
run_options.eval_grating_seed = seeds.eval_grating(run_index);
run_options.balance_only = ismember(network_scale, ...
    options.balance_only_scales);
run_options.measure_ei_balance = options.measure_ei_balance;
run_options.balance_seed = options.balance_seed;
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
condition_id = char(condition_id);
csv_file = fullfile(output_root,sprintf('cn_%d',network_scale), ...
    'csv',sprintf('pcb_shuffle_%s_%s_cn%d_run%02d.csv', ...
    lower(condition_id),mode,network_scale,run_index));
mat_file = fullfile(output_root,sprintf('cn_%d',network_scale), ...
    'mat',sprintf('pcb_shuffle_%s_%s_cn%d_run%02d.mat', ...
    lower(condition_id),mode,network_scale,run_index));
checkpoint_dir = fullfile(output_root,sprintf('cn_%d',network_scale), ...
    'checkpoints');
checkpoint_stem = sprintf('pcb_shuffle_%s_%s_cn%d_run%02d', ...
    lower(condition_id),mode,network_scale,run_index);
run_options.training_checkpoint_dir = checkpoint_dir;
run_options.training_checkpoint_stem = checkpoint_stem;
run_options.balance_checkpoint_file = fullfile(checkpoint_dir, ...
    [checkpoint_stem '_balance.mat']);
run_options.resume_training = resume_enabled;
if resume_enabled && completed_condition_files( ...
        csv_file,mat_file,condition_id,network_scale,run_index,mode, ...
        options.measure_ei_balance,run_options.balance_only)
    if run_options.balance_only
        output.table = table();
    else
        output.table = readtable(csv_file,'TextType','string');
    end
    if options.measure_ei_balance || run_options.balance_only
        loaded = load(mat_file,'result');
        output.result = loaded.result;
    else
        output.result = [];
    end
    output.requires_save = false;
    output.condition_id = condition_id;
    output.balance_only = run_options.balance_only;
    output.run_index = run_index;
    fprintf('Resume: %s cn=%d run=%02d already complete.\n', ...
        condition_id,network_scale,run_index);
    return;
end
run_summary = run_pcb_shuffle_experiment(mode, ...
    condition_id(1),condition_id(2),run_options);
output.table = run_summary.table;
output.result = run_summary.results{1};
output.requires_save = true;
output.condition_id = condition_id;
output.balance_only = run_options.balance_only;
output.run_index = run_index;
clear run_summary
end

function persist_task_output(output,mode,output_root,network_scale,run_index)
% Thread workers perform computation only; the MATLAB client writes files.
if ~output.requires_save
    return;
end
condition_id = output.condition_id;
run_dir = fullfile(output_root,sprintf('cn_%d',network_scale));
csv_dir = fullfile(run_dir,'csv');
mat_dir = fullfile(run_dir,'mat');
ensure_directories({csv_dir,mat_dir});
stem = sprintf('pcb_shuffle_%s_%s_cn%d_run%02d', ...
    lower(condition_id),mode,network_scale,run_index);
T = output.table;
result = output.result;
if output.balance_only
    remove_stale_condition_metrics(run_dir,stem);
else
    writetable(T,fullfile(csv_dir,[stem '.csv']));
end
save(fullfile(mat_dir,[stem '.mat']),'result','-v7');
if isfield(result,'final_diagnostics')
    save_pcb_final_diagnostics(result,run_dir,stem);
end
end

function complete = completed_condition_files(csv_file,mat_file, ...
        condition_id,network_scale,run_index,mode,require_ei_balance, ...
        balance_only)
complete = false;
if ~isfile(mat_file) || (~balance_only && ~isfile(csv_file))
    return;
end
if balance_only
    try
        loaded = load(mat_file,'result');
        result = loaded.result;
        complete = isfield(result,'ei_balance') && ...
            isfield(result,'options') && ...
            isfield(result.options,'balance_only') && ...
            logical(result.options.balance_only) && ...
            ~logical(result.options.measure_standard_metrics) && ...
            string(result.condition.id) == string(condition_id) && ...
            result.cfg.network_scale == network_scale && ...
            result.ei_balance.measurement_batch == result.cfg.N_n;
    catch
        complete = false;
    end
    return;
end
[mat_dir,stem] = fileparts(mat_file);
run_dir = fileparts(mat_dir);
diagnostic_files = { ...
    fullfile(run_dir,'csv','final_diagnostics', ...
    [stem '_final_E_cells.csv']), ...
    fullfile(run_dir,'csv','final_diagnostics', ...
    [stem '_final_E_response_counts.csv']), ...
    fullfile(run_dir,'png','final_diagnostics','OSI', ...
    [stem '_final_E_OSI.png']), ...
    fullfile(run_dir,'png','final_diagnostics','RF', ...
    [stem '_final_E_RF.png']), ...
    fullfile(run_dir,'png','final_diagnostics','Firing', ...
    [stem '_final_E_firing.png'])};
if any(~cellfun(@isfile,diagnostic_files))
    return;
end
mat_info = dir(mat_file);
if isempty(mat_info) || mat_info.bytes == 0
    return;
end
try
    variables = whos('-file',mat_file);
    if ~any(strcmp({variables.name},'result'))
        return;
    end
    if require_ei_balance
        loaded = load(mat_file,'result');
        if ~isfield(loaded.result,'ei_balance')
            return;
        end
    end
    T = readtable(csv_file,'TextType','string');
    if strcmp(mode,'paper')
        expected_batches = (0:100:1200)';
    else
        expected_batches = (0:2:6)';
    end
    required = {'condition','network_scale','run_index','batch', ...
        'population_sparseness_E','lifetime_sparseness_E', ...
        'RF_pairwise_correlation_RMS_E','mean_OSI_E','median_OSI_E', ...
        'fraction_OSI_ge_0_5','valid_OSI_fraction', ...
        'mean_rate_E_Hz','silent_neuron_fraction_E'};
    complete = all(ismember(required,T.Properties.VariableNames)) && ...
        height(T) == numel(expected_batches) && ...
        isequal(T.batch(:),expected_batches) && ...
        all(string(T.condition) == string(condition_id)) && ...
        all(T.network_scale == network_scale) && ...
        all(T.run_index == run_index);
catch
    complete = false;
end
end

function condition_ids = selected_condition_ids(options,definitions)
inhibitory = upper(string(options.inhibitory_selection));
homeostatic = upper(string(options.homeostatic_selection));
if inhibitory == "ALL" && homeostatic == "ALL"
    condition_ids = strings(1, ...
        numel(definitions.valid_inhibitory_codes)* ...
        numel(definitions.valid_homeostatic_codes));
    index = 0;
    for i = 1:numel(definitions.valid_inhibitory_codes)
        for h = 1:numel(definitions.valid_homeostatic_codes)
            index = index + 1;
            condition_ids(index) = ...
                definitions.valid_inhibitory_codes(i) + ...
                definitions.valid_homeostatic_codes(h);
        end
    end
elseif inhibitory ~= "ALL" && homeostatic == "ALL"
    condition_ids = inhibitory + definitions.valid_homeostatic_codes;
elseif inhibitory == "ALL" && homeostatic ~= "ALL"
    condition_ids = definitions.valid_inhibitory_codes + homeostatic;
else
    condition_ids = inhibitory + homeostatic;
end
end

function prepare_scale_output_directories(output_root,network_scale)
run_dir = fullfile(output_root,sprintf('cn_%d',network_scale));
ensure_directories({fullfile(run_dir,'csv'),fullfile(run_dir,'mat'), ...
    fullfile(run_dir,'checkpoints')});
end

function options = fill_all_options(options,definitions)
if ~isfield(options,'network_scales') || isempty(options.network_scales)
    options.network_scales = 3;
end
options.network_scales = validate_integer_selection( ...
    options.network_scales,definitions.network_scales,'network_scales');
if ~isfield(options,'runs_per_scale') || isempty(options.runs_per_scale)
    options.runs_per_scale = ones(size(options.network_scales));
end
if isscalar(options.runs_per_scale)
    options.runs_per_scale = repmat(options.runs_per_scale, ...
        size(options.network_scales));
end
if ~isnumeric(options.runs_per_scale) || ...
        numel(options.runs_per_scale) ~= numel(options.network_scales) || ...
        any(options.runs_per_scale < 1) || ...
        any(options.runs_per_scale ~= floor(options.runs_per_scale)) || ...
        any(options.runs_per_scale > ...
        numel(definitions.reliability_seeds.training))
    error('runs_per_scale must contain 1--10 runs for each selected cn.');
end
options.runs_per_scale = options.runs_per_scale(:)';
if ~isfield(options,'balance_only_scales') || ...
        isempty(options.balance_only_scales)
    options.balance_only_scales = zeros(1,0);
else
    options.balance_only_scales = validate_integer_selection( ...
        options.balance_only_scales,definitions.network_scales, ...
        'balance_only_scales');
    if any(~ismember(options.balance_only_scales,options.network_scales))
        error('balance_only_scales must be a subset of network_scales.');
    end
end

if ~isfield(options,'output_root')
    options.output_root = '';
end
if ~isfield(options,'use_parallel')
    options.use_parallel = true;
end
if ~isfield(options,'max_workers') || isempty(options.max_workers)
    options.max_workers = 8;
end
if ~isfield(options,'resume')
    options.resume = true;
end
if ~isfield(options,'measure_ei_balance')
    options.measure_ei_balance = false;
end
if ~isfield(options,'balance_seed') || isempty(options.balance_seed)
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
    if ~isfield(options,option_name) || isempty(options.(option_name))
        options.(option_name) = balance_defaults.(name);
    end
end
if ~isfield(options,'cn4_max_workers') || isempty(options.cn4_max_workers)
    options.cn4_max_workers = 8;
end
if ~isscalar(options.cn4_max_workers) || ...
        options.cn4_max_workers < 1 || ...
        options.cn4_max_workers ~= floor(options.cn4_max_workers)
    error('cn4_max_workers must be a positive integer.');
end
if ~isscalar(options.max_workers) || options.max_workers < 1 || ...
        options.max_workers ~= floor(options.max_workers)
    error('max_workers must be a positive integer.');
end
options.use_parallel = logical(options.use_parallel);
options.resume = logical(options.resume);
options.measure_ei_balance = logical(options.measure_ei_balance);
if ~isfield(options,'training_checkpoint_batches') || ...
        isempty(options.training_checkpoint_batches)
    options.training_checkpoint_batches = zeros(1,0);
end
if ~isnumeric(options.training_checkpoint_batches) || ...
        any(options.training_checkpoint_batches < 1) || ...
        any(options.training_checkpoint_batches ~= ...
        floor(options.training_checkpoint_batches))
    error('training_checkpoint_batches must contain positive integers.');
end
options.training_checkpoint_batches = ...
    unique(options.training_checkpoint_batches(:)');
if ~isfield(options,'training_progress_every') || ...
        isempty(options.training_progress_every)
    options.training_progress_every = 100;
end
if ~isscalar(options.training_progress_every) || ...
        options.training_progress_every < 1 || ...
        options.training_progress_every ~= floor(options.training_progress_every)
    error('training_progress_every must be a positive integer.');
end
if ~options.measure_ei_balance && ~isempty(options.balance_only_scales)
    error('balance_only_scales requires measure_ei_balance=true.');
end

if isfield(options,'condition_mode') && ~isempty(options.condition_mode)
    condition_mode = lower(string(options.condition_mode));
    if ~isscalar(condition_mode)
        error('condition_mode must be ''selected'' or ''all''.');
    elseif condition_mode == "all"
        options.inhibitory_selection = 'ALL';
        options.homeostatic_selection = 'ALL';
    elseif condition_mode == "selected"
        if ~isfield(options,'selected_condition')
            error('A selected_condition such as AD is required.');
        end
        selected = upper(char(options.selected_condition));
        if numel(selected) ~= 2
            error('selected_condition must be a two-letter code such as AD.');
        end
        pcb_shuffle_conditions(selected(1),selected(2));
        options.inhibitory_selection = selected(1);
        options.homeostatic_selection = selected(2);
    else
        error('condition_mode must be ''selected'' or ''all''.');
    end
else
    if ~isfield(options,'inhibitory_selection') || ...
            isempty(options.inhibitory_selection)
        options.inhibitory_selection = 'ALL';
    end
    if ~isfield(options,'homeostatic_selection') || ...
            isempty(options.homeostatic_selection)
        options.homeostatic_selection = 'ALL';
    end
end
inhibitory = upper(string(options.inhibitory_selection));
homeostatic = upper(string(options.homeostatic_selection));
if ~isscalar(inhibitory) || ...
        ~(inhibitory == "ALL" || ...
        any(inhibitory == definitions.valid_inhibitory_codes))
    error('inhibitory_selection must be A, B, C or ALL.');
end
if ~isscalar(homeostatic) || ...
        ~(homeostatic == "ALL" || ...
        any(homeostatic == definitions.valid_homeostatic_codes))
    error('homeostatic_selection must be D, E, F, G or ALL.');
end
options.inhibitory_selection = char(inhibitory);
options.homeostatic_selection = char(homeostatic);
end

function values = validate_integer_selection(values,valid_values,label)
values = values(:)';
if ~isnumeric(values) || any(~isfinite(values)) || ...
        any(values ~= floor(values)) || ...
        any(~ismember(values,valid_values)) || ...
        numel(unique(values)) ~= numel(values)
    error('%s contains invalid or repeated values.',label);
end
end

function tag = make_selection_tag(options)
inhibitory = upper(string(options.inhibitory_selection));
homeostatic = upper(string(options.homeostatic_selection));
if isscalar(inhibitory) && inhibitory == "ALL" && ...
        isscalar(homeostatic) && homeostatic == "ALL"
    tag = 'all_conditions';
elseif isscalar(inhibitory) && isscalar(homeostatic) && ...
        inhibitory ~= "ALL" && homeostatic ~= "ALL"
    tag = char(inhibitory + homeostatic);
elseif isscalar(inhibitory) && inhibitory ~= "ALL" && ...
        isscalar(homeostatic) && homeostatic == "ALL"
    tag = sprintf('%s_group',lower(char(inhibitory)));
elseif isscalar(inhibitory) && inhibitory == "ALL" && ...
        isscalar(homeostatic) && homeostatic ~= "ALL"
    tag = sprintf('%s_group',lower(char(homeostatic)));
else
    error('Use one condition, one A/B/C or D/E/F/G group, or all conditions.');
end
end

function tag = make_scope_tag(selection_tag,network_scales,valid_scales)
if isequal(network_scales(:)',valid_scales(:)')
    tag = selection_tag;
    return;
end
scale_parts = arrayfun(@(cn) sprintf('cn%d',cn),network_scales, ...
    'UniformOutput',false);
tag = sprintf('%s_%s',selection_tag,strjoin(scale_parts,'_'));
end

function output_root = resolve_output_root(options,mode)
if ~isempty(options.output_root)
    [~,output_root] = pcb_results_paths(options.output_root);
    return;
end
code_dir = fileparts(mfilename('fullpath'));
study_dir = fullfile(code_dir,'pcb_shuffle_study');
if strcmp(mode,'quick')
    output_root = fullfile(study_dir,'cache','quick_validation');
else
    paths = pcb_results_paths();
    output_root = paths.scaling;
end
end

function ensure_repo_path_on_client()
% Modify the MATLAB path before entering parfor so thread workers inherit it.
code_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(code_dir);
addpath(repo_dir);
end

function [enabled,n_workers] = prepare_parallel_pool(options,n_tasks)
enabled = false;
n_workers = 1;
if ~logical(options.use_parallel)
    return;
end
if ~license('test','Distrib_Computing_Toolbox')
    warning('Parallel Computing Toolbox is unavailable; using serial runs.');
    return;
end
try
    pool = gcp('nocreate');
    if isempty(pool)
        requested = min([options.max_workers,n_tasks]);
        pool = parpool('Threads',requested);
    end
    n_workers = min([pool.NumWorkers,options.max_workers,n_tasks]);
    enabled = n_workers > 1;
catch exception
    warning('run_pcb_shuffle_all:ParallelPoolUnavailable', ...
        'Parallel pool could not start; using serial runs: %s', ...
        exception.message);
end
end

function n_workers = workers_for_scale(cn,pool_workers,options,n_tasks)
n_workers = min(pool_workers,n_tasks);
if cn == 4
    n_workers = min(n_workers,options.cn4_max_workers);
end
n_workers = max(1,n_workers);
end

function summary = summarize_runs(raw)
group_names = {'condition','inhibitory_code','homeostatic_code', ...
    'inhibitory_transmission','inhibitory_plasticity', ...
    'adaptive_threshold','l1_normalization','network_scale'};
metrics = {'lifetime_sparseness_E','population_sparseness_E', ...
    'RF_pairwise_correlation_RMS_E','mean_OSI_E','median_OSI_E', ...
    'fraction_OSI_ge_0_5','valid_OSI_fraction','mean_rate_E_Hz', ...
    'silent_neuron_fraction_E'};
summary = unique(raw(:,group_names),'rows','stable');
summary.n = zeros(height(summary),1);
for m = 1:numel(metrics)
    summary.([metrics{m} '_value']) = zeros(height(summary),1);
    summary.([metrics{m} '_sd']) = zeros(height(summary),1);
end

for row = 1:height(summary)
    selected = raw.condition == summary.condition(row) & ...
        raw.network_scale == summary.network_scale(row);
    summary.n(row) = nnz(selected);
    for m = 1:numel(metrics)
        values = raw.(metrics{m})(selected);
        summary.([metrics{m} '_value'])(row) = mean(values);
        summary.([metrics{m} '_sd'])(row) = std(values,0);
    end
end
summary = sortrows(summary,{'condition','network_scale'});
end

function files = plot_checkpoint_trajectory_figures(T,definitions, ...
        selection_tag,png_dir,mode,network_scale)
% For one cn value, show every recorded training checkpoint on the x-axis.
if ~strcmp(selection_tag,'all_conditions')
    files = strings(0,1);
    return;
end
groups = definitions.figure_groups;
group_names = definitions.figure_group_names;
metric_keys = {'RF_pairwise_correlation_RMS_E', ...
    'lifetime_sparseness_E','population_sparseness_E'};
metric_titles = {'E RF pairwise correlation RMS', ...
    'Lifetime sparseness','Population sparseness'};
metric_files = {'rf_pairwise_correlation','lifetime_sparseness', ...
    'population_sparseness'};
files = strings(numel(groups)*numel(metric_keys),1);
file_index = 0;
max_batch = max(T.batch);

for g = 1:numel(groups)
    conditions = string(groups{g});
    colors = lines(numel(conditions));
    for m = 1:numel(metric_keys)
        fig = figure('Visible','off','Color','white', ...
            'Position',[100 100 920 620]);
        ax = axes(fig); %#ok<LAXES>
        hold(ax,'on');
        for condition_index = 1:numel(conditions)
            selected = T.condition == conditions(condition_index);
            [batch,order] = sort(T.batch(selected));
            values = T.(metric_keys{m})(selected);
            plot(ax,batch,values(order),'-o', ...
                'LineWidth',1.8,'MarkerSize',5, ...
                'Color',colors(condition_index,:), ...
                'MarkerFaceColor','white', ...
                'DisplayName',char(conditions(condition_index)));
        end
        xlabel(ax,'Batch');
        ylabel(ax,metric_titles{m});
        title_handle = title(ax,sprintf('%s - %s (c_n=%d)', ...
            metric_titles{m},strrep(group_names{g},'_',' '), ...
            network_scale));
        title_handle.Color = [0.08 0.08 0.08];
        if strcmp(mode,'paper')
            xticks(ax,0:100:max_batch);
        else
            xticks(ax,unique(T.batch));
        end
        xlim(ax,[0 max_batch]);
        ylim(ax,[0 1]);
        grid(ax,'on');
        box(ax,'off');
        legend_handle = legend(ax,'Location','best', ...
            'TextColor',[0.1 0.1 0.1],'Color','white', ...
            'EdgeColor',[0.65 0.65 0.65]);
        legend_handle.AutoUpdate = 'off';
        set(ax,'FontName','Arial','FontSize',11,'LineWidth',1, ...
            'Color','white','XColor',[0.08 0.08 0.08], ...
            'YColor',[0.08 0.08 0.08]);

        file_index = file_index + 1;
        filename = sprintf('%02d_%s_%s_cn%d_by_batch_%s.png', ...
            file_index,group_names{g},metric_files{m}, ...
            network_scale,mode);
        files(file_index) = string(fullfile(png_dir,filename));
        exportgraphics(fig,char(files(file_index)),'Resolution',240, ...
            'BackgroundColor','white');
        close(fig);
    end
end
if file_index ~= 12
    error('The all-condition cn trajectory run must produce 12 charts.');
end
end

function files = plot_final_size_figures(T,definitions,selection_tag, ...
        png_dir,mode)
% Only the supervisor's final 12 charts are produced. Selected-condition
% runs save numerical results only and never create diagnostic figures.
if ~strcmp(selection_tag,'all_conditions')
    files = strings(0,1);
    return;
end
groups = definitions.figure_groups;
group_names = definitions.figure_group_names;

metric_keys = {'RF_pairwise_correlation_RMS_E', ...
    'lifetime_sparseness_E','population_sparseness_E'};
metric_titles = {'E RF pairwise correlation RMS', ...
    'Lifetime sparseness','Population sparseness'};
metric_files = {'rf_pairwise_correlation','lifetime_sparseness', ...
    'population_sparseness'};
files = strings(numel(groups)*numel(metric_keys),1);
file_index = 0;

for g = 1:numel(groups)
    conditions = string(groups{g});
    colors = lines(numel(conditions));
    for m = 1:numel(metric_keys)
        fig = figure('Visible','off','Color','white', ...
            'Position',[100 100 820 600]);
        ax = axes(fig); %#ok<LAXES>
        hold(ax,'on');
        for condition_index = 1:numel(conditions)
            selected = T.condition == conditions(condition_index);
            [cn,order] = sort(T.network_scale(selected));
            values = T.([metric_keys{m} '_value'])(selected);
            deviations = T.([metric_keys{m} '_sd'])(selected);
            errorbar(ax,cn,values(order),deviations(order),'-o', ...
                'LineWidth',1.8,'MarkerSize',6, ...
                'Color',colors(condition_index,:), ...
                'MarkerFaceColor','white', ...
                'DisplayName',char(conditions(condition_index)));
        end
        xlabel(ax,'Network scale c_n');
        ylabel(ax,metric_titles{m});
        title_handle = title(ax,sprintf('%s - %s',metric_titles{m}, ...
            strrep(group_names{g},'_',' ')));
        title_handle.Color = [0.08 0.08 0.08];
        xticks(ax,definitions.network_scales);
        xlim(ax,[0.8 4.2]);
        ylim(ax,[0 1]);
        grid(ax,'on');
        box(ax,'off');
        legend_handle = legend(ax,'Location','best', ...
            'TextColor',[0.1 0.1 0.1],'Color','white', ...
            'EdgeColor',[0.65 0.65 0.65]);
        legend_handle.AutoUpdate = 'off';
        set(ax,'FontName','Arial','FontSize',11,'LineWidth',1, ...
            'Color','white','XColor',[0.08 0.08 0.08], ...
            'YColor',[0.08 0.08 0.08]);

        file_index = file_index + 1;
        filename = sprintf('%02d_%s_%s_%s.png',file_index, ...
            group_names{g},metric_files{m},mode);
        files(file_index) = string(fullfile(png_dir,filename));
        exportgraphics(fig,char(files(file_index)),'Resolution',240, ...
            'BackgroundColor','white');
        close(fig);
    end
end
if file_index ~= 12
    error('The all-condition run must produce exactly 12 final charts.');
end
end

function ensure_directories(directories)
for i = 1:numel(directories)
    if ~exist(directories{i},'dir')
        mkdir(directories{i});
    end
end
end

function remove_stale_condition_metrics(run_dir,stem)
files = { ...
    fullfile(run_dir,'csv',[stem '.csv']), ...
    fullfile(run_dir,'csv','final_diagnostics', ...
        [stem '_final_E_cells.csv']), ...
    fullfile(run_dir,'csv','final_diagnostics', ...
        [stem '_final_E_response_counts.csv']), ...
    fullfile(run_dir,'png','final_diagnostics','OSI', ...
        [stem '_final_E_OSI.png']), ...
    fullfile(run_dir,'png','final_diagnostics','RF', ...
        [stem '_final_E_RF.png']), ...
    fullfile(run_dir,'png','final_diagnostics','Firing', ...
        [stem '_final_E_firing.png'])};
for i = 1:numel(files)
    if isfile(files{i})
        delete(files{i});
    end
end
end

function remove_stale_aggregate_plots(png_dir,mode)
% Direct children are standard checkpoint/size plots. Balance plots live
% in the ei_balance subdirectory and are deliberately left untouched.
files = dir(fullfile(png_dir,sprintf('*_%s.png',mode)));
for i = 1:numel(files)
    delete(fullfile(files(i).folder,files(i).name));
end
end
