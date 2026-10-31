function result = run_ps_factorial_condition(mode, condition, options)
%RUN_PS_FACTORIAL_CONDITION Run one revised PS factorial condition.
% This is the common simulation core used by the revised exp1/exp2 runner.
% Evaluation never updates weights or thresholds.

if nargin < 1 || isempty(mode)
    mode = 'paper';
end
if nargin < 2
    error('A condition structure is required.');
end
if nargin < 3
    options = struct();
end
options = fill_options(options);
condition = normalize_condition(condition);
cfg = make_config(mode);
cfg = apply_config_overrides(cfg, options);

code_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(code_dir);
% Thread workers cannot modify the MATLAB path. The aggregate runner adds
% repo_dir on the client before parfor; direct serial calls still add it here.
if exist('generate_image_patches','file') ~= 2
    addpath(repo_dir);
end

rng(cfg.seed, 'twister');
IMAGES = load_training_images(repo_dir);

if options.measure_standard_metrics || options.run_window_experiment
    training_rng = rng;
    rng(cfg.eval_patch_seed, 'twister');
    R_X_eval = generate_scaled_image_patches(IMAGES, cfg, cfg.N_eval);
    R_X_eval(R_X_eval > 100) = 100;
    rng(training_rng);
else
    R_X_eval = [];
end

if options.measure_standard_metrics && (options.measure_extended_metrics || ...
        options.measure_final_diagnostics)
    [R_X_grating, grating_meta] = make_grating_inputs(cfg);
else
    R_X_grating = [];
    grating_meta = struct();
end
state = initialise_network(cfg);
[state,start_batch] = restore_training_checkpoint( ...
    state,cfg,condition,options);
checkpoints = cfg.checkpoints(:);
result = initialise_result(cfg, condition, checkpoints, options);
if options.save_final_state
    result.initial_state = state;
end

fprintf('\nCondition: %s (%s)\n', condition.id, cfg.mode);
next_checkpoint = 1;
if start_batch == 1 && checkpoints(1) == 0
    if options.measure_standard_metrics
        result = measure_checkpoint(result, 1, state, R_X_eval, ...
            R_X_grating, grating_meta, cfg, condition);
    end
    if options.run_window_experiment
        result.window_analysis = initialise_window_analysis(cfg, checkpoints);
        result.window_analysis = measure_window_checkpoint( ...
            result.window_analysis, 1, state, R_X_eval, cfg, condition);
    end
    print_progress(result, 1, options);
    next_checkpoint = 2;
elseif options.run_window_experiment
    result.window_analysis = initialise_window_analysis(cfg, checkpoints);
end
for batch = start_batch:cfg.N_n
    learn_weights = batch > cfg.N_q;
    stats = train_one_batch(state, IMAGES, cfg, condition, learn_weights);

    % 只在正式训练的最后一个 batch 保存逐神经元自然图像放电率。
    % stats.R_E/R_I 是每个神经元在本 batch 的 100 张自然图像 patch 上
    % 平均得到的脉冲数；除以单张刺激时长 cfg.T 后转换为 Hz。
    if batch == cfg.N_n
        result.final_training_rate_E_Hz = single(stats.R_E/cfg.T);
        result.final_training_rate_I_Hz = single(stats.R_I/cfg.T);
    end

    if learn_weights
        state = apply_weight_plasticity(state, stats, cfg, condition);
    end
    if condition.threshold_enabled
        state = apply_threshold_plasticity(state, stats, cfg);
    end

    if batch == cfg.halve_at_1 || batch == cfg.halve_at_2
        state.learning_scale = state.learning_scale / 2;
    end

    if ismember(batch,options.training_checkpoint_batches)
        save_training_checkpoint(state,batch,cfg,condition,options,stats);
    end

    if next_checkpoint <= numel(checkpoints) && ...
            batch == checkpoints(next_checkpoint)
        if options.measure_standard_metrics
            result = measure_checkpoint(result, next_checkpoint, state, ...
                R_X_eval, R_X_grating, grating_meta, cfg, condition);
        end
        if options.run_window_experiment
            result.window_analysis = measure_window_checkpoint( ...
                result.window_analysis, next_checkpoint, state, ...
                R_X_eval, cfg, condition);
        end
        print_progress(result, next_checkpoint, options);
        next_checkpoint = next_checkpoint + 1;
    elseif ~options.measure_standard_metrics && ...
            mod(batch,options.training_progress_every) == 0
        fprintf('  batch %4d | training only; evaluation skipped\n',batch);
    end
end

if options.measure_ei_balance
    fprintf('Measuring frozen-network E/I balance diagnostics...\n');
    result.ei_balance = measure_ei_balance( ...
        state,IMAGES,cfg,condition,options);
end

if options.measure_standard_metrics
    if result.batch(1) == 0
        result.initial_ps = result.PS_E(1);
        result.delta_ps = result.PS_E(end) - result.PS_E(1);
    else
        result.initial_ps = NaN;
        result.delta_ps = NaN;
    end
    result.final_ps = result.PS_E(end);
end
if options.save_final_state
    result.final_state = state;
end
end

function options = fill_options(options)
defaults.run_window_experiment = false;
defaults.save_final_state = false;
defaults.measure_standard_metrics = true;
defaults.measure_extended_metrics = true;
defaults.measure_final_diagnostics = false;
defaults.measure_ei_balance = false;
defaults.balance_only = false;
defaults.balance_neuron_count = 40;
defaults.balance_example_candidates = 1000;
defaults.balance_population_candidates = 100;
defaults.balance_scatter_candidates = 100;
defaults.balance_tight_candidates = 1000;
defaults.balance_tight_top_k = 10;
defaults.balance_max_lag_ms = 50;
defaults.balance_transient_ms = 100;
defaults.balance_seed_override = [];
defaults.training_checkpoint_batches = zeros(1,0);
defaults.training_progress_every = 100;
defaults.training_checkpoint_dir = '';
defaults.training_checkpoint_stem = '';
defaults.balance_checkpoint_file = '';
defaults.resume_training = false;
defaults.N_n_override = [];
defaults.eval_every_override = [];
defaults.checkpoints_override = [];
defaults.window_steps_override = [];
defaults.window_repeats_override = [];
defaults.seed_override = [];
defaults.eval_patch_seed_override = [];
defaults.eval_dynamics_seed_override = [];
defaults.eval_grating_seed_override = [];
defaults.finite_sparseness_correction = true;
defaults.network_scale_override = 1;
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i})
        options.(names{i}) = defaults.(names{i});
    end
end
end

function cfg = apply_config_overrides(cfg, options)
if ~isempty(options.seed_override)
    cfg.seed = options.seed_override;
end
if ~isempty(options.eval_patch_seed_override)
    cfg.eval_patch_seed = options.eval_patch_seed_override;
end
if ~isempty(options.eval_dynamics_seed_override)
    cfg.eval_dynamics_seed = options.eval_dynamics_seed_override;
end
if ~isempty(options.eval_grating_seed_override)
    cfg.eval_grating_seed = options.eval_grating_seed_override;
end
if ~isempty(options.balance_seed_override)
    cfg.balance_seed = options.balance_seed_override;
end
cfg.finite_sparseness_correction = ...
    logical(options.finite_sparseness_correction);
cfg = apply_network_scale(cfg,options.network_scale_override);
if ~isempty(options.N_n_override)
    cfg.N_n = options.N_n_override;
end
if ~isempty(options.eval_every_override)
    cfg.eval_every = options.eval_every_override;
end
if ~isempty(options.window_steps_override)
    cfg.window_steps = options.window_steps_override;
end
if ~isempty(options.window_repeats_override)
    cfg.window_repeats = options.window_repeats_override;
end
if ~isempty(options.checkpoints_override)
    cfg.checkpoints = unique(options.checkpoints_override);
else
    cfg.checkpoints = unique([0:cfg.eval_every:cfg.N_n cfg.N_n]);
end
cfg.checkpoints = cfg.checkpoints( ...
    cfg.checkpoints >= 0 & cfg.checkpoints <= cfg.N_n);
if isempty(cfg.checkpoints)
    error('At least one valid checkpoint is required.');
end
end

function cfg = apply_network_scale(cfg,network_scale)
if ~isscalar(network_scale) || ~isfinite(network_scale) || ...
        network_scale ~= floor(network_scale) || ...
        ~ismember(network_scale,1:4)
    error('network_scale_override must be one of 1, 2, 3 or 4.');
end

cfg.network_scale = network_scale;
cfg.N_X = cfg.N_pixels * network_scale;
cfg.N_E = cfg.N_E * network_scale;
cfg.N_I = cfg.N_I * network_scale;

% PCB Methods 2.8.5: L1 norms scale as sqrt(N). Equation 18 then
% requires the corresponding synaptic learning rates to scale by the
% same ratio. Threshold learning rates and all other parameters stay fixed.
scale = sqrt(network_scale);
l1_fields = {'L1_EX','L1_EE','L1_EI','L1_IX','L1_IE','L1_II'};
eta_fields = {'eta_EX','eta_EE','eta_EI','eta_IX','eta_IE','eta_II'};
for i = 1:numel(l1_fields)
    cfg.(l1_fields{i}) = cfg.(l1_fields{i}) * scale;
end
for i = 1:numel(eta_fields)
    cfg.(eta_fields{i}) = cfg.(eta_fields{i}) * scale;
end
end

function condition = normalize_condition(condition)
% Accept the former Exp2 fields while the new PCB-shuffle runner uses the
% explicit transmission/plasticity/homeostasis fields.
if ~isfield(condition, 'inhibitory_transmission_enabled')
    if ~isfield(condition, 'inhibition_enabled')
        error('Condition must define inhibitory transmission.');
    end
    condition.inhibitory_transmission_enabled = ...
        logical(condition.inhibition_enabled);
end
if ~isfield(condition, 'inhibitory_plasticity_enabled')
    condition.inhibitory_plasticity_enabled = ...
        condition.inhibitory_transmission_enabled;
end
if ~isfield(condition, 'threshold_enabled')
    error('Condition must define threshold_enabled.');
end
if ~isfield(condition, 'l1_normalization_enabled')
    condition.l1_normalization_enabled = true;
end
if condition.inhibitory_plasticity_enabled && ...
        ~condition.inhibitory_transmission_enabled
    error(['I->E plasticity cannot be enabled while I->E transmission ' ...
        'is disabled in the A/B/C design.']);
end
end

function IMAGES = load_training_images(repo_dir)
% Reuse the immutable image dataset across conditions on each MATLAB
% process/parallel worker. This removes repeated MAT-file reads without
% changing patch generation or any random-number stream.
persistent cached_repo cached_images
if isempty(cached_images) || ~strcmp(cached_repo,repo_dir)
    data = load(fullfile(repo_dir,'VanHateren_DoG_small.mat'), ...
        'IMAGES_DoG');
    cached_images = data.IMAGES_DoG;
    cached_repo = repo_dir;
end
IMAGES = cached_images;
end

function cfg = make_config(mode)
cfg.mode = lower(char(mode));
cfg.seed = 20250718;
cfg.eval_patch_seed = 12025;
cfg.eval_dynamics_seed = 22025;
cfg.eval_grating_seed = 1202603;
cfg.balance_seed = 3202603;
cfg.dt = 1e-3;
cfg.tau_m = 10e-3;
cfg.tau_i = 20e-3;
cfg.tau_l = 50e-3;
cfg.V_min = -10;
cfg.rho_E_Hz = 2;
cfg.rho_I_Hz = 4;
cfg.theta_E0 = 10;
cfg.theta_I0 = 10;
cfg.noise_sigma = 1;
cfg.strong_connection_probability = 0.2;

cfg.eta_EX = 2e-4; cfg.eta_EE = 1e-4; cfg.eta_EI = 9e-2;
cfg.eta_IX = 3e-3; cfg.eta_IE = 4e-2; cfg.eta_II = 6e-2;
cfg.eta_tE = 1; cfg.eta_tI = 1;
cfg.L1_EX = 100; cfg.L1_EE = 10; cfg.L1_EI = 120;
cfg.L1_IX = 80; cfg.L1_IE = 240; cfg.L1_II = 120;

switch cfg.mode
    case 'paper'
        cfg.N_pixels = 16^2; cfg.N_X = cfg.N_pixels;
        cfg.N_E = 20^2; cfg.N_I = cfg.N_E/4;
        cfg.N_b = 100; cfg.N_eval = 100; cfg.N_t = 400;
        cfg.N_q = 100; cfg.N_n = 1200; cfg.eval_every = 100;
        cfg.halve_at_1 = 400; cfg.halve_at_2 = 800;
        cfg.window_steps = [50 100 200 400];
        cfg.window_repeats = 20;
        cfg.grating_eval_repeats = 5;
        cfg.grating_orientations_deg = 0:22.5:157.5;
        cfg.grating_spatial_frequencies = [1 2 3];
        cfg.grating_phases_deg = [0 90 180 270];
    case 'quick'
        cfg.N_pixels = 8^2; cfg.N_X = cfg.N_pixels;
        cfg.N_E = 10^2; cfg.N_I = cfg.N_E/4;
        cfg.N_b = 8; cfg.N_eval = 16; cfg.N_t = 50;
        cfg.N_q = 2; cfg.N_n = 6; cfg.eval_every = 2;
        cfg.halve_at_1 = 3; cfg.halve_at_2 = 5;
        cfg.window_steps = unique(max(1, round(cfg.N_t * [0.125 0.25 0.5 1])));
        cfg.window_repeats = 3;
        cfg.grating_eval_repeats = 1;
        cfg.grating_orientations_deg = 0:45:135;
        cfg.grating_spatial_frequencies = [1 2];
        cfg.grating_phases_deg = [0 90];

        scale_X = sqrt(cfg.N_X / 16^2);
        scale_E = sqrt(cfg.N_E / 20^2);
        scale_I = sqrt(cfg.N_I / 100);
        cfg.L1_EX = cfg.L1_EX * scale_X;
        cfg.L1_IX = cfg.L1_IX * scale_X;
        cfg.L1_EE = cfg.L1_EE * scale_E;
        cfg.L1_IE = cfg.L1_IE * scale_E;
        cfg.L1_EI = cfg.L1_EI * scale_I;
        cfg.L1_II = cfg.L1_II * scale_I;
    otherwise
        error('Mode must be ''quick'' or ''paper''.');
end

cfg.T = cfg.N_t * cfg.dt;
cfg.rho_E = cfg.rho_E_Hz * cfg.T;
cfg.rho_I = cfg.rho_I_Hz * cfg.T;
cfg.checkpoints = unique([0:cfg.eval_every:cfg.N_n cfg.N_n]);
end

function state = initialise_network(cfg)
p = cfg.strong_connection_probability;
state.W_EX = random_weights(cfg.N_E, 2*cfg.N_X, p, true, cfg.L1_EX);
state.W_EE = random_weights(cfg.N_E, cfg.N_E, p, false, cfg.L1_EE);
state.W_EI = random_weights(cfg.N_E, cfg.N_I, p, true, cfg.L1_EI);
state.W_IX = random_weights(cfg.N_I, 2*cfg.N_X, p, true, cfg.L1_IX);
state.W_IE = random_weights(cfg.N_I, cfg.N_E, p, true, cfg.L1_IE);
state.W_II = random_weights(cfg.N_I, cfg.N_I, p, false, cfg.L1_II);
state.theta_E = cfg.theta_E0 * ones(cfg.N_E, 1);
state.theta_I = cfg.theta_I0 * ones(cfg.N_I, 1);
state.learning_scale = 1;
end

function W = random_weights(n_post, n_pre, p, allow_self, target_norm)
W = 1 + 0.5 * randn(n_post, n_pre);
weak = rand(n_post, n_pre) > p;
W(weak) = W(weak) / 100;
W(W < 0) = 0;
if ~allow_self && n_post == n_pre
    W(1:n_post+1:end) = 0;
end
W = W ./ sum(W, 2) * target_norm;
end

function [state,start_batch] = restore_training_checkpoint( ...
        state,cfg,condition,options)
start_batch = 1;
if ~options.resume_training || isempty(options.training_checkpoint_dir) || ...
        isempty(options.training_checkpoint_stem)
    return;
end
batches = sort(options.training_checkpoint_batches,'descend');
for batch = batches
    filename = training_checkpoint_path(options,batch);
    if ~isfile(filename)
        continue;
    end
    try
        loaded = load(filename,'checkpoint');
        checkpoint = loaded.checkpoint;
        valid = checkpoint.schema_version == 1 && ...
            string(checkpoint.condition) == string(condition.id) && ...
            checkpoint.network_scale == cfg.network_scale && ...
            checkpoint.training_seed == cfg.seed && ...
            checkpoint.final_batch == cfg.N_n && ...
            checkpoint.batch == batch;
        if ~valid
            continue;
        end
        state = checkpoint.state;
        rng(checkpoint.rng_state);
        start_batch = batch + 1;
        fprintf('  resumed training state from batch %d\n',batch);
        return;
    catch exception
        warning('run_ps_factorial_condition:BadTrainingCheckpoint', ...
            'Ignoring unreadable checkpoint %s: %s',filename, ...
            exception.message);
    end
end
end

function save_training_checkpoint(state,batch,cfg,condition,options,stats)
if isempty(options.training_checkpoint_dir) || ...
        isempty(options.training_checkpoint_stem)
    return;
end
if ~exist(options.training_checkpoint_dir,'dir')
    mkdir(options.training_checkpoint_dir);
end
checkpoint.schema_version = 1;
checkpoint.condition = string(condition.id);
checkpoint.network_scale = cfg.network_scale;
checkpoint.training_seed = cfg.seed;
checkpoint.final_batch = cfg.N_n;
checkpoint.batch = batch;
checkpoint.state = state;
checkpoint.rng_state = rng;
% 将当前训练 batch 的逐神经元自然图像放电率与网络状态一起保存。
% 这使第 1200 个 batch 的散点图能够直接读取真实训练活动，而不需要
% 用冻结网络的额外测试刺激代替。
checkpoint.training_rate_E_Hz = single(stats.R_E/cfg.T);
checkpoint.training_rate_I_Hz = single(stats.R_I/cfg.T);
filename = training_checkpoint_path(options,batch);
temporary = [tempname(options.training_checkpoint_dir) '.mat'];
save(temporary,'checkpoint','-v7');
movefile(temporary,filename,'f');
fprintf('  saved restart checkpoint at batch %d\n',batch);
end

function filename = training_checkpoint_path(options,batch)
filename = fullfile(options.training_checkpoint_dir,sprintf( ...
    '%s_batch%04d.mat',options.training_checkpoint_stem,batch));
end

function result = initialise_result(cfg, condition, checkpoints, options)
n = numel(checkpoints);
result.cfg = cfg;
result.condition = condition;
result.options = options;
result.batch = checkpoints;
result.phase = strings(n, 1);
result.final_training_rate_E_Hz = nan(cfg.N_E,1,'single');
result.final_training_rate_I_Hz = nan(cfg.N_I,1,'single');
if condition.threshold_enabled
    result.phase(checkpoints <= cfg.N_q) = "threshold-adaptation";
    result.phase(checkpoints > cfg.N_q) = "weight+threshold";
else
    result.phase(checkpoints <= cfg.N_q) = "fixed-threshold";
    result.phase(checkpoints > cfg.N_q) = "weight-only";
end

if options.measure_standard_metrics
    result.PS_E = nan(n,1);
    result.lifetime_sparseness_E = nan(n,1);
    result.RF_correlation_RMS_E = nan(n,1);
    if options.measure_extended_metrics
        result.PS_I = nan(n,1);
        result.lifetime_sparseness_I = nan(n,1);
        result.silent_neuron_fraction_E = nan(n,1);
        result.silent_neuron_fraction_I = nan(n,1);
        result.mean_rate_E_Hz = nan(n,1);
        result.mean_rate_I_Hz = nan(n,1);
        result.silent_image_fraction_E = nan(n,1);
        result.active_E_fraction = nan(n,1);
        result.response_correlation_RMS_E = nan(n,1);
        result.mean_theta_E = nan(n,1);
        result.mean_theta_I = nan(n,1);
        result.mean_OSI_E = nan(n,1);
        result.median_OSI_E = nan(n,1);
        result.fraction_OSI_ge_0_5 = nan(n,1);
        result.valid_OSI_fraction = nan(n,1);
        result.OSI_E = nan(cfg.N_E, n, 'single');
        result.RF_E = nan(cfg.N_E, cfg.N_pixels, n, 'single');
        result.RF_I = nan(cfg.N_I, cfg.N_pixels, n, 'single');
    end
end
end

function result = measure_checkpoint(result, index, state, R_X_eval, ...
        R_X_grating, grating_meta, cfg, condition)
training_rng = rng;

rng(cfg.eval_dynamics_seed, 'twister');
[R_E, R_I] = simulate_response(state, R_X_eval, cfg, condition);
[result.PS_E(index), ~, silent_image_fraction_E] = ...
    modified_treves_rolls(R_E, cfg.finite_sparseness_correction);
[result.lifetime_sparseness_E(index), ~, silent_neuron_fraction_E] = ...
    modified_treves_rolls(R_E', cfg.finite_sparseness_correction);

RF_E = effective_receptive_fields(state.W_EX,cfg);
result.RF_correlation_RMS_E(index) = ...
    rms_offdiagonal_correlation(RF_E);

if result.options.measure_extended_metrics
    [result.PS_I(index), ~, ~] = modified_treves_rolls( ...
        R_I, cfg.finite_sparseness_correction);
    [result.lifetime_sparseness_I(index), ~, ...
            result.silent_neuron_fraction_I(index)] = ...
        modified_treves_rolls(R_I', cfg.finite_sparseness_correction);
    result.silent_image_fraction_E(index) = silent_image_fraction_E;
    result.silent_neuron_fraction_E(index) = silent_neuron_fraction_E;
    result.mean_rate_E_Hz(index) = mean(R_E(:)) / cfg.T;
    result.mean_rate_I_Hz(index) = mean(R_I(:)) / cfg.T;
    result.active_E_fraction(index) = mean(R_E(:) > 0);
    result.response_correlation_RMS_E(index) = ...
        rms_offdiagonal_correlation(R_E);

    RF_I = effective_receptive_fields(state.W_IX,cfg);
    result.RF_E(:,:,index) = single(RF_E);
    result.RF_I(:,:,index) = single(RF_I);
    result.mean_theta_E(index) = mean(state.theta_E);
    result.mean_theta_I(index) = mean(state.theta_I);

    grating_counts = zeros(cfg.N_E, size(R_X_grating,2));
    for repeat = 1:cfg.grating_eval_repeats
        rng(cfg.eval_grating_seed + repeat, 'twister');
        [counts_E, ~] = simulate_response( ...
            state, R_X_grating, cfg, condition);
        grating_counts = grating_counts + counts_E;
    end
    grating_rates = grating_counts / cfg.grating_eval_repeats / cfg.T;
    [OSI_E,tuning_E,preferred_sf_E,preferred_orientation_E] = ...
        calculate_osi(grating_rates, grating_meta, cfg);
    valid = isfinite(OSI_E);
    result.OSI_E(:,index) = single(OSI_E);
    result.mean_OSI_E(index) = mean(OSI_E(valid), 'omitnan');
    result.median_OSI_E(index) = median(OSI_E(valid), 'omitnan');
    result.fraction_OSI_ge_0_5(index) = mean(OSI_E(valid) >= 0.5);
    result.valid_OSI_fraction(index) = mean(valid);
end

if result.options.measure_final_diagnostics && index == numel(result.batch)
    if ~result.options.measure_extended_metrics
        grating_counts = zeros(cfg.N_E, size(R_X_grating,2));
        for repeat = 1:cfg.grating_eval_repeats
            rng(cfg.eval_grating_seed + repeat, 'twister');
            [counts_E, ~] = simulate_response( ...
                state, R_X_grating, cfg, condition);
            grating_counts = grating_counts + counts_E;
        end
        grating_rates = grating_counts / cfg.grating_eval_repeats / cfg.T;
        [OSI_E,tuning_E,preferred_sf_E,preferred_orientation_E] = ...
            calculate_osi(grating_rates, grating_meta, cfg);
    end
    valid = isfinite(OSI_E);
    diagnostics.RF_E = single(RF_E);
    diagnostics.response_counts_E = uint16(R_E);
    diagnostics.mean_rate_E_Hz = single(mean(R_E,2) / cfg.T);
    diagnostics.active_stimulus_fraction_E = single(mean(R_E > 0,2));
    diagnostics.silent_neuron_fraction_E = mean(sum(R_E,2) == 0);
    diagnostics.OSI_E = single(OSI_E);
    diagnostics.orientation_tuning_rates_E = single(tuning_E);
    diagnostics.orientations_deg = cfg.grating_orientations_deg;
    diagnostics.preferred_spatial_frequency_E = single(preferred_sf_E);
    diagnostics.preferred_orientation_deg_E = ...
        single(preferred_orientation_E);
    diagnostics.mean_OSI_E = mean(OSI_E(valid),'omitnan');
    diagnostics.median_OSI_E = median(OSI_E(valid),'omitnan');
    diagnostics.fraction_OSI_ge_0_5 = mean(OSI_E(valid) >= 0.5);
    diagnostics.valid_OSI_fraction = mean(valid);
    result.final_diagnostics = diagnostics;
end

rng(training_rng);
end

function value = rms_offdiagonal_correlation(rows)
C = corrcoef(rows');
mask = ~eye(size(C)) & isfinite(C);
if any(mask(:))
    value = sqrt(mean(C(mask).^2));
else
    value = NaN;
end
end

function RF = effective_receptive_fields(W,cfg)
W_on = W(:,1:cfg.N_X);
W_off = W(:,cfg.N_X+1:end);
if cfg.network_scale > 1
    n_post = size(W,1);
    W_on = reshape(sum(reshape(W_on,n_post, ...
        cfg.network_scale,cfg.N_pixels),2),n_post,cfg.N_pixels);
    W_off = reshape(sum(reshape(W_off,n_post, ...
        cfg.network_scale,cfg.N_pixels),2),n_post,cfg.N_pixels);
end
RF = W_on - W_off;
end

function [R_E, R_I] = simulate_response(state, R_X, cfg, condition)
n_stim = size(R_X, 2);
V_E = zeros(cfg.N_E, n_stim);
V_I = zeros(cfg.N_I, n_stim);
R_E = zeros(cfg.N_E, n_stim);
R_I = zeros(cfg.N_I, n_stim);
exp_m = exp(-cfg.dt / cfg.tau_m);
R_XDT = R_X * cfg.dt;
use_inhibition = condition.inhibitory_transmission_enabled;

for t = 1:cfg.N_t
    S_X = rand(2*cfg.N_X, n_stim) < R_XDT;
    S_E = V_E > state.theta_E;
    S_I = V_I > state.theta_I;
    V_E(S_E) = 0;
    V_I(S_I) = 0;
    R_E = R_E + S_E;
    R_I = R_I + S_I;

    if use_inhibition
        V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
            - state.W_EI*S_I + cfg.noise_sigma*randn(cfg.N_E,n_stim);
    else
        V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
            + cfg.noise_sigma*randn(cfg.N_E,n_stim);
    end
    V_I = V_I*exp_m + state.W_IX*S_X + state.W_IE*S_E ...
        - state.W_II*S_I + cfg.noise_sigma*randn(cfg.N_I,n_stim);
    V_E(V_E < cfg.V_min) = cfg.V_min;
    V_I(V_I < cfg.V_min) = cfg.V_min;
end
end

function balance = measure_ei_balance(state,IMAGES,cfg,condition,options)
% Measure Fig. 8/9-style balance after training without changing the state.
training_rng = rng;
cleanup_rng = onCleanup(@() rng(training_rng)); %#ok<NASGU>

RF_E = effective_receptive_fields(state.W_EX,cfg);
[stage,balance] = restore_balance_checkpoint(cfg,condition,options);
if stage == "tight_complete"
    fprintf('  resumed completed Loose/Tight balance diagnostics\n');
    return;
end

if stage ~= "loose_complete"
    requested_neurons = validate_positive_integer( ...
        options.balance_neuron_count,'balance_neuron_count');
    RF_centered = RF_E - mean(RF_E,2);
    rf_valid = all(isfinite(RF_E),2) & ...
        sqrt(sum(RF_centered.^2,2)) > 0;
    valid_neurons = find(rf_valid);
    if isempty(valid_neurons)
        error('No E neuron had a finite, nonconstant receptive field.');
    end
    n_neurons = min(requested_neurons,numel(valid_neurons));
    if n_neurons < requested_neurons
        warning('run_ps_factorial_condition:FewerValidRFs', ...
            ['Only %d E neurons had finite, nonconstant receptive fields; ' ...
            'using all of them instead of the requested %d.'], ...
            n_neurons,requested_neurons);
    end
    n_example = validate_positive_integer( ...
        options.balance_example_candidates,'balance_example_candidates');
    n_population = validate_positive_integer( ...
        options.balance_population_candidates, ...
        'balance_population_candidates');
    n_scatter = validate_positive_integer( ...
        options.balance_scatter_candidates,'balance_scatter_candidates');

    rng(cfg.balance_seed,'twister');
    selected = randperm(numel(valid_neurons),n_neurons);
    population_neurons = sort(valid_neurons(selected));
    example_neuron = population_neurons(1);
    tight_neuron = valid_neurons(randi(numel(valid_neurons)));
    assert(all(rf_valid(population_neurons)) && rf_valid(tight_neuron), ...
        'RF-matched balance analysis selected an invalid receptive field.');
    fprintf('  RF matching: %d/%d E neurons valid; %d selected\n', ...
        numel(valid_neurons),cfg.N_E,n_neurons);

    rng(cfg.balance_seed + 1,'twister');
    R_example_candidates = generate_scaled_image_patches( ...
        IMAGES,cfg,n_example);
    rng(cfg.balance_seed + 2,'twister');
    R_population_candidates = generate_scaled_image_patches( ...
        IMAGES,cfg,n_population);
    rng(cfg.balance_seed + 3,'twister');
    R_scatter_candidates = generate_scaled_image_patches( ...
        IMAGES,cfg,n_scatter);

    P_example = signed_base_patches(R_example_candidates,cfg);
    P_population = signed_base_patches(R_population_candidates,cfg);
    P_scatter = signed_base_patches(R_scatter_candidates,cfg);

    C_example = normalised_patch_correlations( ...
        RF_E(example_neuron,:),P_example);
    example_indices = select_low_middle_high(C_example);
    example_correlations = C_example(example_indices);

    C_population = normalised_patch_correlations( ...
        RF_E(population_neurons,:),P_population);
    population_indices = select_low_middle_high(C_population);
    C_scatter = normalised_patch_correlations( ...
        RF_E(population_neurons,:),P_scatter);
    scatter_indices = select_low_middle_high(C_scatter);
    row_ids = (1:n_neurons)';
    population_correlations = selected_correlations( ...
        C_population,population_indices,row_ids);
    scatter_correlations = selected_correlations( ...
        C_scatter,scatter_indices,row_ids);

    population_flat_indices = [population_indices(:,1); ...
        population_indices(:,2);population_indices(:,3)];
    scatter_flat_indices = [scatter_indices(:,1); ...
        scatter_indices(:,2);scatter_indices(:,3)];
    R_selected = [R_example_candidates(:,example_indices) ...
        R_population_candidates(:,population_flat_indices) ...
        R_scatter_candidates(:,scatter_flat_indices)];
    target_neurons = [repmat(example_neuron,3,1); ...
        repmat(population_neurons,3,1); ...
        repmat(population_neurons,3,1)];

    rng(cfg.balance_seed + 100,'twister');
    traces = simulate_balance_shadow( ...
        state,R_selected,cfg,condition,target_neurons);
    example_rows = 1:3;
    population_rows = 4:(3 + 3*n_neurons);
    scatter_rows = (4 + 3*n_neurons):(3 + 6*n_neurons);

    balance.schema_version = 2;
    balance.condition = string(condition.id);
    balance.network_scale = cfg.network_scale;
    balance.measurement_batch = cfg.N_n;
    balance.dt_ms = cfg.dt*1000;
    balance.time_ms = single((1:cfg.N_t)*cfg.dt*1000);
    balance.threshold_removed_from_shadow_only = true;
    balance.network_dynamics_used_normal_thresholds = true;
    balance.learning_and_homeostasis_frozen = true;
    balance.rf_valid_mask = rf_valid;
    balance.rf_valid_neuron_ids = valid_neurons;
    balance.zero_rf_neuron_ids = find(~rf_valid);
    balance.zero_rf_fraction = mean(~rf_valid);
    balance.requested_neuron_count = requested_neurons;
    balance.selected_neuron_count = n_neurons;
    balance.class_names = ["Low (anti-correlated)" ...
        "Middle (near zero)" "High (correlated)"];
    balance.tight_neuron_id = tight_neuron;

    balance.example.neuron_id = example_neuron;
    balance.example.RF = single(RF_E(example_neuron,:));
    balance.example.candidate_count = n_example;
    balance.example.patch_indices = example_indices;
    balance.example.patch_correlations = single(example_correlations);
    balance.example.signed_patches = ...
        single(P_example(:,example_indices));
    balance.example.threshold = single(state.theta_E(example_neuron));
    balance.example = attach_trace_fields( ...
        balance.example,traces,example_rows);

    balance.loose.neuron_ids = population_neurons;
    balance.loose.candidate_count = n_population;
    balance.loose.patch_indices = population_indices;
    balance.loose.patch_correlations = single(population_correlations);
    balance.loose.thresholds = single(state.theta_E(population_neurons));
    balance.loose = attach_trace_fields( ...
        balance.loose,traces,population_rows);
    balance.loose = summarise_loose_population( ...
        balance.loose,n_neurons);

    balance.loose_scatter.neuron_ids = population_neurons;
    balance.loose_scatter.candidate_count = n_scatter;
    balance.loose_scatter.patch_indices = scatter_indices;
    balance.loose_scatter.patch_correlations = ...
        single(scatter_correlations);
    balance.loose_scatter.thresholds = ...
        single(state.theta_E(population_neurons));
    balance.loose_scatter = attach_trace_fields( ...
        balance.loose_scatter,traces,scatter_rows);
    balance.loose_scatter = summarise_loose_population( ...
        balance.loose_scatter,n_neurons);
    save_balance_checkpoint(balance,"loose_complete", ...
        cfg,condition,options);
    fprintf('  saved Loose-balance checkpoint\n');
else
    fprintf('  resumed Loose-balance checkpoint\n');
    population_neurons = balance.loose.neuron_ids;
    n_neurons = numel(population_neurons);
    tight_neuron = balance.tight_neuron_id;
end

n_tight = validate_positive_integer( ...
    options.balance_tight_candidates,'balance_tight_candidates');
n_top = min(validate_positive_integer( ...
    options.balance_tight_top_k,'balance_tight_top_k'),n_tight);
rng(cfg.balance_seed + 4,'twister');
R_tight_candidates = generate_scaled_image_patches(IMAGES,cfg,n_tight);
P_tight = signed_base_patches(R_tight_candidates,cfg);
C_tight = normalised_patch_correlations( ...
    RF_E(tight_neuron,:),P_tight);
tight_sort_values = C_tight;
tight_sort_values(~isfinite(tight_sort_values)) = -Inf;
[~,tight_order] = sort(tight_sort_values,'descend');
tight_indices = tight_order(1:n_top);

rng(cfg.balance_seed + 200,'twister');
tight_traces = simulate_balance_shadow(state, ...
    R_tight_candidates(:,tight_indices),cfg,condition, ...
    repmat(tight_neuron,n_top,1));
balance.tight.neuron_id = tight_neuron;
balance.tight.RF = single(RF_E(tight_neuron,:));
balance.tight.candidate_count = n_tight;
balance.tight.patch_indices = tight_indices;
balance.tight.patch_correlations = single(C_tight(tight_indices));
balance.tight.signed_patches = single(P_tight(:,tight_indices));
balance.tight.threshold = single(state.theta_E(tight_neuron));
balance.tight = attach_trace_fields(balance.tight,tight_traces,1:n_top);

transient_steps = min(round(options.balance_transient_ms/1000/cfg.dt), ...
    max(0,cfg.N_t - 10));
remaining_steps = cfg.N_t - transient_steps;
requested_lag = round(options.balance_max_lag_ms/1000/cfg.dt);
max_lag_steps = min(requested_lag,max(0,floor((remaining_steps-1)/2)));
lags = -max_lag_steps:max_lag_steps;
balance.tight.transient_ms = transient_steps*cfg.dt*1000;
balance.tight.lag_ms = single(lags*cfg.dt*1000);
balance.tight.correlation_curves = single(lagged_correlations( ...
    double(balance.tight.V_exc(:,transient_steps+1:end)), ...
    double(balance.tight.V_inh_magnitude(:,transient_steps+1:end)), ...
    lags));
balance.tight = summarise_tight_curves(balance.tight);

high_rows = (2*n_neurons+1):(3*n_neurons);
population_curves = lagged_correlations( ...
    double(balance.loose.V_exc(high_rows,transient_steps+1:end)), ...
    double(balance.loose.V_inh_magnitude( ...
    high_rows,transient_steps+1:end)),lags);
balance.tight_population.neuron_ids = population_neurons;
balance.tight_population.source = ...
    'highest-correlation patch among loose-population candidates';
balance.tight_population.lag_ms = single(lags*cfg.dt*1000);
balance.tight_population.correlation_curves = single(population_curves);
balance.tight_population = summarise_tight_population( ...
    balance.tight_population);
save_balance_checkpoint(balance,"tight_complete",cfg,condition,options);
fprintf('  saved Tight-balance checkpoint\n');
end

function values = selected_correlations(C,indices,row_ids)
values = zeros(numel(row_ids),3);
for class_index = 1:3
    values(:,class_index) = C(sub2ind(size(C),row_ids, ...
        indices(:,class_index)));
end
end

function [stage,balance] = restore_balance_checkpoint(cfg,condition,options)
stage = "";
balance = struct();
if ~options.resume_training || isempty(options.balance_checkpoint_file) || ...
        ~isfile(options.balance_checkpoint_file)
    return;
end
try
    loaded = load(options.balance_checkpoint_file,'balance_checkpoint');
    checkpoint = loaded.balance_checkpoint;
    valid = checkpoint.schema_version == 1 && ...
        string(checkpoint.condition) == string(condition.id) && ...
        checkpoint.network_scale == cfg.network_scale && ...
        checkpoint.measurement_batch == cfg.N_n && ...
        checkpoint.balance.schema_version == 2;
    if valid
        stage = string(checkpoint.stage);
        balance = checkpoint.balance;
    end
catch exception
    warning('run_ps_factorial_condition:BadBalanceCheckpoint', ...
        'Ignoring unreadable balance checkpoint: %s',exception.message);
end
end

function save_balance_checkpoint(balance,stage,cfg,condition,options)
if isempty(options.balance_checkpoint_file)
    return;
end
directory = fileparts(options.balance_checkpoint_file);
if ~exist(directory,'dir')
    mkdir(directory);
end
balance_checkpoint.schema_version = 1;
balance_checkpoint.stage = string(stage);
balance_checkpoint.condition = string(condition.id);
balance_checkpoint.network_scale = cfg.network_scale;
balance_checkpoint.measurement_batch = cfg.N_n;
balance_checkpoint.balance = balance;
temporary = [tempname(directory) '.mat'];
save(temporary,'balance_checkpoint','-v7');
movefile(temporary,options.balance_checkpoint_file,'f');
end

function value = validate_positive_integer(value,label)
if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value) || ...
        value < 1 || value ~= floor(value)
    error('%s must be a positive integer.',label);
end
end

function P = signed_base_patches(R_X,cfg)
if cfg.network_scale == 1
    on_rates = R_X(1:cfg.N_pixels,:);
    off_rates = R_X(cfg.N_pixels+1:end,:);
else
    on_all = R_X(1:cfg.N_X,:);
    off_all = R_X(cfg.N_X+1:end,:);
    on_rates = on_all(1:cfg.network_scale:end,:);
    off_rates = off_all(1:cfg.network_scale:end,:);
end
P = on_rates - off_rates;
end

function C = normalised_patch_correlations(RF,P)
RF = double(RF);
P = double(P);
RF = RF - mean(RF,2);
P = P - mean(P,1);
rf_norm = sqrt(sum(RF.^2,2));
patch_norm = sqrt(sum(P.^2,1));
denominator = rf_norm*patch_norm;
C = (RF*P)./denominator;
C(denominator == 0) = NaN;
end

function indices = select_low_middle_high(C)
n_rows = size(C,1);
indices = zeros(n_rows,3);
low_values = C;
low_values(~isfinite(low_values)) = Inf;
[~,indices(:,1)] = min(low_values,[],2);
middle_values = abs(C);
middle_values(~isfinite(middle_values)) = Inf;
[~,indices(:,2)] = min(middle_values,[],2);
high_values = C;
high_values(~isfinite(high_values)) = -Inf;
[~,indices(:,3)] = max(high_values,[],2);
if any(~isfinite(low_values(sub2ind(size(low_values), ...
        (1:n_rows)',indices(:,1)))))
    error('A receptive field had no finite patch correlations.');
end
end

function traces = simulate_balance_shadow( ...
        state,R_X,cfg,condition,target_neurons)
% The full network spikes normally. Only the parallel voltage observers omit
% threshold, reset and V_min, preserving the trained recurrent dynamics.
n_trials = size(R_X,2);
target_neurons = target_neurons(:);
if numel(target_neurons) ~= n_trials
    error('One target E neuron is required for every balance trial.');
end

V_E = zeros(cfg.N_E,n_trials);
V_I = zeros(cfg.N_I,n_trials);
shadow_exc = zeros(n_trials,1);
shadow_inh = zeros(n_trials,1);
shadow_noise = zeros(n_trials,1);
shadow_total = zeros(n_trials,1);
traces.V_exc = zeros(n_trials,cfg.N_t,'single');
traces.V_inh_signed = zeros(n_trials,cfg.N_t,'single');
traces.V_inh_magnitude = zeros(n_trials,cfg.N_t,'single');
traces.V_noise = zeros(n_trials,cfg.N_t,'single');
traces.V_total = zeros(n_trials,cfg.N_t,'single');

W_EX_target = state.W_EX(target_neurons,:);
W_EE_target = state.W_EE(target_neurons,:);
W_EI_target = state.W_EI(target_neurons,:);
exp_m = exp(-cfg.dt/cfg.tau_m);
R_XDT = R_X*cfg.dt;
use_inhibition = condition.inhibitory_transmission_enabled;
target_linear = sub2ind([cfg.N_E n_trials], ...
    target_neurons,(1:n_trials)');

for t = 1:cfg.N_t
    S_X = rand(2*cfg.N_X,n_trials) < R_XDT;
    S_E = V_E > state.theta_E;
    S_I = V_I > state.theta_I;
    V_E(S_E) = 0;
    V_I(S_I) = 0;

    target_exc = sum(W_EX_target.*double(S_X'),2) + ...
        sum(W_EE_target.*double(S_E'),2);
    if use_inhibition
        target_inh = sum(W_EI_target.*double(S_I'),2);
    else
        target_inh = zeros(n_trials,1);
    end

    noise_E = cfg.noise_sigma*randn(cfg.N_E,n_trials);
    noise_I = cfg.noise_sigma*randn(cfg.N_I,n_trials);
    target_noise = noise_E(target_linear);

    inhibition = double(use_inhibition)*(state.W_EI*S_I);
    V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
        - inhibition + noise_E;
    V_I = V_I*exp_m + state.W_IX*S_X + state.W_IE*S_E ...
        - state.W_II*S_I + noise_I;
    V_E(V_E < cfg.V_min) = cfg.V_min;
    V_I(V_I < cfg.V_min) = cfg.V_min;

    shadow_exc = shadow_exc*exp_m + target_exc;
    shadow_inh = shadow_inh*exp_m - target_inh;
    shadow_noise = shadow_noise*exp_m + target_noise;
    shadow_total = shadow_total*exp_m + target_exc ...
        - target_inh + target_noise;
    traces.V_exc(:,t) = single(shadow_exc);
    traces.V_inh_signed(:,t) = single(shadow_inh);
    traces.V_inh_magnitude(:,t) = single(-shadow_inh);
    traces.V_noise(:,t) = single(shadow_noise);
    traces.V_total(:,t) = single(shadow_total);
end
end

function output = attach_trace_fields(output,traces,rows)
output.V_exc = traces.V_exc(rows,:);
output.V_inh_signed = traces.V_inh_signed(rows,:);
output.V_inh_magnitude = traces.V_inh_magnitude(rows,:);
output.V_noise = traces.V_noise(rows,:);
output.V_total = traces.V_total(rows,:);
end

function loose = summarise_loose_population(loose,n_neurons)
fields = {'V_exc','V_inh_signed','V_inh_magnitude','V_noise','V_total'};
for i = 1:numel(fields)
    per_trial = mean(double(loose.(fields{i})),2);
    per_neuron_class = reshape(per_trial,n_neurons,3);
    loose.(['time_mean_' fields{i}]) = single(per_neuron_class);
    loose.(['group_mean_' fields{i}]) = ...
        mean(per_neuron_class,1,'omitnan');
    loose.(['group_sd_' fields{i}]) = ...
        std(per_neuron_class,0,1,'omitnan');
end
end

function curves = lagged_correlations(E,I,lags)
n_trials = size(E,1);
curves = nan(n_trials,numel(lags));
for trial = 1:n_trials
    for lag_index = 1:numel(lags)
        lag = lags(lag_index);
        if lag >= 0
            x = E(trial,1:end-lag);
            y = I(trial,1+lag:end);
        else
            x = E(trial,1-lag:end);
            y = I(trial,1:end+lag);
        end
        x = x - mean(x);
        y = y - mean(y);
        denominator = sqrt(sum(x.^2)*sum(y.^2));
        if denominator > 0
            curves(trial,lag_index) = sum(x.*y)/denominator;
        end
    end
end
end

function tight = summarise_tight_curves(tight)
curves = double(tight.correlation_curves);
tight.mean_correlation = single(mean(curves,1,'omitnan'));
tight.sd_correlation = single(std(curves,0,1,'omitnan'));
n_valid = sum(isfinite(curves),1);
tight.n_valid = n_valid;
tight.ci95_half_width = single(1.96*double(tight.sd_correlation) ...
    ./sqrt(max(n_valid,1)));
zero_index = find(tight.lag_ms == 0,1);
tight.zero_lag_mean = tight.mean_correlation(zero_index);
tight.zero_lag_ci95 = tight.ci95_half_width(zero_index);
[tight.peak_mean_correlation,peak_index] = ...
    max(tight.mean_correlation,[],'omitnan');
if isempty(peak_index) || ~isfinite(tight.peak_mean_correlation)
    tight.peak_mean_correlation = single(NaN);
    tight.peak_lag_ms = single(NaN);
else
    tight.peak_lag_ms = tight.lag_ms(peak_index);
end
end

function population = summarise_tight_population(population)
curves = double(population.correlation_curves);
zero_index = find(population.lag_ms == 0,1);
n_trials = size(curves,1);
zero_values = curves(:,zero_index);
peak_values = nan(n_trials,1);
peak_lags = nan(n_trials,1);
for trial = 1:n_trials
    valid = find(isfinite(curves(trial,:)));
    if ~isempty(valid)
        [peak_values(trial),local_index] = max(curves(trial,valid));
        peak_lags(trial) = population.lag_ms(valid(local_index));
    end
end
population.zero_lag_values = single(zero_values);
population.peak_correlation_values = single(peak_values);
population.peak_lag_ms_values = single(peak_lags);
population.zero_lag_mean = mean(zero_values,'omitnan');
population.zero_lag_sd = std(zero_values,0,'omitnan');
population.peak_correlation_mean = mean(peak_values,'omitnan');
population.peak_correlation_sd = std(peak_values,0,'omitnan');
population.peak_lag_ms_mean = mean(peak_lags,'omitnan');
population.peak_lag_ms_sd = std(peak_lags,0,'omitnan');
end

function analysis = initialise_window_analysis(cfg, checkpoints)
analysis.batch = checkpoints(:);
analysis.window_steps = cfg.window_steps(:)';
analysis.window_ms = analysis.window_steps * cfg.dt * 1000;
analysis.repeats = cfg.window_repeats;
n_b = numel(checkpoints);
n_w = numel(analysis.window_steps);
analysis.single_trial_PS_E = nan(n_b,n_w);
analysis.trial_averaged_PS_E = nan(n_b,n_w);
analysis.mean_rate_E_Hz = nan(n_b,n_w);
analysis.var_rate_E_Hz2 = nan(n_b,n_w);
analysis.std_rate_E_Hz = nan(n_b,n_w);
analysis.mean_spike_count_E = nan(n_b,n_w);
analysis.poisson_PS_approx = nan(n_b,n_w);
analysis.per_neuron_rate_E_Hz = ...
    nan(cfg.N_E,n_b,n_w,'single');
analysis.silent_image_fraction_E = nan(n_b,n_w);
end

function analysis = measure_window_checkpoint(analysis, index, state, ...
        R_X_eval, cfg, condition)
training_rng = rng;
n_w = numel(analysis.window_steps);
sum_counts = cell(1,n_w);
trial_ps = nan(analysis.repeats,n_w);
trial_silent = nan(analysis.repeats,n_w);
for w = 1:n_w
    sum_counts{w} = zeros(cfg.N_E,size(R_X_eval,2));
end

for repeat = 1:analysis.repeats
    rng(cfg.eval_dynamics_seed + repeat, 'twister');
    counts = simulate_response_windows( ...
        state, R_X_eval, cfg, condition, analysis.window_steps);
    for w = 1:n_w
        [trial_ps(repeat,w),~,trial_silent(repeat,w)] = ...
            modified_treves_rolls(counts{w});
        sum_counts{w} = sum_counts{w} + counts{w};
    end
end

for w = 1:n_w
    averaged_counts = sum_counts{w} / analysis.repeats;
    analysis.single_trial_PS_E(index,w) = ...
        mean(trial_ps(:,w), 'omitnan');
    analysis.trial_averaged_PS_E(index,w) = ...
        modified_treves_rolls(averaged_counts);
    duration = analysis.window_steps(w) * cfg.dt;
    per_neuron_rate = mean(averaged_counts,2) / duration;
    analysis.per_neuron_rate_E_Hz(:,index,w) = ...
        single(per_neuron_rate);
    analysis.mean_rate_E_Hz(index,w) = mean(per_neuron_rate);
    analysis.var_rate_E_Hz2(index,w) = var(per_neuron_rate,0);
    analysis.std_rate_E_Hz(index,w) = std(per_neuron_rate,0);
    analysis.mean_spike_count_E(index,w) = ...
        analysis.mean_rate_E_Hz(index,w) * duration;
    analysis.poisson_PS_approx(index,w) = ...
        1 / (1 + analysis.mean_spike_count_E(index,w));
    analysis.silent_image_fraction_E(index,w) = ...
        mean(trial_silent(:,w), 'omitnan');
end
rng(training_rng);
end

function counts = simulate_response_windows(state, R_X, cfg, condition, ...
        window_steps)
n_stim = size(R_X,2);
V_E = zeros(cfg.N_E,n_stim);
V_I = zeros(cfg.N_I,n_stim);
R_E = zeros(cfg.N_E,n_stim);
exp_m = exp(-cfg.dt/cfg.tau_m);
R_XDT = R_X*cfg.dt;
counts = cell(1,numel(window_steps));

for t = 1:max(window_steps)
    S_X = rand(2*cfg.N_X,n_stim) < R_XDT;
    S_E = V_E > state.theta_E;
    S_I = V_I > state.theta_I;
    V_E(S_E) = 0;
    V_I(S_I) = 0;
    R_E = R_E + S_E;

    inhibition = double(condition.inhibitory_transmission_enabled) ...
        * (state.W_EI*S_I);
    V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
        - inhibition + cfg.noise_sigma*randn(cfg.N_E,n_stim);
    V_I = V_I*exp_m + state.W_IX*S_X + state.W_IE*S_E ...
        - state.W_II*S_I + cfg.noise_sigma*randn(cfg.N_I,n_stim);
    V_E(V_E < cfg.V_min) = cfg.V_min;
    V_I(V_I < cfg.V_min) = cfg.V_min;

    w = find(window_steps == t,1);
    if ~isempty(w)
        counts{w} = R_E;
    end
end
end

function stats = train_one_batch(state, IMAGES, cfg, condition, learn_weights)
R_X = generate_scaled_image_patches(IMAGES,cfg,cfg.N_b);
R_X(R_X > 100) = 100;
R_XDT = R_X*cfg.dt;

V_E = zeros(cfg.N_E,cfg.N_b);
V_I = zeros(cfg.N_I,cfg.N_b);
exp_m = exp(-cfg.dt/cfg.tau_m);
% 所有条件均记录逐神经元训练放电数。阈值关闭的条件也需要该数据，
% 用于绘制第 1200 个自然图像训练 batch 的 E/I 放电率散点图。
track_rates = true;
use_inhibition = condition.inhibitory_transmission_enabled;
use_inhibitory_plasticity = condition.inhibitory_plasticity_enabled;
stats = struct();

if track_rates
    R_E = zeros(cfg.N_E,cfg.N_b);
    R_I = zeros(cfg.N_I,cfg.N_b);
end

if learn_weights
    U_Xi = zeros(2*cfg.N_X,cfg.N_b);
    U_Ei = zeros(cfg.N_E,cfg.N_b);
    U_El = zeros(cfg.N_E,cfg.N_b);
    U_Ii = zeros(cfg.N_I,cfg.N_b);
    dW_EX = zeros(size(state.W_EX));
    dW_EE = zeros(size(state.W_EE));
    if use_inhibitory_plasticity
        dW_EI = zeros(size(state.W_EI));
    end
    dW_IX = zeros(size(state.W_IX));
    dW_IE = zeros(size(state.W_IE));
    dW_II = zeros(size(state.W_II));
    exp_i = exp(-cfg.dt/cfg.tau_i);
    exp_l = exp(-cfg.dt/cfg.tau_l);
    invtau_i = 1/cfg.tau_i;
    invtau_l = 1/cfg.tau_l;
    A_i = 0.5; A_p = 1; A_d = cfg.rho_E_Hz;
end

for t = 1:cfg.N_t
    S_X = rand(2*cfg.N_X,cfg.N_b) < R_XDT;
    S_E = V_E > state.theta_E;
    S_I = V_I > state.theta_I;
    V_E(S_E) = 0;
    V_I(S_I) = 0;
    if track_rates
        R_E = R_E + S_E;
        R_I = R_I + S_I;
    end

    if use_inhibition
        V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
            - state.W_EI*S_I + cfg.noise_sigma*randn(cfg.N_E,cfg.N_b);
    else
        V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
            + cfg.noise_sigma*randn(cfg.N_E,cfg.N_b);
    end
    V_I = V_I*exp_m + state.W_IX*S_X + state.W_IE*S_E ...
        - state.W_II*S_I + cfg.noise_sigma*randn(cfg.N_I,cfg.N_b);
    V_E(V_E < cfg.V_min) = cfg.V_min;
    V_I(V_I < cfg.V_min) = cfg.V_min;

    if learn_weights
        U_Xi = U_Xi*exp_i;
        U_Ei = U_Ei*exp_i;
        U_El = U_El*exp_l;
        U_Ii = U_Ii*exp_i;

        dW_EX = dW_EX + A_p*(S_E.*U_El)*U_Xi' - A_d*U_Ei*S_X';
        dW_EE = dW_EE + A_p*(S_E.*U_El)*U_Ei' ...
            - (A_d.*U_Ei)*S_E';
        if use_inhibitory_plasticity
            dW_EI = dW_EI + A_i*S_E*U_Ii' + A_i*U_Ei*S_I' ...
                + A_i*S_E*S_I';
        end
        dW_IX = dW_IX + A_i*S_I*U_Xi' + A_i*U_Ii*S_X' ...
            + A_i*S_I*S_X';
        dW_IE = dW_IE + A_i*S_I*U_Ei' + A_i*U_Ii*S_E' ...
            + A_i*S_I*S_E';
        dW_II = dW_II + A_i*U_Ii*S_I' + A_i*S_I*U_Ii' ...
            + A_i*(S_I*S_I');

        U_Xi(S_X) = U_Xi(S_X) + invtau_i;
        U_Ei(S_E) = U_Ei(S_E) + invtau_i;
        U_El(S_E) = U_El(S_E) + invtau_l;
        U_Ii(S_I) = U_Ii(S_I) + invtau_i;
    end
end

if track_rates
    stats.R_E = mean(R_E,2);
    stats.R_I = mean(R_I,2);
end
if learn_weights
    scale = state.learning_scale/cfg.N_b;
    stats.dW_EX = dW_EX*scale*cfg.eta_EX;
    stats.dW_EE = dW_EE*scale*cfg.eta_EE;
    if use_inhibitory_plasticity
        stats.dW_EI = dW_EI*scale*cfg.eta_EI;
    end
    stats.dW_IX = dW_IX*scale*cfg.eta_IX;
    stats.dW_IE = dW_IE*scale*cfg.eta_IE;
    stats.dW_II = dW_II*scale*cfg.eta_II;
end
end

function state = apply_weight_plasticity(state, stats, cfg, condition)
use_l1 = condition.l1_normalization_enabled;
state.W_EX = update_group(state.W_EX,stats.dW_EX,cfg.L1_EX,true,use_l1);
state.W_EE = update_group(state.W_EE,stats.dW_EE,cfg.L1_EE,false,use_l1);
if condition.inhibitory_plasticity_enabled
    state.W_EI = update_group( ...
        state.W_EI,stats.dW_EI,cfg.L1_EI,true,use_l1);
end
state.W_IX = update_group(state.W_IX,stats.dW_IX,cfg.L1_IX,true,use_l1);
state.W_IE = update_group(state.W_IE,stats.dW_IE,cfg.L1_IE,true,use_l1);
state.W_II = update_group(state.W_II,stats.dW_II,cfg.L1_II,false,use_l1);
end

function W = update_group(W, dW, upper_bound, allow_self, use_l1)
W = W + dW;
if use_l1
    W = normalize_group(W,upper_bound,allow_self);
else
    % Removing L1 normalization must not violate the sign convention or
    % introduce recurrent self-connections.
    W(W < 0) = 0;
    if ~allow_self && size(W,1) == size(W,2)
        W(1:size(W,1)+1:end) = 0;
    end
end
if any(~isfinite(W(:)))
    error('A synaptic weight became NaN or Inf.');
end
end

function state = apply_threshold_plasticity(state, stats, cfg)
scale = state.learning_scale;
dtheta_E = (stats.R_E - cfg.rho_E)*cfg.eta_tE*scale;
dtheta_I = (stats.R_I - cfg.rho_I)*cfg.eta_tI*scale;
dtheta_E = min(dtheta_E,cfg.rho_E*cfg.eta_tE*scale);
dtheta_I = min(dtheta_I,cfg.rho_I*cfg.eta_tI*scale);
state.theta_E = state.theta_E + dtheta_E;
state.theta_I = state.theta_I + dtheta_I;
% 每次阈值更新后立即施加生理下限：E、I 阈值均不得小于 0。
state.theta_E(state.theta_E < 0) = 0;
state.theta_I(state.theta_I < 0) = 0;
end

function W = normalize_group(W,upper_bound,allow_self)
W(W < 0) = 0;
if ~allow_self && size(W,1) == size(W,2)
    W(1:size(W,1)+1:end) = 0;
end
row_sum = sum(W,2);
W = W + (-row_sum + min(row_sum,upper_bound))/size(W,2);
W(W < 0) = 0;
row_sum = sum(W,2);
nonzero = row_sum > 0;
W(nonzero,:) = W(nonzero,:)./row_sum(nonzero) ...
    .* min(row_sum(nonzero),upper_bound);
end

function [R_X,meta] = make_grating_inputs(cfg)
n_side = round(sqrt(cfg.N_pixels));
coords = ((1:n_side) - (n_side+1)/2)/n_side;
[x,y] = meshgrid(coords,coords);
n_ori = numel(cfg.grating_orientations_deg);
n_sf = numel(cfg.grating_spatial_frequencies);
n_phase = numel(cfg.grating_phases_deg);
n_stim = n_ori*n_sf*n_phase;
R_X_base = zeros(2*cfg.N_pixels,n_stim);
meta.orientation_index = zeros(n_stim,1);
meta.spatial_frequency_index = zeros(n_stim,1);
meta.phase_index = zeros(n_stim,1);

index = 0;
for oi = 1:n_ori
    theta = deg2rad(cfg.grating_orientations_deg(oi));
    projection = x*cos(theta) + y*sin(theta);
    for si = 1:n_sf
        sf = cfg.grating_spatial_frequencies(si);
        for pi = 1:n_phase
            index = index + 1;
            phase = deg2rad(cfg.grating_phases_deg(pi));
            grating = cos(2*pi*sf*projection + phase);
            grating = grating - mean(grating(:));
            grating = grating/max(std(grating(:),1),eps);
            input_rate = 70*grating(:);
            R_X_base(1:cfg.N_pixels,index) = max(input_rate,0);
            R_X_base(cfg.N_pixels+1:end,index) = max(-input_rate,0);
            meta.orientation_index(index) = oi;
            meta.spatial_frequency_index(index) = si;
            meta.phase_index(index) = pi;
        end
    end
end
R_X = expand_input_population(R_X_base,cfg);
R_X(R_X > 100) = 100;
end

function R_X = generate_scaled_image_patches(IMAGES,cfg,n_patches)
R_X_base = generate_image_patches(IMAGES,cfg.N_pixels,n_patches);
R_X = expand_input_population(R_X_base,cfg);
end

function R_X = expand_input_population(R_X_base,cfg)
if cfg.network_scale == 1
    R_X = R_X_base;
    return;
end
on_rates = repelem(R_X_base(1:cfg.N_pixels,:), ...
    cfg.network_scale,1);
off_rates = repelem(R_X_base(cfg.N_pixels+1:end,:), ...
    cfg.network_scale,1);
R_X = [on_rates;off_rates];
end

function [OSI,tuning,preferred_sf_value,preferred_orientation] = ...
        calculate_osi(grating_rates,meta,cfg)
n_ori = numel(cfg.grating_orientations_deg);
n_sf = numel(cfg.grating_spatial_frequencies);
n_E = size(grating_rates,1);
orientation_response = zeros(n_E,n_ori,n_sf);

for oi = 1:n_ori
    for si = 1:n_sf
        select = meta.orientation_index == oi & ...
            meta.spatial_frequency_index == si;
        orientation_response(:,oi,si) = mean(grating_rates(:,select),2);
    end
end

sf_score = squeeze(mean(orientation_response,2));
if n_E == 1
    sf_score = reshape(sf_score,1,[]);
end
[~,preferred_sf] = max(sf_score,[],2);
tuning = zeros(n_E,n_ori);
for e = 1:n_E
    tuning(e,:) = reshape( ...
        orientation_response(e,:,preferred_sf(e)),1,[]);
end

angles = deg2rad(cfg.grating_orientations_deg(:)');
vector = sum(tuning.*exp(2i*angles),2);
denominator = sum(tuning,2);
OSI = abs(vector)./denominator;
OSI(denominator <= eps) = NaN;
OSI = min(max(real(OSI),0),1);
[~,preferred_orientation_index] = max(tuning,[],2);
preferred_orientation = ...
    cfg.grating_orientations_deg(preferred_orientation_index)';
preferred_sf_value = cfg.grating_spatial_frequencies(preferred_sf)';
invalid = ~isfinite(OSI);
preferred_orientation(invalid) = NaN;
preferred_sf_value(invalid) = NaN;
end

function print_checkpoint(result,index)
if result.options.measure_extended_metrics
    fprintf(['  batch %4d | PS_E %.4f | LS_E %.4f | E %.3f Hz | ' ...
        'I %.3f Hz | RF corr %.4f | mean OSI %.4f | ' ...
        'theta E/I %.3f/%.3f\n'],result.batch(index), ...
        result.PS_E(index),result.lifetime_sparseness_E(index), ...
        result.mean_rate_E_Hz(index),result.mean_rate_I_Hz(index), ...
        result.RF_correlation_RMS_E(index),result.mean_OSI_E(index), ...
        result.mean_theta_E(index),result.mean_theta_I(index));
else
    fprintf('  batch %4d | PS_E %.4f | LS_E %.4f | RF corr %.4f\n', ...
        result.batch(index),result.PS_E(index), ...
        result.lifetime_sparseness_E(index), ...
        result.RF_correlation_RMS_E(index));
end
end

function print_progress(result,index,options)
if options.measure_standard_metrics
    print_checkpoint(result,index);
elseif options.run_window_experiment
    fprintf('  batch %4d | window metrics recorded\n', ...
        result.batch(index));
else
    fprintf('  batch %4d | training checkpoint; evaluation skipped\n', ...
        result.batch(index));
end
end
