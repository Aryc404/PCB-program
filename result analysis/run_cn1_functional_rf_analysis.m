% 作用：读取cn=1训练检查点，并计算感受野、调谐、稀疏性、权重和E/I平衡指标并作图。
function summary = run_cn1_functional_rf_analysis(options)
if nargin < 1
    options = struct();
end
% fill_defaults：补齐用户未提供的分析参数。
options = fill_defaults(options);
if options.self_test
    % self_test：快速检查Gabor拟合和PS,LS各类公式能否正常运行
    self_test();
    summary = table();
    return;
end

analysis_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(analysis_dir);
if strlength(string(options.training_results_dir)) == 0
    training_results_dir = fullfile(repo_dir,'shuffle_study', ...
        'pcb_shuffle_study','results');
else
    training_results_dir = char(options.training_results_dir);
end
checkpoint_dir = fullfile(training_results_dir, ...
    'ei_balance_cn1','cn_1','checkpoints');
organized_checkpoint_dir = fullfile(training_results_dir, ...
    '01_实验数据','cn1_EI平衡','cn_1','checkpoints');
if isfolder(organized_checkpoint_dir)
    checkpoint_dir = organized_checkpoint_dir;
end
if ~isfolder(checkpoint_dir)
    error(['Cannot find training checkpoints: %s\nRun the cn=1 training ' ...
        'program first, or set options.training_results_dir.'],checkpoint_dir);
end
if strlength(string(options.output_root)) == 0
    output_root = fullfile(analysis_dir,'result');
else
    output_root = char(options.output_root);
end

ensure_dir(output_root);
pool_ready = false;

conditions = upper(string(options.conditions));
conditions = conditions(:);
valid_conditions = ["AD","AE","AF","AG", ...
    "BD","BE","BF","BG","CD","CE","CF","CG"];
if isempty(conditions)
    error('At least one PCB condition must be selected.');
end
invalid_conditions = unique(conditions(~ismember(conditions,valid_conditions)));
if ~isempty(invalid_conditions)
    error('Unknown PCB condition(s): %s. Valid conditions are: %s.', ...
        strjoin(cellstr(invalid_conditions),', '), ...
        strjoin(cellstr(valid_conditions),', '));
end
% 删除重复条件，避免同一条件被重复测试。
conditions = unique(conditions,'stable');
batches = options.analysis_batches(:)';
rows = numel(conditions)*numel(batches);
condition_column = strings(rows,1);
batch_column = zeros(rows,1);
status_column = strings(rows,1);
output_column = strings(rows,1);
row = 0;

for condition_index = 1:numel(conditions)
    % ===== 逐个分析A/B/C抑制机制 × D/E/F/G稳态机制的十二种组合 =====
    condition_id = conditions(condition_index);
    condition_dir = fullfile(output_root,char(condition_id));
    csv_dir = fullfile(condition_dir,'csv');
    mat_dir = fullfile(condition_dir,'mat');
    png_dir = fullfile(condition_dir,'png');
    % ensure_dir：建立当前条件的CSV、MAT和PNG输出目录。
    ensure_dir(csv_dir); ensure_dir(mat_dir); ensure_dir(png_dir);

    for batch = batches
        row = row + 1;
        condition_column(row) = condition_id;
        batch_column(row) = batch;
        output_file = fullfile(mat_dir,sprintf( ...
            '%s_functional_rf_batch%04d.mat',lower(condition_id),batch));
        output_column(row) = string(output_file);
        checkpoint_file = fullfile(checkpoint_dir,sprintf( ...
            'pcb_shuffle_%s_paper_cn1_run01_batch%04d.mat', ...
            lower(condition_id),batch));
    
        if ~pool_ready
            % ensure_parallel_pool：开启Processes并行池。
            ensure_parallel_pool(options.max_workers);
            pool_ready = true;
        end

        if ~isfile(checkpoint_file)
            error('Missing checkpoint: %s',checkpoint_file);
        end
        loaded = load(checkpoint_file,'checkpoint');
        state = loaded.checkpoint.state;
        % evaluation_config：读取网络规模，并补齐后续白噪声图片测试时所需的原模型参数
        cfg = evaluation_config(state,options);
        use_inhibition = extractBefore(condition_id,2) ~= "C";
        fprintf('\n%s batch %d: measuring functional RF...\n', ...
            condition_id,batch);

        % 【感受野】冻结学习，用白噪声图片训练，并利用STA估计每个E神经元的16×16感受野。
        [functional_rf,white_counts,white_stimuli] = measure_sta_rf( ...
            state,cfg,use_inhibition,options.white_noise_count, ...
            options.white_noise_rate_hz,options.stimulus_block_size, ...
            options.white_noise_seed + batch);
        fprintf('%s batch %d: measuring grating tuning...\n', ...
            condition_id,batch);
        % 【调谐测试】测量不同朝向、空间频率和相位光栅下的平均放电率。
        [grating_rates,grating_meta] = measure_grating_responses( ...
            state,cfg,use_inhibition,options);
        % 【神经元指标】由光栅响应计算1-CirVar、偏好朝向/频率及两类带宽。
        tuning = calculate_tuning_metrics(grating_rates,grating_meta);
        fprintf('%s batch %d: fitting Gabor functions...\n', ...
            condition_id,batch);
        % 【Gabor指标】对每个感受野拟合二维Gabor函数并记录参数和误差。
        gabor = fit_gabor_population(functional_rf);

        % 【放电率】白噪声测试中的脉冲数除以单次刺激持续时间。
        white_duration = cfg.n_steps*cfg.dt;
        white_rates = double(white_counts)/white_duration;
        % 【LS】
        lifetime = corrected_sparseness(white_rates,2);
        % 【PS】
        population = corrected_sparseness(white_rates,1);
        % 【感受野相关性】所有有效E神经元感受野两两Pearson相关的RMS。
        rf_corr_rms = pairwise_correlation_rms(functional_rf);
        % 【响应相关性】所有E神经元白噪声响应两两相关的RMS。
        response_corr_rms = pairwise_correlation_rms(white_rates);
        % 读取训练检查点中已经测好的loose/tight E-I balance结果。
        balance = load_balance_metrics(checkpoint_dir,condition_id,batch);

        % make_neuron_table：把每个E神经元的指标合并为逐神经元表格。
        neuron_table = make_neuron_table(white_rates,lifetime,tuning,gabor);
        % make_network_table：把当前分析得到的群体指标合并为网络级表格。
        network_table = make_network_table(condition_id,batch,population, ...
            lifetime,rf_corr_rms,response_corr_rms,tuning,gabor,balance);
        % 【权重指标】统计六类突触的均值、中位数、标准差、零权重数和直方图。
        [weight_summary,weight_histograms,weights] = ...
            analyse_weights(state,condition_id,batch,options.weight_bins);

        writetable(neuron_table,fullfile(csv_dir,sprintf( ...
            '%s_neuron_metrics_batch%04d.csv',lower(condition_id),batch)));
        writetable(network_table,fullfile(csv_dir,sprintf( ...
            '%s_network_metrics_batch%04d.csv',lower(condition_id),batch)));
        writetable(weight_summary,fullfile(csv_dir,sprintf( ...
            '%s_weight_summary_batch%04d.csv',lower(condition_id),batch)));
        writetable(weight_histograms,fullfile(csv_dir,sprintf( ...
            '%s_weight_histograms_batch%04d.csv',lower(condition_id),batch)));
        % write_rf_csv：将每个E神经元的感受野像素逐行保存为CSV文件。
        write_rf_csv(functional_rf,fullfile(csv_dir,sprintf( ...
            '%s_functional_rf_batch%04d.csv',lower(condition_id),batch)));

        analysis = struct();
        analysis.condition = char(condition_id);
        analysis.batch = batch;
        analysis.cfg = cfg;
        analysis.functional_rf = single(functional_rf);
        analysis.white_noise_response_counts = uint16(white_counts);
        analysis.white_noise_stimuli = single(white_stimuli);
        analysis.grating_rates = single(grating_rates);
        analysis.grating_meta = grating_meta;
        analysis.tuning = tuning;
        analysis.neuron_metrics = neuron_table;
        analysis.network_metrics = network_table;
        analysis.weights = weights;
        analysis.weight_summary = weight_summary;
        analysis.weight_histograms = weight_histograms;
        save(output_file,'analysis','-v7.3');

        % 输出感受野、神经元指标、相关性和权重分布图片。
        plot_analysis_figures(analysis,condition_id,batch,png_dir, ...
            options.weight_bins);
        % 输出 loose balance 和 tight balance 图片。
        plot_ei_balance_figures(checkpoint_dir,condition_id,png_dir);
        status_column(row) = "complete";
    end
end

summary = table(condition_column,batch_column,status_column,output_column, ...
    'VariableNames',{'condition','batch','status','mat_file'});
writetable(summary,fullfile(output_root,'analysis_run_summary.csv'));
fprintf('\nAnalysis outputs: %s\n',output_root);
end

% 作用：把没有设置的参数补成默认值。

function options = fill_defaults(options)
defaults.conditions = ["AD","AE","AF","AG", ...
    "BD","BE","BF","BG","CD","CE","CF","CG"];
defaults.analysis_batches = 1200;
defaults.white_noise_count = 100000;
defaults.white_noise_rate_hz = 70;
defaults.white_noise_seed = 4202603;
defaults.grating_seed = 5202603;
defaults.grating_repeats = 10;
defaults.grating_orientations_deg = 0:11.25:168.75;
defaults.grating_spatial_frequencies_cycles_per_pixel = 0.05:0.05:0.40;
defaults.grating_phases_deg = [0 90 180 270];
defaults.stimulus_block_size = 200;
defaults.max_workers = 6;
defaults.n_steps = 400;
defaults.weight_bins = 50;
defaults.self_test = false;
defaults.training_results_dir = "";
defaults.output_root = "";
names = fieldnames(defaults);
for index = 1:numel(names)
    name = names{index};
    if ~isfield(options,name)
        options.(name) = defaults.(name);
    end
end
end

% 作用：读取网络规模，并补齐后续白噪声图片测试时所需的原模型参数
function cfg = evaluation_config(state,options)
cfg.dt = 1e-3;
cfg.tau_m = 10e-3;
cfg.V_min = -10;
cfg.noise_sigma = 1;
cfg.n_steps = options.n_steps;
cfg.N_E = size(state.W_EX,1);
cfg.N_I = size(state.W_IX,1);
cfg.N_pixels = size(state.W_EX,2)/2;
if cfg.N_pixels ~= 256
    error('Expected a cn=1 checkpoint with 256 input pixels.');
end
end

% 作用：用高斯白噪声刺激训练完1200 batch 后的网络，用 STA 算出每个E神经元的感受野。
function [rf,counts,stimuli] = measure_sta_rf(state,cfg,use_inhibition, ...
        n_stimuli,rate_hz,block_size,seed)
rng(seed,'twister');
stimuli = randn(cfg.N_pixels,n_stimuli);
stimuli = stimuli-mean(stimuli,2);
input_rates = [max(rate_hz*stimuli,0);max(-rate_hz*stimuli,0)];
input_rates(input_rates > 100) = 100;
% simulate_in_blocks：把大量白噪声刺激分块后并行运行冻结网络（学习率设置为0）。
counts = simulate_in_blocks(state,input_rates,cfg,use_inhibition, ...
    block_size,seed+1);
spike_sum = sum(double(counts),2);
rf = double(counts)*stimuli';
active = spike_sum > 0;
rf(active,:) = rf(active,:)./spike_sum(active);
rf(~active,:) = NaN;
end

% 作用：对冻结后的网络进行光栅刺激和空白刺激，并计算每个E神经元的平均放电率。
function [rates,meta] = measure_grating_responses(state,cfg,use_inhibition,options)
% make_grating_inputs：生成光栅并建立每列刺激的参数标签。
[input_rates,meta] = make_grating_inputs(cfg,options);
duration = cfg.n_steps*cfg.dt;
counts = zeros(cfg.N_E,size(input_rates,2));
for repeat = 1:options.grating_repeats
    % simulate_in_blocks：将当前一轮光栅刺激分块并行运行。
    repeat_counts = simulate_in_blocks(state,input_rates,cfg, ...
        use_inhibition,options.stimulus_block_size,options.grating_seed+repeat);
    counts = counts+double(repeat_counts);
end
rates = counts/(options.grating_repeats*duration);
end

% 作用：把大批刺激分成小块并行测试
function counts = simulate_in_blocks(state,input_rates,cfg,use_inhibition, ...
        block_size,seed)
n_stimuli = size(input_rates,2);
n_blocks = ceil(n_stimuli/block_size);
block_widths = repmat(block_size,1,n_blocks);
block_widths(end) = n_stimuli-block_size*(n_blocks-1);
rate_blocks = mat2cell(input_rates,size(input_rates,1),block_widths);
count_blocks = cell(1,n_blocks);
parfor block = 1:n_blocks
    rng(seed+block,'twister');
    % simulate_response：对分块后的白噪声刺激进行LIF网络的模拟运行 并返回E神经元在该刺激下的脉冲数。
    count_blocks{block} = uint16(simulate_response(state, ...
        rate_blocks{block},cfg,use_inhibition));
end
counts = cat(2,count_blocks{:});
assert(isequal(size(counts),[cfg.N_E n_stimuli]), ...
    'Parallel response blocks were assembled incorrectly.');
end

% 作用：对分块后的白噪声刺激进行LIF网络的模拟运行 并返回E神经元在该刺激下的脉冲数。
function R_E = simulate_response(state,R_X,cfg,use_inhibition)
n_stimuli = size(R_X,2);
V_E = zeros(cfg.N_E,n_stimuli);
V_I = zeros(cfg.N_I,n_stimuli);
R_E = zeros(cfg.N_E,n_stimuli);
exp_m = exp(-cfg.dt/cfg.tau_m);
R_XDT = R_X*cfg.dt;
for t = 1:cfg.n_steps
    S_X = rand(2*cfg.N_pixels,n_stimuli) < R_XDT;
    S_E = V_E > state.theta_E;
    S_I = V_I > state.theta_I;
    V_E(S_E) = 0;
    V_I(S_I) = 0;
    R_E = R_E+S_E;
    inhibition = double(use_inhibition)*(state.W_EI*S_I);
    V_E = V_E*exp_m+state.W_EX*S_X+state.W_EE*S_E ...
        -inhibition+cfg.noise_sigma*randn(cfg.N_E,n_stimuli);
    V_I = V_I*exp_m+state.W_IX*S_X+state.W_IE*S_E ...
        -state.W_II*S_I+cfg.noise_sigma*randn(cfg.N_I,n_stimuli);
    V_E(V_E < cfg.V_min) = cfg.V_min;
    V_I(V_I < cfg.V_min) = cfg.V_min;
end
end

% 作用：生成所有“朝向×空间频率×相位”的光栅(包含空白刺激)

function [R_X,meta] = make_grating_inputs(cfg,options)
n_side = sqrt(cfg.N_pixels);
coords = ((1:n_side)-(n_side+1)/2)/n_side;
[x,y] = meshgrid(coords,coords);
ori = options.grating_orientations_deg;
sf_cpp = options.grating_spatial_frequencies_cycles_per_pixel;
sf_cpi = sf_cpp*n_side;
phase = options.grating_phases_deg;
n_gratings = numel(ori)*numel(sf_cpp)*numel(phase);
n_stimuli = n_gratings+1;
R_X = zeros(2*cfg.N_pixels,n_stimuli);
meta.orientation_deg = nan(n_stimuli,1);
meta.spatial_frequency_cycles_per_pixel = nan(n_stimuli,1);
meta.spatial_frequency_cycles_per_image = nan(n_stimuli,1);
meta.phase_deg = nan(n_stimuli,1);
meta.is_blank = false(n_stimuli,1);
index = 0;
for oi = 1:numel(ori)
    theta = deg2rad(ori(oi));
    projection = x*cos(theta)+y*sin(theta);
    for si = 1:numel(sf_cpp)
        for pi = 1:numel(phase)
            index = index+1;
            grating = cos(2*pi*sf_cpi(si)*projection+deg2rad(phase(pi)));
            grating = grating-mean(grating,'all');
            grating = grating/max(std(grating,1,'all'),eps);
            input = 70*grating(:);
            R_X(:,index) = [max(input,0);max(-input,0)];
            meta.orientation_deg(index) = ori(oi);
            meta.spatial_frequency_cycles_per_pixel(index) = sf_cpp(si);
            meta.spatial_frequency_cycles_per_image(index) = sf_cpi(si);
            meta.phase_deg(index) = phase(pi);
        end
    end
end
meta.is_blank(end) = true;
end

% 作用：从光栅放电率中计算每个E神经元的1-CirVar、偏好朝向、偏好SF和朝向及SF带宽。
% 输入：rates是各刺激下的平均放电率，meta记录每列对应的光栅参数。
% 输出：tuning，每个E神经元的调谐指标和拟合质量。
function tuning = calculate_tuning_metrics(rates,meta)
if ~isfield(meta,'is_blank') || ~any(meta.is_blank)
    error('The current grating protocol requires an explicit blank stimulus.');
end
grating_columns = ~meta.is_blank;
orientations = unique(meta.orientation_deg(grating_columns),'stable')';
frequencies_cpp = unique(meta.spatial_frequency_cycles_per_pixel(grating_columns),'stable')';
phases = unique(meta.phase_deg(grating_columns),'stable')';
n_E = size(rates,1);

% 通过空白刺激计算出每个神经元的自发放电率；将神经元面对光栅刺激的放电率减去面对空白刺激的放电率
% 并把负值截为0(保证1-CirVar使用非负响应)，并以此计算后续指标。
blank_rate = mean(double(rates(:,meta.is_blank)),2);
response = assemble_grating_grid(rates,meta,blank_rate,orientations, ...
    frequencies_cpp,phases);
% 先计算跨相位平均的“朝向×空间频率”二维响应分布，以便计算后续的偏好朝向与SF频率。
response_surface = mean(response,4);

% 在跨相位的二维响应分布中寻找最大值，同时提取一维调谐曲线(相当于固定朝向/SF频率，
% 并在此朝向/SF频率下，拟合放电率相对于所有SF频率/朝向的变化曲线)。
[~,linear_surface_index] = max(reshape(response_surface,n_E,[]),[],2);
[preferred_orientation_index,preferred_sf_index] = ind2sub( ...
    [numel(orientations),numel(frequencies_cpp)],linear_surface_index);

orientation_tuning = zeros(n_E,numel(orientations));
sf_tuning = zeros(n_E,numel(frequencies_cpp));
for neuron = 1:n_E
    orientation_tuning(neuron,:) = reshape(response_surface(neuron,:, ...
        preferred_sf_index(neuron)),1,[]);
    sf_tuning(neuron,:) = reshape(response_surface(neuron, ...
        preferred_orientation_index(neuron),:),1,[]);
end

% 1-CirVar及偏好朝向：朝向具有180度周期，因此使用二倍角复向量。
orientation_angles = deg2rad(orientations);
orientation_vector = sum(orientation_tuning.*exp(2i*orientation_angles),2);
orientation_sum = sum(orientation_tuning,2);
one_minus_CirVar = abs(orientation_vector)./orientation_sum;
one_minus_CirVar(orientation_sum <= eps) = NaN;
preferred_orientation = mod(rad2deg(angle(orientation_vector))/2,180);
preferred_orientation(orientation_sum <= eps) = NaN;

orientation_bandwidth = nan(n_E,1);
orientation_fit_error = nan(n_E,1);
% 与朝向带宽保持一致：SF拟合无效时不列入统计
preferred_sf_cpp = nan(n_E,1);
sf_bandwidth = nan(n_E,1);
sf_fit_error = nan(n_E,1);
sf_tuning_type = strings(n_E,1);
for neuron = 1:n_E
    [fit_bandwidth,fit_error,fit_ok] = fit_von_mises_orientation_tuning( ...
        orientations,orientation_tuning(neuron,:));
    orientation_fit_error(neuron) = fit_error;
    if fit_ok
        orientation_bandwidth(neuron) = fit_bandwidth;
    end

    [fit_sf,fit_bw,fit_error,sf_type,preference_ok,bandwidth_ok] = ...
        fit_log_gaussian_sf(frequencies_cpp,sf_tuning(neuron,:));
    sf_fit_error(neuron) = fit_error;
    sf_tuning_type(neuron) = sf_type;
    if preference_ok
        preferred_sf_cpp(neuron) = fit_sf;
    end
    if bandwidth_ok
        sf_bandwidth(neuron) = fit_bw;
    end
end
preferred_sf_cpi = preferred_sf_cpp*16;

tuning.blank_rate_Hz = blank_rate;
tuning.one_minus_CirVar = one_minus_CirVar;
tuning.preferred_orientation_deg = preferred_orientation;
tuning.orientation_bandwidth_deg = orientation_bandwidth;
tuning.orientation_fit_error = orientation_fit_error;
tuning.preferred_spatial_frequency_cycles_per_pixel = preferred_sf_cpp;
tuning.preferred_spatial_frequency_cycles_per_image = preferred_sf_cpi;
tuning.spatial_frequency_bandwidth_octaves = sf_bandwidth;
tuning.spatial_frequency_fit_error = sf_fit_error;
tuning.spatial_frequency_tuning_type = sf_tuning_type;
tuning.orientation_tuning = orientation_tuning;
tuning.spatial_frequency_tuning = sf_tuning;
tuning.orientations_deg = orientations;
tuning.spatial_frequencies_cycles_per_pixel = frequencies_cpp;
tuning.spatial_frequencies_cycles_per_image = frequencies_cpp*16;
tuning.response_grid = response;
tuning.response_surface = response_surface;
end

% 作用：将神经元面对光栅刺激时的放电率减去面对空白响应时的放电率，再把光栅响应做成“神经元×朝向×SF×相位”的矩阵。
function response = assemble_grating_grid(rates,meta,baseline,orientations,frequencies,phases)
n_E = size(rates,1);
response = zeros(n_E,numel(orientations),numel(frequencies),numel(phases));
for oi = 1:numel(orientations)
    for si = 1:numel(frequencies)
        for pi = 1:numel(phases)
            select = ~meta.is_blank & meta.orientation_deg == orientations(oi) & ...
                meta.spatial_frequency_cycles_per_pixel == frequencies(si) & ...
                meta.phase_deg == phases(pi);
            response(:,oi,si,pi) = max(mean(double(rates(:,select)),2)-baseline,0);
        end
    end
end
end

% 作用：用180度周期的von Mises曲线拟合朝向响应，并算半高带宽。
function [bandwidth,fit_error,fit_ok] = fit_von_mises_orientation_tuning(x,y)
x = double(x(:)'); y = double(y(:)');
bandwidth = NaN; fit_error = NaN; fit_ok = false;
if any(~isfinite(y)) || max(y)-min(y) <= eps, return; end
[~,peak_index] = max(y);
baseline0 = max(min(y),1e-6);
amplitude0 = max(max(y)-min(y),1e-6);
p0 = [log(baseline0) log(amplitude0) x(peak_index) log(1.5)];
objective = @(p)sum((y-von_mises_orientation_model(p,x)).^2);
p = fminsearch(objective,p0,optimset('Display','off','MaxIter',1000,'MaxFunEvals',3000));
fit_y = von_mises_orientation_model(p,x);
fit_error = sum((y-fit_y).^2)/max(sum((y-mean(y)).^2),eps);
kappa = min(max(exp(p(4)),eps),1e4);
half_argument = 1-log(2)/kappa;
if half_argument <= -1
    bandwidth = 90;
elseif half_argument < 1
    bandwidth = rad2deg(0.5*acos(half_argument));
end
fit_ok = isfinite(fit_error) && fit_error <= 0.30 && ...
    isfinite(bandwidth) && bandwidth > 0 && bandwidth <= 90;
if ~fit_ok, bandwidth = NaN; end
end

% 作用：生成拟合von Mises函数的朝向调谐曲线。
function y = von_mises_orientation_model(p,x)
baseline = exp(p(1)); amplitude = exp(p(2));
center = mod(p(3),180);
kappa = min(max(exp(p(4)),eps),1e4);
y = baseline+amplitude*exp(kappa*(cos(2*deg2rad(x-center))-1));
end

% 作用：在log2(SF)轴上拟合高斯曲线，计算偏好SF和SF带宽。
function [preferred_sf,bandwidth,fit_error,tuning_type,preference_ok,bandwidth_ok] = fit_log_gaussian_sf(frequencies,response)
frequencies = double(frequencies(:)'); response = double(response(:)');
preferred_sf = NaN; bandwidth = NaN; fit_error = NaN;
tuning_type = "unreliable"; preference_ok = false; bandwidth_ok = false;
if any(~isfinite(response)) || max(response)-min(response) <= eps, return; end
x = log2(frequencies);
[~,peak_index] = max(response);
p0 = [log(max(min(response),1e-6)) log(max(max(response)-min(response),1e-6)) ...
    x(peak_index) log(0.6)];
objective = @(p)sum((response-log_gaussian_model(p,x)).^2);
p = fminsearch(objective,p0,optimset('Display','off','MaxIter',1000,'MaxFunEvals',3000));
fit_y = log_gaussian_model(p,x);
fit_error = sum((response-fit_y).^2)/max(sum((response-mean(response)).^2),eps);
mu = p(3); sigma = exp(p(4));
half_level = min(fit_y)+(max(fit_y)-min(fit_y))/2;
if response(1) >= half_level
    tuning_type = "lowpass";
elseif response(end) >= half_level
    tuning_type = "highpass-or-unresolved";
else
    tuning_type = "bandpass";
end
hwhm = sigma*sqrt(2*log(2));
crossings_inside = (mu-hwhm > min(x)) && (mu+hwhm < max(x));
preference_ok = tuning_type == "bandpass" && fit_error <= 0.30 && ...
    mu > min(x) && mu < max(x) && isfinite(sigma);
bandwidth_ok = preference_ok && crossings_inside;
if preference_ok
    preferred_sf = 2^mu;
end
if bandwidth_ok
    bandwidth = 2*hwhm;
elseif fit_error > 0.30
    tuning_type = "unreliable";
end
end

% 作用：拟合生成对数高斯SF调谐曲线。
function y = log_gaussian_model(p,x)
baseline = exp(p(1)); amplitude = exp(p(2));
center = p(3); sigma = max(exp(p(4)),0.02);
y = baseline+amplitude*exp(-0.5*((x-center)/sigma).^2);
end

% 作用：逐个拟合E神经元的感受野，并整理Gabor参数和拟合误差。
function gabor = fit_gabor_population(rf)
n_E = size(rf,1);
fields = {'fit_error','orientation_deg','spatial_frequency', ...
    'phase_deg','sigma_x','sigma_y','center_x','center_y','n_x','n_y'};
for index = 1:numel(fields)
    gabor.(fields{index}) = nan(n_E,1);
end
for neuron = 1:n_E
    % fit_one_gabor：对当前神经元的二维感受野完成一次Gabor参数拟合。
    fit = fit_one_gabor(rf(neuron,:));
    for index = 1:numel(fields)
        gabor.(fields{index})(neuron) = fit.(fields{index});
    end
end
gabor.well_fit = gabor.fit_error < 0.30;
gabor = filter_invalid_gabor_parameters(gabor);
end

% 作用：记录拟合误差的分布情况，同时把不合格Gabor拟合(误差率＞0.3)的朝向和相位设为NaN。
function gabor = filter_invalid_gabor_parameters(gabor)
if ~isfield(gabor,'well_fit')
    gabor.well_fit = gabor.fit_error < 0.30;
end
invalid_fit = ~logical(gabor.well_fit);
gabor.orientation_deg(invalid_fit) = NaN;
gabor.phase_deg(invalid_fit) = NaN;
end

% 作用：将E神经元的感受野拟合为二维Gabor模型。

function fit = fit_one_gabor(rf_vector)
names = {'fit_error','orientation_deg','spatial_frequency','phase_deg', ...
    'sigma_x','sigma_y','center_x','center_y','n_x','n_y'};
for index = 1:numel(names), fit.(names{index}) = NaN; end
n = round(sqrt(numel(rf_vector)));
z = reshape(double(rf_vector),n,n);
if any(~isfinite(z),'all') || std(z,1,'all') <= eps
    return;
end
[x,y] = meshgrid(1:n,1:n);
z0 = z-mean(z,'all');
energy = z0.^2;
total = sum(energy,'all');
x0 = sum(x.*energy,'all')/total;
y0 = sum(y.*energy,'all')/total;
F = abs(fftshift(fft2(z0)));
center = floor(n/2)+1;
F(center,center) = 0;
[~,peak] = max(F,[],'all');
[fy_index,fx_index] = ind2sub([n n],peak);
fx = (fx_index-center)/n;
fy = (fy_index-center)/n;
frequency = max(hypot(fx,fy),1/n);
theta = atan2(fy,fx);
xp = (x-x0)*cos(theta)+(y-y0)*sin(theta);
yp = -(x-x0)*sin(theta)+(y-y0)*cos(theta);
sigma_x = max(sqrt(sum(energy.*xp.^2,'all')/total),1);
sigma_y = max(sqrt(sum(energy.*yp.^2,'all')/total),1);
carrier = 2*pi*frequency*xp;
phase = atan2(-sum(z0.*sin(carrier),'all'),sum(z0.*cos(carrier),'all'));
p0 = [max(abs(z0),[],'all'),x0,y0,log(sigma_x),log(sigma_y), ...
    theta,log(frequency),phase,mean(z,'all')];
settings = optimset('Display','off','MaxIter',500,'MaxFunEvals',2000, ...
    'TolX',1e-7,'TolFun',1e-9);
% gabor_error：计算拟合出来的Gabor图案与实际感受野之间的差异。
p = fminsearch(@(candidate) gabor_error(candidate,z,x,y),p0,settings);
% make_gabor：把拟合参数转换成朝向、空间频率、相位等指标，并生成对应的二维Gabor图案。
[model,parameters] = make_gabor(p,x,y,n);
fit.fit_error = sum((z-model).^2,'all')/max(sum(z.^2,'all'),eps);
fit.orientation_deg = mod(rad2deg(parameters.theta),180);
fit.spatial_frequency = parameters.frequency*n;
fit.phase_deg = mod(rad2deg(parameters.phase)+180,360)-180;
fit.sigma_x = parameters.sigma_x;
fit.sigma_y = parameters.sigma_y;
fit.center_x = parameters.x0;
fit.center_y = parameters.y0;
fit.n_x = parameters.sigma_x*parameters.frequency;
fit.n_y = parameters.sigma_y*parameters.frequency;
end

% 作用：计算拟合出来的Gabor图案与实际感受野之间的差异。
function value = gabor_error(p,z,x,y)
n = size(z,1);

[model,parameters] = make_gabor(p,x,y,n);
value = sum((z-model).^2,'all');
outside = max([0,0.5-parameters.x0,parameters.x0-(n+0.5), ...
    0.5-parameters.y0,parameters.y0-(n+0.5), ...
    parameters.frequency-0.5]);
value = value+1e4*outside^2;
end

% 作用：把拟合参数转换成朝向、空间频率、相位等指标，并生成对应的二维Gabor图案。
function [model,p] = make_gabor(raw,x,y,n)
p.amplitude = raw(1);
p.x0 = raw(2); p.y0 = raw(3);
p.sigma_x = min(max(exp(raw(4)),0.3),2*n);
p.sigma_y = min(max(exp(raw(5)),0.3),2*n);
p.theta = raw(6);
p.frequency = min(max(exp(raw(7)),1/(4*n)),0.5);
p.phase = raw(8);
p.offset = raw(9);
xp = (x-p.x0)*cos(p.theta)+(y-p.y0)*sin(p.theta);
yp = -(x-p.x0)*sin(p.theta)+(y-p.y0)*cos(p.theta);
envelope = exp(-0.5*((xp/p.sigma_x).^2+(yp/p.sigma_y).^2));
model = p.offset+p.amplitude*envelope.*cos(2*pi*p.frequency*xp+p.phase);
end

% 作用：计算PS或LS。
% （dimension=1时算每张刺激的PS，dimension=2时算每个神经元的LS）

function values = corrected_sparseness(rates,dimension)
rates = double(rates);
n = size(rates,dimension);
mean_rate = mean(rates,dimension);
mean_square = mean(rates.^2,dimension);
ratio = mean_rate.^2./mean_square;
values = (1-ratio)/(1-1/n);
values(mean_square <= eps) = NaN;
values = min(max(values,0),1);
if dimension == 2, values = values(:); else, values = values(:); end
end

% 作用：计算所有有效行两两Pearson相关系数的RMS。

function value = pairwise_correlation_rms(data)
valid = all(isfinite(data),2) & std(data,0,2) > 0;
if nnz(valid) < 2
    value = NaN;
    return;
end
% row_correlation_matrix：计算矩阵中每两行数据之间的相关系数。
C = row_correlation_matrix(double(data(valid,:)));
mask = triu(true(size(C)),1) & isfinite(C);
value = sqrt(mean(C(mask).^2));
end

% 作用：从MAT中读取loose和tight E/I balance结果。
function output = load_balance_metrics(checkpoint_dir,condition_id,batch)
output = struct('loose_high_exc',NaN,'loose_high_inh',NaN, ...
    'loose_high_total',NaN,'tight_zero_lag',NaN, ...
    'tight_peak_correlation',NaN,'tight_peak_lag_ms',NaN);
if batch ~= 1200, return; end
file = fullfile(checkpoint_dir,sprintf( ...
    'pcb_shuffle_%s_paper_cn1_run01_balance.mat',lower(condition_id)));
if ~isfile(file), return; end
loaded = load(file,'balance_checkpoint');
b = loaded.balance_checkpoint.balance;
output.loose_high_exc = b.loose.group_mean_V_exc(3);
output.loose_high_inh = b.loose.group_mean_V_inh_signed(3);
output.loose_high_total = b.loose.group_mean_V_total(3);
output.tight_zero_lag = b.tight_population.zero_lag_mean;
output.tight_peak_correlation = b.tight_population.peak_correlation_mean;
output.tight_peak_lag_ms = b.tight_population.peak_lag_ms_mean;
end

% 作用：把每个E神经元的放电率、LS、光栅调谐和Gabor指标可视化图表。

function T = make_neuron_table(white_rates,lifetime,tuning,gabor)
n_E = size(white_rates,1);
T = table((1:n_E)','VariableNames',{'neuron'});
T.white_noise_rate_Hz = mean(white_rates,2);
T.lifetime_sparseness = lifetime;
T.blank_rate_Hz = tuning.blank_rate_Hz;
T.one_minus_CirVar = tuning.one_minus_CirVar;
T.preferred_orientation_deg = tuning.preferred_orientation_deg;
T.orientation_bandwidth_deg = tuning.orientation_bandwidth_deg;
T.orientation_fit_error = tuning.orientation_fit_error;
T.preferred_spatial_frequency_cycles_per_pixel = ...
    tuning.preferred_spatial_frequency_cycles_per_pixel;
T.preferred_spatial_frequency_cycles_per_image = ...
    tuning.preferred_spatial_frequency_cycles_per_image;
T.spatial_frequency_bandwidth_octaves = ...
    tuning.spatial_frequency_bandwidth_octaves;
T.spatial_frequency_fit_error = tuning.spatial_frequency_fit_error;
T.spatial_frequency_tuning_type = tuning.spatial_frequency_tuning_type;
T.gabor_orientation_deg = gabor.orientation_deg;
T.gabor_spatial_frequency = gabor.spatial_frequency;
T.gabor_phase_deg = gabor.phase_deg;
T.gabor_fit_error = gabor.fit_error;
T.gabor_well_fit = gabor.well_fit;
T.gabor_sigma_x = gabor.sigma_x;
T.gabor_sigma_y = gabor.sigma_y;
T.gabor_center_x = gabor.center_x;
T.gabor_center_y = gabor.center_y;
T.gabor_n_x = gabor.n_x;
T.gabor_n_y = gabor.n_y;
end

% 作用：把一个条件的PS、LS、相关RMS、调谐、Gabor和E/I平衡结果汇总成表格。

function T = make_network_table(condition_id,batch,population,lifetime, ...
        rf_corr,response_corr,tuning,gabor,balance)
T = table(condition_id,batch,'VariableNames',{'condition','batch'});
T.functional_population_sparseness = mean(population,'omitnan');
T.functional_lifetime_sparseness = mean(lifetime,'omitnan');
T.functional_RF_correlation_RMS = rf_corr;
T.response_correlation_RMS = response_corr;
T.mean_one_minus_CirVar = mean(tuning.one_minus_CirVar,'omitnan');
T.median_one_minus_CirVar = median(tuning.one_minus_CirVar,'omitnan');
T.gabor_well_fit_fraction = mean(gabor.well_fit,'omitnan');
T.loose_high_exc = balance.loose_high_exc;
T.loose_high_inh_signed = balance.loose_high_inh;
T.loose_high_total = balance.loose_high_total;
T.tight_zero_lag = balance.tight_zero_lag;
T.tight_peak_correlation = balance.tight_peak_correlation;
T.tight_peak_lag_ms = balance.tight_peak_lag_ms;
end

% 作用：统计六类突触权重(排除E→E和I→I的自连接)。

function [summary,histograms,weights] = analyse_weights(state,condition,batch,n_bins)
names = {'W_EX','W_EE','W_EI','W_IX','W_IE','W_II'};
labels = {'X_to_E','E_to_E','I_to_E','X_to_I','E_to_I','I_to_I'};
summary = table(); histograms = table(); weights = struct();
for index = 1:numel(names)
    W = double(state.(names{index}));
    if strcmp(names{index},'W_EE') || strcmp(names{index},'W_II')
        W(eye(size(W,1)) > 0) = NaN;
    end
    values = W(isfinite(W));
    weights.(names{index}) = W;
    row = table(condition,batch,string(labels{index}),numel(values), ...
        mean(values),median(values),std(values),min(values),max(values), ...
        nnz(values == 0),nnz(values > 0),'VariableNames',{'condition', ...
        'batch','synapse_type','n','mean','median','sd','minimum','maximum', ...
        'zero_count','positive_count'});
    summary = [summary;row]; %#ok<AGROW>
    [counts,edges] = histcounts(values,n_bins);
    H = table(repmat(condition,n_bins,1),repmat(batch,n_bins,1), ...
        repmat(string(labels{index}),n_bins,1),edges(1:end-1)',edges(2:end)', ...
        counts','VariableNames',{'condition','batch','synapse_type', ...
        'bin_left','bin_right','count'});
    histograms = [histograms;H]; %#ok<AGROW>
end
end

% 作用：把所有E神经元的感受野转换成CSV文件，每行对应一个神经元。

function write_rf_csv(rf,file)
n_pixels = size(rf,2);
names = cellstr(compose('pixel_%03d',1:n_pixels));
T = array2table(rf,'VariableNames',names);
T = addvars(T,(1:size(rf,1))','Before',1,'NewVariableNames','neuron');
writetable(T,file);
end

% 作用：调用各个绘图函数

function plot_analysis_figures(analysis,condition,batch,png_dir,n_bins)
rf = double(analysis.functional_rf);
% pairwise_correlation_rms：先计算所有神经元两两之间的相关系数，再用这些相关系数计算一个总体RMS值。
rf_corr_rms = pairwise_correlation_rms(rf);
prefix = char(lower(condition));
% plot_rf_mosaic：把所有神经元 16×16的感受野排成一张总图。
plot_rf_mosaic(rf,condition,batch,fullfile(png_dir,sprintf( ...
    '%s_functional_rf_batch%04d.png',prefix,batch)));
% plot_neuron_metrics：把每项神经元指标分别画成直方图并保存。
plot_neuron_metrics(analysis.neuron_metrics,condition,batch,png_dir);
% plot_rf_correlations：绘制全部有效感受野对的相关系数分布。
plot_rf_correlations(rf,rf_corr_rms,condition,batch,fullfile(png_dir, ...
    sprintf('%s_functional_rf_correlations_batch%04d.png',prefix,batch)));
% plot_weight_histograms：绘制六类突触的权重分布直方图。
plot_weight_histograms(analysis.weights,condition,batch,n_bins, ...
    fullfile(png_dir,sprintf('%s_weight_histograms_batch%04d.png', ...
    prefix,batch)));
% plot_non_near_zero_weight_histograms：删除原始直方图第一分箱后重新绘制权重分布直方图。
plot_non_near_zero_weight_histograms(analysis.weights,condition,batch,n_bins, ...
    fullfile(png_dir,sprintf( ...
    '%s_weight_histograms_without_near_zero_batch%04d.png',prefix,batch)));
end

% 作用：把所有神经元 16×16的感受野排成一张总图。

function plot_rf_mosaic(rf,condition,batch,file)
n_side = round(sqrt(size(rf,2)));
n_E = size(rf,1);
n_cols = ceil(sqrt(n_E)); n_rows = ceil(n_E/n_cols);
mosaic = nan(n_rows*n_side,n_cols*n_side);
for neuron = 1:n_E
    row = floor((neuron-1)/n_cols);
    col = mod(neuron-1,n_cols);
    mosaic(row*n_side+(1:n_side),col*n_side+(1:n_side)) = ...
        reshape(rf(neuron,:),n_side,n_side);
end
fig = figure('Visible','off','Color','w','Position',[100 100 1100 900]);
% redblue_colormap：生成蓝—白—红图表以区分负值和正值。
imagesc(mosaic); axis image off; colormap(redblue_colormap());
limit = max(abs(mosaic),[],'all','omitnan');
if isfinite(limit) && limit > 0, clim([-limit limit]); end
cb = colorbar; cb.Label.String = 'STA response amplitude';
title(sprintf('%s: Receptive Fields (STA)',condition),'FontWeight','bold');
% force_white_style：统一图片为白底、黑字和浅灰网格。
force_white_style(fig);
% export_white_png：覆盖同名PNG并写入白底图片。
export_white_png(fig,file); close(fig);
end

% 作用：把每项神经元指标分别画成直方图并保存
function plot_neuron_metrics(T,condition,batch,png_dir)
plots = {
    T.blank_rate_Hz,[],'Spontaneous firing rate (Hz)', ...
    'Spontaneous Firing Rate','spontaneous_firing_rate';
    T.one_minus_CirVar,0:0.05:1,'1-CirVar','1-CirVar','one_minus_cirvar';
    T.preferred_orientation_deg,0:11.25:180,'Preferred orientation (deg)','Preferred Orientation','preferred_orientation';
    T.orientation_bandwidth_deg,0:5:90,'Orientation bandwidth (deg)','Orientation Bandwidth','orientation_bandwidth';
    T.preferred_spatial_frequency_cycles_per_pixel,0.025:0.05:0.425,'Preferred spatial frequency (cycles/pixel)','Preferred Spatial Frequency','preferred_spatial_frequency';
    T.spatial_frequency_bandwidth_octaves,0:0.2:3,'Spatial-frequency bandwidth (octaves)','Spatial-Frequency Bandwidth','spatial_frequency_bandwidth';
    T.gabor_fit_error,[],'Gabor fit error','Gabor Fit Error','gabor_fit_error';
    T.gabor_phase_deg,-180:15:180,'Gabor phase (deg)','Gabor Phase','gabor_phase';
    T.gabor_orientation_deg,0:11.25:180,'Gabor orientation (deg)','Gabor Orientation','gabor_orientation'};
prefix = char(lower(condition));
for index = 1:size(plots,1)
    file = fullfile(png_dir,sprintf('%s_%s_batch%04d.png', ...
        prefix,plots{index,5},batch));
    fig = figure('Visible','off','Color','w','Position',[100 100 850 650]);
    ax = axes(fig);
    % hist_panel：过滤无效值并绘制当前指标的神经元计数直方图。
    hist_panel(ax,plots{index,1},plots{index,2},plots{index,3});
    title(ax,sprintf('%s: %s',condition,plots{index,4}), ...
        'FontWeight','bold');
    if strcmp(plots{index,5},'orientation_bandwidth')
        errors = T.orientation_fit_error(isfinite(T.orientation_fit_error));
        valid_count = nnz(isfinite(T.orientation_bandwidth_deg));
        if isempty(errors)
            annotation_text = sprintf('Valid fits: %d/%d',valid_count,height(T));
        else
            annotation_text = sprintf(['von Mises fit error (median): %.3f\n', ...
                'Valid fits: %d/%d'],median(errors,'omitnan'),valid_count,height(T));
        end
        text(ax,0.97,0.95,annotation_text,'Units','normalized', ...
            'HorizontalAlignment','right','VerticalAlignment','top', ...
            'BackgroundColor','white','Margin',5);
    elseif strcmp(plots{index,5},'preferred_spatial_frequency')
        valid_count = nnz(isfinite( ...
            T.preferred_spatial_frequency_cycles_per_pixel));
        annotation_text = sprintf('Valid fits: %d/%d',valid_count,height(T));
        text(ax,0.97,0.95,annotation_text,'Units','normalized', ...
            'HorizontalAlignment','right','VerticalAlignment','top', ...
            'BackgroundColor','white','Margin',5);
    elseif strcmp(plots{index,5},'spatial_frequency_bandwidth')
        errors = T.spatial_frequency_fit_error( ...
            isfinite(T.spatial_frequency_fit_error));
        valid_count = nnz(isfinite(T.spatial_frequency_bandwidth_octaves));
        if isempty(errors)
            annotation_text = sprintf('Valid fits: %d/%d',valid_count,height(T));
        else
            annotation_text = sprintf(['Log-Gaussian fit error (median): %.3f\n', ...
                'Valid fits: %d/%d'],median(errors,'omitnan'),valid_count,height(T));
        end
        text(ax,0.97,0.95,annotation_text,'Units','normalized', ...
            'HorizontalAlignment','right','VerticalAlignment','top', ...
            'BackgroundColor','white','Margin',5);
    elseif strcmp(plots{index,5},'gabor_orientation') || ...
            strcmp(plots{index,5},'gabor_phase')
        valid_count = nnz(T.gabor_well_fit);
        annotation_text = sprintf('Valid Gabor fits: %d/%d', ...
            valid_count,height(T));
        text(ax,0.97,0.95,annotation_text,'Units','normalized', ...
            'HorizontalAlignment','right','VerticalAlignment','top', ...
            'BackgroundColor','white','Margin',5);
    end
    % force_white_style：统一图片为白底、黑字和浅灰网格。
    force_white_style(fig);
    % export_white_png：覆盖已有PNG。
    export_white_png(fig,file);
    close(fig);
end

end

% 作用：在指定坐标轴上画一项指标的分布直方图。

function hist_panel(ax,values,edges,label)
values = values(isfinite(values));
if isempty(edges)
    histogram(ax,values,40,'Normalization','count');
else
    histogram(ax,values,edges,'Normalization','count');
end
if isempty(values)

    ylim(ax,[0 1]);
end
xlabel(ax,label); ylabel(ax,'Neuron count'); grid(ax,'on');
end

% 作用：绘制所有E神经元感受野两两Pearson相关系数的分布直方图。

function plot_rf_correlations(rf,rms_value,condition,batch,file)
valid = all(isfinite(rf),2) & std(rf,0,2) > 0;
% row_correlation_matrix：计算所有有效感受野行之间的Pearson相关矩阵。
C = row_correlation_matrix(double(rf(valid,:)));
values = C(triu(true(size(C)),1));
values = values(isfinite(values));
fig = figure('Visible','off','Color','w','Position',[100 100 850 650]);
histogram(values,-1:0.025:1,'Normalization','count', ...
    'FaceColor',[0.18 0.47 0.72],'EdgeColor','none');
    xlabel('Excitatory neuron receptive field correlation');
    ylabel('Excitatory neuron-pair count'); grid on;
    title(sprintf('%s: Pairwise Receptive Field Correlations (RMS = %.4f)', ...
        condition,rms_value), ...
        'FontWeight','bold');
% force_white_style：统一图片为白底、黑字和浅灰网格。
force_white_style(fig);
% export_white_png：覆盖已有PNG。
export_white_png(fig,file); close(fig);
end

% 作用：先删掉原始直方图第一分箱内的权重，再画剩余权重的分布。

function plot_non_near_zero_weight_histograms(weights,condition,batch,n_bins, ...
        file) %#ok<INUSD>
names = {'W_EX','W_EE','W_EI','W_IX','W_IE','W_II'};
labels = {'X to E','E to E','I to E','X to I','E to I','I to I'};
fig = figure('Visible','off','Color','w','Position',[50 50 1450 850]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
for index = 1:numel(names)
    ax = nexttile;
    values = weights.(names{index});
    values = values(isfinite(values));
    if isempty(values)
        first_left = NaN;
        first_right = NaN;
        removed_count = 0;
        retained = values;
    else
        [~,edges] = histcounts(values,n_bins);
        first_left = edges(1);
        first_right = edges(2);
        in_original_first_bin = values >= first_left & values < first_right;
        removed_count = sum(in_original_first_bin);
        retained = values(~in_original_first_bin);
    end
    if isempty(retained)
        text(ax,0.5,0.5,'No connections remain after removing first bin', ...
            'Units','normalized','HorizontalAlignment','center');
    else
        histogram(ax,retained,n_bins,'Normalization','count', ...
            'FaceColor',[0.18 0.47 0.72],'EdgeColor','none');
    end
    xlabel(ax,'Synaptic weight'); ylabel(ax,'Connection count'); grid(ax,'on');
    title(ax,sprintf('%s | first bin: [%.4g, %.4g) | removed: %d | retained: %d', ...
        labels{index},first_left,first_right,removed_count,numel(retained)), ...
        'Interpreter','none','FontSize',10);
end
title(tl,sprintf('%s: Synaptic Weight Distributions (Original First Bin Removed)', ...
    condition),'FontWeight','bold','Color','k','FontSize',16);
force_white_style(fig);
% export_white_png：安全覆盖已有PNG。
export_white_png(fig,file); close(fig);
end

% 作用：把六类突触权重画成2×3计数直方图。
% 输入：weights是六类权重，condition和batch用于标记，n_bins是分箱数，file是图片路径。
% 输出：没有返回值，图片直接保存到file。
function plot_weight_histograms(weights,condition,batch,n_bins,file)
names = {'W_EX','W_EE','W_EI','W_IX','W_IE','W_II'};
labels = {'X to E','E to E','I to E','X to I','E to I','I to I'};
fig = figure('Visible','off','Color','w','Position',[50 50 1450 850]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
for index = 1:numel(names)
    ax = nexttile;
    values = weights.(names{index});
    values = values(isfinite(values));
    histogram(ax,values,n_bins,'Normalization','count', ...
        'FaceColor',[0.18 0.47 0.72],'EdgeColor','none');
    xlabel(ax,'Synaptic weight'); ylabel(ax,'Connection count'); grid(ax,'on');
    title(ax,sprintf('%s | Zero-weight connections: %d', ...
        labels{index},nnz(values == 0)), ...
        'Interpreter','none','FontSize',10);
end
title(tl,sprintf('%s: Synaptic Weight Distributions',condition), ...
    'FontWeight','bold','Color','k','FontSize',16);
% force_white_style：统一图片为白底、黑字和浅灰网格。
force_white_style(fig);
% export_white_png：覆盖已有PNG。
export_white_png(fig,file); close(fig);
end

% 作用：把图统一设为白底、黑字和浅灰网格。

function force_white_style(fig)
set(fig,'Color','w','InvertHardcopy','off');
set(findall(fig,'Type','axes'),'Color','w','XColor','k','YColor','k', ...
    'GridColor',[0.82 0.82 0.82],'MinorGridColor',[0.90 0.90 0.90], ...
    'FontName','Arial');
set(findall(fig,'Type','text'),'Color','k','FontName','Arial');
set(findall(fig,'Type','ColorBar'),'Color','k','FontName','Arial');
end


function export_white_png(fig,file)
if isfile(file)
    delete(file);
end
exportgraphics(fig,file,'Resolution',220,'BackgroundColor','white');
end

% 作用：读取已保存的E/I电流数据，绘制loose balance和tight balance图表。

function plot_ei_balance_figures(checkpoint_dir,condition,png_dir)
condition_lower = lower(char(condition));
balance_file = fullfile(checkpoint_dir, ...
    sprintf('pcb_shuffle_%s_paper_cn1_run01_balance.mat',condition_lower));
if ~isfile(balance_file)
    warning('Missing E/I balance data: %s',balance_file);
    return;
end
loaded = load(balance_file,'balance_checkpoint');
b = loaded.balance_checkpoint.balance;

fig = figure('Visible','off','Color','w','Position',[100 100 920 720]);
ax = axes(fig); hold(ax,'on');

n_neurons = numel(b.loose.neuron_ids);
n_groups = 3;
thresholds = double(b.loose.thresholds(:));
if n_neurons ~= 40
    warning('Loose balance contains %d neurons; original paper used 40.', ...
        n_neurons);
end
if numel(thresholds) ~= n_neurons || any(~isfinite(thresholds)) || ...
        any(thresholds <= 0)
    error('Invalid loose-balance firing thresholds for condition %s.',condition);
end

series_means = {double(b.loose.time_mean_V_total), ...
    double(b.loose.time_mean_V_inh_signed), ...
    double(b.loose.time_mean_V_exc)};
series_traces = {double(b.loose.V_total),double(b.loose.V_inh_signed), ...
    double(b.loose.V_exc)};
series_colors = {[0.93 0.62 0.10],[0.47 0.20 0.61],[0.36 0.67 0.20]};
series_labels = {'Total','Inhibitory','Excitatory'};

for series = 1:numel(series_means)
    means = series_means{series} ./ thresholds;
    time_sd = reshape(std(series_traces{series},0,2,'omitnan'), ...
        n_neurons,n_groups) ./ thresholds;
    if ~isequal(size(means),[n_neurons n_groups])
        error('Unexpected loose-balance array size for condition %s.',condition);
    end
    for neuron = 1:n_neurons
        visibility = 'off';
        if neuron == 1, visibility = 'on'; end
        errorbar(ax,1:n_groups,means(neuron,:),time_sd(neuron,:),'-', ...
            'Color',series_colors{series},'LineWidth',0.75, ...
            'CapSize',3,'DisplayName',series_labels{series}, ...
            'HandleVisibility',visibility);
    end
end

yline(ax,1,'--','Spike threshold','Color',[0.25 0.25 0.25], ...
    'LineWidth',1.2,'LabelHorizontalAlignment','left', ...
    'HandleVisibility','off');
xlim(ax,[0.75 3.25]); xticks(ax,1:3);
xticklabels(ax,{'Anticorrelated','Uncorrelated','Correlated'});
xlabel(ax,'Image-receptive field correlation group');
ylabel(ax,'Normalized membrane voltage, V_m / V_{th}', ...
    'Interpreter','tex');
title(ax,sprintf('Condition %s: Loose E/I Balance (40 E neurons)',condition), ...
    'FontWeight','bold');

correlations = double(b.loose.patch_correlations);
if isequal(size(correlations),[n_neurons n_groups])
    correlation_min = min(correlations,[],1,'omitnan');
    correlation_max = max(correlations,[],1,'omitnan');
    range_text = sprintf(['Receptive field correlation ranges\n', ...
        'Low: [%.3f, %.3f]\nMiddle: [%.3f, %.3f]\n', ...
        'High: [%.3f, %.3f]'],correlation_min(1),correlation_max(1), ...
        correlation_min(2),correlation_max(2),correlation_min(3), ...
        correlation_max(3));
    text(ax,0.02,0.98,range_text,'Units','normalized', ...
        'HorizontalAlignment','left','VerticalAlignment','top', ...
        'BackgroundColor','white','EdgeColor',[0.75 0.75 0.75], ...
        'Margin',6,'FontSize',9);
end
legend(ax,'Location','southeast','Color','white','TextColor','k');
grid(ax,'on');
% force_white_style：统一loose balance图的白底样式。
force_white_style(fig);
loose_file = fullfile(png_dir,sprintf( ...
    '%s_loose_balance_errorbar.png',condition_lower));
% export_white_png：覆盖已有loose balance图片。
export_white_png(fig,loose_file);
close(fig);

fig = figure('Visible','off','Color','w','Position',[100 100 800 650]);
ax = axes(fig); hold(ax,'on');
curves = double(b.tight.correlation_curves);
lags = double(b.tight.lag_ms);
if any(isfinite(curves(:)))
    for image_index = 1:size(curves,1)
        plot(ax,lags,curves(image_index,:),'-','Color',[0.78 0.78 0.78], ...
            'LineWidth',0.8,'HandleVisibility','off');
    end
    plot(ax,lags,b.tight.mean_correlation,'k-','LineWidth',2.1, ...
        'DisplayName','Mean of high-input images');
    errorbar(ax,0,b.tight.zero_lag_mean,b.tight.zero_lag_ci95,'o', ...
        'Color',[0.82 0.25 0.20],'MarkerFaceColor','white', ...
        'LineWidth',1.4,'DisplayName','Lag 0, 95% CI');
    legend(ax,'Location','best','Color','white','TextColor','k');
else
    text(ax,0.5,0.5,'Undefined: inhibitory input is zero', ...
        'Units','normalized','HorizontalAlignment','center');
end
xline(ax,0,':','Color',[0.5 0.5 0.5],'HandleVisibility','off');
xlabel(ax,'Lag (ms)');
ylabel(ax,'Cross-correlation'); ylim(ax,[-1 1]); grid(ax,'on');
title(ax,sprintf('Condition %s: Tight E/I Cross-Correlation',condition), ...
    'FontWeight','bold');
% force_white_style：统一tight balance图的白底样式。
force_white_style(fig);
tight_file = fullfile(png_dir,sprintf( ...
    '%s_tight_balance_cross_correlation.png',condition_lower));
% export_white_png：覆盖已有tight balance图片。
export_white_png(fig,tight_file);
close(fig);
end

% 作用：生成感受野图使用的蓝—白—红色表。

function map = redblue_colormap()
n = 256;
map = [linspace(0,1,n/2)' linspace(0,1,n/2)' ones(n/2,1); ...
    ones(n/2,1) linspace(1,0,n/2)' linspace(1,0,n/2)'];
end

% 作用：计算矩阵所有行向量两两之间的Pearson相关系数。

function C = row_correlation_matrix(X)
X = X-mean(X,2);
norms = sqrt(sum(X.^2,2));
X = X./norms;
C = X*X';
end

% 作用：检查目录，不存在就创建。
function ensure_dir(path)
if ~isfolder(path), mkdir(path); end
end

% 作用：启动指定数量的Processes并行worker。

function ensure_parallel_pool(n_workers)
if ~license('test','Distrib_Computing_Toolbox')
    error('Parallel Computing Toolbox is required for this analysis.');
end
pool = gcp('nocreate');
if ~isempty(pool) && pool.NumWorkers ~= n_workers
    delete(pool);
    pool = [];
end
if isempty(pool)
    pool = parpool('Processes',n_workers);
end
fprintf('Parallel pool ready: %d workers.\n',pool.NumWorkers);
end

% 作用：使用几组结果已知的简单模拟数据，检查主要指标的计算函数是否正常。

function self_test()
n = 16; [x,y] = meshgrid(1:n,1:n);
z = exp(-((x-8).^2+(y-8).^2)/(2*3^2)).*cos(2*pi*0.12*(x-8)+0.7);
% fit_one_gabor：验证Gabor拟合是否正常。
fit = fit_one_gabor(z(:)');
assert(isfinite(fit.fit_error) && fit.fit_error < 0.05, ...
    'Gabor fitting self-test failed.');
gabor_test = fit_gabor_population([z(:)';ones(1,numel(z))]);
assert(gabor_test.well_fit(1) && isfinite(gabor_test.orientation_deg(1)) && ...
    isfinite(gabor_test.phase_deg(1)), ...
    'Valid Gabor fits must retain orientation and phase.');
assert(~gabor_test.well_fit(2) && isnan(gabor_test.orientation_deg(2)) && ...
    isnan(gabor_test.phase_deg(2)), ...
    'Invalid Gabor fits must be excluded from orientation and phase statistics.');
% corrected_sparseness：验证稀疏响应应比均匀响应得到更高LS。
s = corrected_sparseness([1 0 0;1 1 1],2);
assert(s(1) > s(2),'Sparseness self-test failed.');

% 构造一个已知二维调谐峰，验证跨相位平均、1-CirVar、偏好朝向和偏好SF。
test_ori = [0 45 90 135];
test_sf = [0.05 0.10 0.15];
test_phase = [0 90 180 270];
[O,S,P] = ndgrid(test_ori,test_sf,test_phase);
meta.orientation_deg = [O(:);NaN];
meta.spatial_frequency_cycles_per_pixel = [S(:);NaN];
meta.spatial_frequency_cycles_per_image = [16*S(:);NaN];
meta.phase_deg = [P(:);NaN];
meta.is_blank = [false(numel(O),1);true];
test_rates = ones(1,numel(O)+1);
peak = O == 45 & S == 0.10 & P == 90;
peak_index = find(peak(:));
test_rates(peak_index) = 20;
% 让固定SF=0.10并跨相位平均后，45度仍是明确的偏好朝向。
shoulder = O == 45 & S == 0.10 & ~peak;
test_rates(find(shoulder(:))) = 8;
test_tuning = calculate_tuning_metrics(test_rates,meta);
assert(abs(test_tuning.preferred_spatial_frequency_cycles_per_pixel-0.10) < 0.03, ...
    'Preferred-SF self-test failed.');
assert(abs(test_tuning.preferred_orientation_deg-45) < 1e-10, ...
    'Phase-averaged preferred-orientation self-test failed.');
assert(test_tuning.one_minus_CirVar > 0.95, ...
    '1-CirVar self-test failed.');
assert(isfinite(test_tuning.orientation_bandwidth_deg), ...
    'von Mises bandwidth self-test failed.');

% 平坦SF曲线没有可识别峰值：偏好SF和带宽必须同时留空，不能回退到最低网格频率。
flat_rates = 5*ones(1,numel(meta.is_blank));
flat_rates(meta.is_blank) = 0;
flat_tuning = calculate_tuning_metrics(flat_rates,meta);
assert(isnan(flat_tuning.preferred_spatial_frequency_cycles_per_pixel) && ...
    isnan(flat_tuning.spatial_frequency_bandwidth_octaves) && ...
    flat_tuning.spatial_frequency_tuning_type == "unreliable", ...
    'Invalid SF tuning must remain blank instead of using a grid fallback.');
fprintf('run_cn1_functional_rf_analysis self-test passed.\n');
end
