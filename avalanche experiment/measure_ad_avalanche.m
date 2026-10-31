% 作用：读取完整网络运行后的checkpoint，连续记录活动，再提取雪崩并计算DCC。
% 输入：检查点、输入单词natural/blank/noise、背景噪声强度、步数和保存位置。
% 输出：原始记录、逐bin计数、雪崩与拟合结果，并保存到本目录的results。
% simulate_frozen_ad_stream：保持原LIF更新顺序，换图不清空膜电位。
% calculate_avalanche_dcc：划分事件，用离散幂律MLE拟合，并检验标度关系。
% 本文件不会重新训练，也不会覆盖原来的AD检查点。
function result = measure_ad_avalanche(options)
if nargin<1, options = struct(); end
experiment_dir = fileparts(mfilename('fullpath'));
options = complete_options(options,experiment_dir);
validate_options(options);
output_dir='';
if options.save_outputs
    if isempty(options.output_dir)
        stamp=char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
        output_dir=fullfile(experiment_dir,'results', ...
            ['AD_cn1_' options.input_mode '_' stamp]);
    else
        output_dir=options.output_dir;
    end
    if isfolder(output_dir)
        entries=dir(output_dir);
        assert(~any(~ismember({entries.name},{'.','..'})),'PCB:ExistingOutput', ...
            'Output is not empty; choose a new directory to avoid overwriting files.');
    else
        mkdir(output_dir);
    end
end
project_dir = options.project_dir;
[state,source] = read_ad_checkpoint(options.checkpoint_file);
original_state = state;
cfg = struct('dt',1e-3,'tau_m',10e-3,'V_min',-10, ...
    'noise_sigma',options.background_noise_sigma, ...
    'total_steps',options.total_steps,'patch_steps',options.patch_steps, ...
    'progress_every_patches',options.progress_every_patches);
nBins = cfg.total_steps/options.bin_steps;
nPatches = ceil(cfg.total_steps/cfg.patch_steps);
fprintf('AD cn=1 | training seed %d | frozen checkpoint batch %d\n', ...
    source.training_seed,source.batch);
fprintf('%d steps = %.3f s | %d bins, %.1f ms/bin | %d recording blocks\n', ...
    cfg.total_steps,cfg.total_steps*cfg.dt,nBins, ...
    options.bin_steps*cfg.dt*1000,nPatches);

caller_rng = rng;
rng_cleanup = onCleanup(@() rng(caller_rng));
% make_input_rates：按入口单词准备输入；blank/noise不读取任何图像文件。
[rates,input_info] = make_input_rates(options,project_dir,nPatches,size(state.W_EX,2));
fprintf('Input: %s | background sigma: %g | mean LGN channel rate: %.4g Hz\n', ...
    options.input_mode,cfg.noise_sigma,input_info.mean_LGN_channel_rate_Hz);
if strcmp(options.input_mode,'noise')
    fprintf('Independent LGN noise: %.4g Hz/channel; not a white-noise image stimulus.\n', ...
        options.noise_rate_hz);
elseif strcmp(options.input_mode,'blank')
    fprintf('Blank: no LGN spikes, no initial pulse. Background noise sigma remains %g.\n',cfg.noise_sigma);
end
rng(options.dynamics_seed,'twister');
recording = simulate_frozen_ad_stream(state,rates,cfg);
assert(isequaln(state,original_state),'PCB:StateChanged','Frozen state changed.');

% 这是按时间累计真实spike数，不是把每个细胞在整个bin内压成0或1。
bin_map = sparse((1:cfg.total_steps)', ...
    ceil((1:cfg.total_steps)'/options.bin_steps),1,cfg.total_steps,nBins);
counts_E = uint16(full(recording.spikes_E*bin_map));
counts_I = uint16(full(recording.spikes_I*bin_map));
activity_E = sum(double(counts_E),1)';
activity_I = sum(double(counts_I),1)';
input_spike_count = sum(reshape(double(recording.step_count_X),options.bin_steps,[]),1)';
assert(isequal(activity_E,sum(reshape(double(recording.step_count_E), ...
    options.bin_steps,[]),1)'),'PCB:BinCountMismatch','E bin counts disagree.');
assert(isequal(activity_I,sum(reshape(double(recording.step_count_I), ...
    options.bin_steps,[]),1)'),'PCB:BinCountMismatch','I bin counts disagree.');
% bin的开始/结束时间指完整50步时间区间；spike_time_step保留原始1ms步编号。
bin_start_step = (0:nBins-1)'*options.bin_steps+1;
bin_end_step = (1:nBins)'*options.bin_steps;
input_per_step = repelem(sum(rates,1)',cfg.patch_steps);
input_per_step = input_per_step(1:cfg.total_steps);
input_per_bin = mean(reshape(input_per_step,options.bin_steps,[]),1)';
first_patch = floor((bin_start_step-1)/cfg.patch_steps)+1;
last_patch = floor((bin_end_step-1)/cfg.patch_steps)+1;
bin_table = table((1:nBins)',bin_start_step,bin_end_step, ...
    (bin_start_step-1)*cfg.dt,bin_end_step*cfg.dt,activity_E,activity_I, ...
    input_per_bin,first_patch,last_patch, ...
    'VariableNames',{'bin','start_step','end_step','start_seconds','end_seconds', ...
    'E_spike_count','I_spike_count','LGN_total_rate_Hz','first_patch','last_patch'});
bin_table.LGN_spike_count = input_spike_count;
bin_table.input_mode = repmat(string(options.input_mode),nBins,1);
if ~strcmp(options.input_mode,'natural')
    % 没有图片时，400步只是存储分块，不是假装每400步更换了一张刺激。
    bin_table.first_patch(:)=NaN;
    bin_table.last_patch(:)=NaN;
end
patch_start = (0:nPatches-1)'*cfg.patch_steps+1;
patch_end = min((1:nPatches)'*cfg.patch_steps,cfg.total_steps);
patch_table = table((1:nPatches)',patch_start,patch_end,(patch_start-1)*cfg.dt, ...
    sum(rates,1)',mean(rates(1:256,:),1)',mean(rates(257:512,:),1)', ...
    'VariableNames',{'patch','start_step','end_step','start_seconds', ...
    'LGN_total_rate_Hz','ON_mean_rate_Hz','OFF_mean_rate_Hz'});
patch_table.input_mode = repmat(string(options.input_mode),nPatches,1);
patch_table.is_visual_patch = repmat(strcmp(options.input_mode,'natural'),nPatches,1);
result = struct('schema_version',1,'source',source,'options',options,'cfg',cfg, ...
    'input',input_info, ...
    'frozen_state',state,'matlab_release',version('-release'), ...
    'recording',recording,'counts_E_per_bin',counts_E,'counts_I_per_bin',counts_I, ...
    'activity_E',activity_E,'activity_I',activity_I,'input_rates_Hz',rates, ...
    'bin_table',bin_table,'patch_table',patch_table,'analysis',struct('status','not_analyzed'), ...
    'weights_and_thresholds_unchanged',true, ...
    'note','Fresh frozen trajectory from zero membrane voltage; burn-in=0; not continuation of training V.');
if options.save_outputs
    result.output_dir=output_dir;
    % 网络记录先落盘；统计报错时仍可加载此文件，单独重新分析，不必重跑网络。
    save_result(result,output_dir);
end
analysis_options=options.analysis_options;
analysis_options.threshold_percentile=options.threshold_percentile;
analysis_options.bootstrap_reps=options.bootstrap_samples;
analysis_options.bootstrap_seed=options.bootstrap_seed;
analysis=calculate_avalanche_dcc(activity_E,options.bin_steps,cfg.dt,analysis_options);
result.analysis=analysis;
if options.save_outputs
    save_result(result,output_dir);
    writetable(bin_table,fullfile(output_dir,'network_activity.csv'));
    if strcmp(options.input_mode,'natural')
        writetable(patch_table,fullfile(output_dir,'patch_schedule.csv'));
    else
        input_table=renamevars(patch_table,'patch','input_block');
        writetable(input_table,fullfile(output_dir,'input_schedule.csv'));
    end
    writetable(analysis.events,fullfile(output_dir,'avalanches.csv'));
    if strcmp(options.input_mode,'natural')
        % 只有natural真实换图；恒定blank/noise没有400ms换图周期，不生成此诊断。
        phase=mod(bin_start_step-1,cfg.patch_steps)+1;
        [phase_values,~,group]=unique(phase);
        start_counts=accumarray(analysis.events.start_bin,ones(height(analysis.events),1),[nBins,1]);
        end_counts=accumarray(analysis.events.end_bin,ones(height(analysis.events),1),[nBins,1]);
        phase_table=table(phase_values,accumarray(group,1), ...
            accumarray(group,activity_E,[],@mean),accumarray(group,double(analysis.active),[],@mean), ...
            accumarray(group,start_counts),accumarray(group,end_counts), ...
            'VariableNames',{'bin_start_step_within_patch','bin_count','mean_E_spike_count', ...
            'active_bin_fraction','event_start_count','event_end_count'});
        writetable(phase_table,fullfile(output_dir,'patch_phase_diagnostics.csv'));
    end
    neuron_names = cellstr(compose('E%03d',1:size(counts_E,1)));
    neuron_table = array2table(double(counts_E'),'VariableNames',neuron_names);
    neuron_table = addvars(neuron_table,(1:nBins)','Before',1,'NewVariableNames','bin');
    writetable(neuron_table,fullfile(output_dir,'E_neuron_bin_counts.csv'));
    summary = make_summary(result);
    writetable(summary,fullfile(output_dir,'dcc_summary.csv'));
    diagnostics=fit_diagnostics(analysis);
    writetable(diagnostics,fullfile(output_dir,'fit_diagnostics.csv'));
    if options.make_figures, plot_ad_avalanche(result,output_dir); end
    fprintf('Saved to: %s\n',output_dir);
end

fprintf('tau=%g | alpha=%g | beta_fit=%g | beta_predicted=%g | DCC=%g\n', ...
    analysis.tau,analysis.alpha,analysis.beta_fit,analysis.beta_predicted,analysis.dcc);
fprintf('Fit valid: %d | %s\n',analysis.valid,char(analysis.reason));
fprintf('DCC alone does not establish criticality. No automatic tuning to DCC < 0.2.\n');
end

% 作用：单独列出size和duration拟合的区间、样本数与KS检验，避免只看一个DCC数值。
function T=fit_diagnostics(analysis)
fits={analysis.fit_size;analysis.fit_duration};
T=table(["Size";"Duration"],zeros(2,1),zeros(2,1),zeros(2,1),zeros(2,1), ...
    zeros(2,1),zeros(2,1),zeros(2,1),zeros(2,1),false(2,1),strings(2,1), ...
    'VariableNames',{'distribution','exponent','xmin','xmax','fit_event_count', ...
    'unique_values','retained_fraction','KS','bootstrap_p','valid','reason'});
for k=1:2
    f=fits{k};
    T.exponent(k)=f.exponent; T.xmin(k)=f.xmin; T.xmax(k)=f.xmax;
    T.fit_event_count(k)=f.n; T.unique_values(k)=f.unique;
    T.retained_fraction(k)=f.fit_fraction; T.KS(k)=f.ks;
    T.bootstrap_p(k)=f.p_value; T.valid(k)=f.valid; T.reason(k)=string(f.reason);
end
T.exponent_ci95_low=[analysis.fit_size.exponent_ci95(1);analysis.fit_duration.exponent_ci95(1)];
T.exponent_ci95_high=[analysis.fit_size.exponent_ci95(2);analysis.fit_duration.exponent_ci95(2)];
end

% 作用：先写临时MAT再替换本次运行的MAT，避免统计/存盘中断破坏已经保存的原始记录。
% 输入result是待保存内容，output_dir是本次新建的目录；不会接触原训练检查点。
function save_result(result,output_dir)
temporary=fullfile(output_dir,'ad_avalanche_result_writing.mat');
save(temporary,'result','-v7.3');
movefile(temporary,fullfile(output_dir,'ad_avalanche_result.mat'),'f');
end

% 作用：补齐入口没指定的参数。输入options是要改的设置，输出为完整设置。
function options = complete_options(options,experiment_dir)
defaults = struct('project_dir',fileparts(experiment_dir), ...
    'checkpoint_file',fullfile(experiment_dir,'input', ...
    'pcb_shuffle_ad_paper_cn1_run01_batch1200.mat'), ...
    'input_mode','natural','noise_rate_hz',10,'background_noise_sigma',1, ...
    'total_steps',200000,'bin_steps',50,'patch_steps',400, ...
    'patch_seed',20261023,'dynamics_seed',20261024,'bootstrap_seed',20261025, ...
    'threshold_percentile',35,'bootstrap_samples',1000, ...
    'progress_every_patches',10,'make_figures',true,'save_outputs',true, ...
    'output_dir','','analysis_options',struct());
names = fieldnames(defaults);
for k=1:numel(names)
    if ~isfield(options,names{k}), options.(names{k})=defaults.(names{k}); end
end
if ~((ischar(options.input_mode) && isrow(options.input_mode)) || ...
        (isstring(options.input_mode) && isscalar(options.input_mode) && ~ismissing(options.input_mode)))
    error('PCB:InputMode','input_mode must be one word: natural, blank, or noise.');
end
options.input_mode=lower(strtrim(char(options.input_mode)));
end

% 作用：防止错误步数、半个bin或者错误类型进入仿真。无输出；错误时直接停止。
function validate_options(options)
assert(any(strcmp(options.input_mode,{'natural','blank','noise'})), ...
    'PCB:InputMode','Unknown input_mode. Use natural, blank, or noise.');
validateattributes(options.background_noise_sigma,{'numeric'}, ...
    {'real','finite','scalar','nonnegative'});
validateattributes(options.noise_rate_hz,{'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',100});
for name={'total_steps','bin_steps','patch_steps'}
    validateattributes(options.(name{1}),{'numeric'},{'scalar','integer','positive','finite'});
end
assert(mod(options.total_steps,options.bin_steps)==0,'PCB:PartialBin', ...
    'total_steps must be an exact multiple of bin_steps.');
assert(options.bin_steps<=65535,'PCB:CountRange','bin_steps exceeds uint16 storage.');
validateattributes(options.threshold_percentile,{'numeric'},{'scalar','>=',0,'<=',100});
validateattributes(options.bootstrap_samples,{'numeric'},{'scalar','integer','nonnegative'});
for name={'patch_seed','dynamics_seed','bootstrap_seed'}
    validateattributes(options.(name{1}),{'numeric'},{'scalar','integer','>=',0,'<=',2^32-1});
end
end

% 作用：根据natural、blank或noise，准备每段时间的LGN放电率。
% 输入：options说明选哪种输入、noise的放电率和背景噪声；project_dir是项目根目录。
% 输入：nBlocks是记录分块数量；nX是LGN通道数，当前cn=1为512。
% 输出：rates给网络提供LGN放电率，大小为nX×nBlocks，单位Hz。
% 输出：info记录实际采用的输入类型、平均输入率和背景噪声，避免比较时混淆条件。
% natural调用generate_image_patches抽图；blank/noise不依赖图像文件。
% 随机LGN脉冲仍由simulate_frozen_ad_stream逐毫秒生成；这里的noise不另加高斯电流。
function [rates,info] = make_input_rates(options,project_dir,nBlocks,nX)
info=struct('mode',options.input_mode,'description','','noise_rate_hz',NaN, ...
    'background_noise_sigma',options.background_noise_sigma,'initial_pulse',false);
switch options.input_mode
    case 'natural'
        generator=fullfile(project_dir,'generate_image_patches.m');
        assert(isfile(generator),'PCB:MissingDependency', ...
            'Cannot find generate_image_patches.m in project_dir.');
        addpath(project_dir);
        images_file=fullfile(project_dir,'VanHateren_DoG_small.mat');
        assert(isfile(images_file),'PCB:MissingImages','Missing image dataset: %s',images_file);
        data=load(images_file,'IMAGES_DoG');
        assert(isfield(data,'IMAGES_DoG'),'PCB:ImageVariable','Dataset lacks IMAGES_DoG.');
        rng(options.patch_seed,'twister');
        rates=generate_image_patches(data.IMAGES_DoG,16^2,nBlocks);
        rates(rates>100)=100;
        info.description='Natural DoG image patches with ON/OFF rates; one patch per block.';
    case 'blank'
        rates=zeros(nX,nBlocks);
        info.description='Zero LGN rates throughout; no initial kick; background noise is separately specified.';
    case 'noise'
        rates=repmat(options.noise_rate_hz,nX,nBlocks);
        info.noise_rate_hz=options.noise_rate_hz;
        info.description='Independent Bernoulli LGN spikes at constant rates; not spatial Gaussian white-noise images.';
end
info.mean_LGN_channel_rate_Hz=mean(rates(:));
info.mean_LGN_total_rate_Hz=mean(sum(rates,1));
end

% 作用：只接受cn=1、AD、1200批的完整终点。输入是MAT路径，输出是权重阈值和来源。
function [state,source] = read_ad_checkpoint(path)
assert(isfile(path),'PCB:MissingCheckpoint', ...
    'Missing AD checkpoint. Set options.checkpoint_file to a complete checkpoint MAT: %s',path);
data = load(path,'checkpoint');
assert(isfield(data,'checkpoint'),'PCB:CheckpointVariable','MAT must contain checkpoint.');
c = data.checkpoint;
assert(c.schema_version==1 && string(c.condition)=="AD" && c.network_scale==1 ...
    && c.batch==1200 && c.final_batch==1200,'PCB:WrongCheckpoint', ...
    'Expected a complete AD cn=1 checkpoint at batch 1200.');
state = c.state;
names={'W_EX','W_EE','W_EI','W_IX','W_IE','W_II','theta_E','theta_I'};
shapes={[400 512],[400 400],[400 100],[100 512],[100 400],[100 100],[400 1],[100 1]};
for k=1:numel(names)
    x=state.(names{k});
    assert(isequal(size(x),shapes{k}) && isreal(x) && ...
        all(isfinite(x(:))) && all(x(:)>=0),'PCB:StateValues', ...
        'Invalid dimensions or values in %s.',names{k});
end
assert(all(diag(state.W_EE)==0) && all(diag(state.W_II)==0), ...
    'PCB:SelfConnections','Unexpected recurrent self-connections.');
source = struct('checkpoint_file',path,'training_seed',c.training_seed, ...
    'batch',c.batch,'final_batch',c.final_batch,'condition','AD','cn',1, ...
    'E_count',400,'I_count',100,'LGN_count',512, ...
    'E_threshold_range',[min(state.theta_E),max(state.theta_E)], ...
    'I_threshold_range',[min(state.theta_I),max(state.theta_I)]);
end

% 作用：把拟合结果整理成一行CSV。输入是完整result，输出是摘要表。
% dcc_below_reference只是数值比较，不表示已经证明临界；必须同时读fit_valid和说明。
function T = make_summary(result)
a=result.analysis;
complete_count=sum(a.events.complete);
T=table(result.source.training_seed,result.options.total_steps,result.cfg.dt, ...
    result.options.bin_steps,result.options.bin_steps*result.cfg.dt*1000, ...
    result.options.patch_steps,numel(result.activity_E),a.threshold, ...
    height(a.events),complete_count,a.tau,a.alpha,a.beta_fit,a.beta_predicted,a.dcc, ...
    a.fit_size.p_value,a.fit_duration.p_value,a.valid,string(a.reason), ...
    0.2,isfinite(a.dcc)&&a.dcc<0.2, ...
    "Not established by DCC alone", ...
    'VariableNames',{'training_seed','total_steps','dt_seconds','bin_steps','bin_ms', ...
    'patch_steps','bin_count','activity_threshold','all_event_count','complete_event_count', ...
    'tau','alpha','beta_fit','beta_predicted','DCC','size_bootstrap_p','duration_bootstrap_p', ...
    'fit_valid','fit_reason','DCC_reference','dcc_below_reference','criticality_conclusion'});
T.active_bin_fraction=mean(a.active);
T.threshold_tie_fraction=mean(result.activity_E==a.threshold);
T.scaling_duration_classes=sum(a.scaling.use_for_fit);
T.scaling_r_squared=a.scaling.r_squared;
T.burn_in_steps=0;
T.input_mode=string(result.input.mode);
T.background_noise_sigma=result.input.background_noise_sigma;
T.noise_rate_hz=result.input.noise_rate_hz;
T.mean_LGN_channel_rate_Hz=result.input.mean_LGN_channel_rate_Hz;
T.total_LGN_spike_count=sum(double(result.recording.step_count_X));
T.initial_pulse=false;
end
