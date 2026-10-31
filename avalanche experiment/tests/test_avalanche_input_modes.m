% 作用：用合成AD检查点检查输入切换，不读取真实网络或自然图像。
% 输入：无。输出：无；不符合预期时直接报错，不生成正式实验结果。
% 检查blank是否真的没有LGN脉冲、背景噪声能否单独关闭，以及noise能否重复。
function test_avalanche_input_modes()
old_rng=rng;
rng_cleanup=onCleanup(@() rng(old_rng));
fixture_dir=tempname;
mkdir(fixture_dir);
checkpoint_file=fullfile(fixture_dir,'synthetic_ad_checkpoint.mat');
fixture_cleanup=onCleanup(@() remove_fixture(fixture_dir,checkpoint_file));

% 六类权重尺寸与真实cn=1一致；不设复发连接，便于分开检查LGN和背景噪声。
state=struct('W_EX',ones(400,512)*.25,'W_EE',zeros(400,400), ...
    'W_EI',zeros(400,100),'W_IX',ones(100,512)*.1, ...
    'W_IE',zeros(100,400),'W_II',zeros(100,100), ...
    'theta_E',ones(400,1),'theta_I',ones(100,1));
checkpoint=struct('schema_version',1,'condition','AD','network_scale',1, ...
    'batch',1200,'final_batch',1200,'training_seed',7,'state',state);
save(checkpoint_file,'checkpoint');
options=struct('checkpoint_file',checkpoint_file, ...
    'project_dir',fullfile(fixture_dir,'no_image_dataset'), ...
    'total_steps',200,'bin_steps',10,'patch_steps',40, ...
    'patch_seed',71,'dynamics_seed',82,'bootstrap_seed',93, ...
    'bootstrap_samples',0,'progress_every_patches',0, ...
    'save_outputs',false,'make_figures',false, ...
    'input_mode','blank','background_noise_sigma',0,'noise_rate_hz',10);

% 没有LGN输入、背景噪声和起始刺激时，零初始电位应始终保持为零。
caller_rng=rng;
blank=measure_ad_avalanche(options);
assert(isequal(rng,caller_rng),'The analysis changed the caller RNG state.');
assert(strcmp(blank.input.mode,'blank') && isnan(blank.input.noise_rate_hz), ...
    'Blank input metadata is wrong.');
assert(blank.input.background_noise_sigma==0 && ~blank.input.initial_pulse, ...
    'Blank mode unexpectedly adds background noise or an initial pulse.');
assert(all(blank.input_rates_Hz(:)==0) && all(blank.recording.step_count_X==0), ...
    'Blank mode generated LGN drive.');
assert(nnz(blank.recording.spikes_E)==0 && nnz(blank.recording.spikes_I)==0, ...
    'Zero-drive, zero-noise network generated spikes.');
assert(all(blank.activity_E==0) && all(blank.activity_I==0) && ...
    all(blank.recording.final_V_E==0) && all(blank.recording.final_V_I==0), ...
    'A blank deterministic trajectory did not remain at zero.');
assert(isnan(blank.analysis.dcc) && ~blank.analysis.valid, ...
    'A silent network must not be labelled critical or assigned DCC=0.');
assert(isequaln(blank.frozen_state,state),'The test changed frozen weights or thresholds.');

% 保留原模型的背景噪声不等于增加LGN输入；使用低合成阈值保证能看见背景驱动。
options.input_mode=' BLANK ';
options.background_noise_sigma=1;
with_background=measure_ad_avalanche(options);
assert(strcmp(with_background.input.mode,'blank'), ...
    'Input mode is not normalized for capitalization and surrounding spaces.');
assert(all(with_background.input_rates_Hz(:)==0) && ...
    all(with_background.recording.step_count_X==0), ...
    'Background noise was incorrectly counted as LGN spikes.');
assert(nnz(with_background.recording.spikes_E)>0 && ...
    nnz(with_background.recording.spikes_I)>0, ...
    'The enabled background noise did not reach the synthetic low-threshold cells.');

% 固定LGN率产生随机脉冲；不应变成每步固定脉冲数或额外高斯电流。
options.input_mode="NoIsE";
options.background_noise_sigma=0;
noise=measure_ad_avalanche(options);
assert(strcmp(noise.input.mode,'noise') && noise.input.noise_rate_hz==10 && ...
    noise.input.background_noise_sigma==0 && ~noise.input.initial_pulse, ...
    'Noise input metadata is wrong.');
assert(all(noise.input_rates_Hz(:)==10),'Noise LGN rates are not constant 10 Hz.');
assert(sum(double(noise.recording.step_count_X))>0 && ...
    numel(unique(noise.recording.step_count_X))>1, ...
    'Noise input did not generate variable, nonzero LGN spike counts.');
assert(nnz(noise.recording.spikes_E)>0,'Noise input did not drive the synthetic E cells.');
assert(sum(double(noise.recording.step_count_E))==sum(noise.activity_E), ...
    'Noise E spike totals disagree before and after binning.');
assert(isequaln(noise.frozen_state,state),'Noise testing changed the frozen network.');

% 相同种子必须重现同一条轨迹，运行耗时不参与比较。
repeated=measure_ad_avalanche(options);
assert(isequal(noise.input_rates_Hz,repeated.input_rates_Hz) && ...
    isequal(noise.recording.step_count_X,repeated.recording.step_count_X) && ...
    isequal(noise.recording.spikes_E,repeated.recording.spikes_E) && ...
    isequal(noise.recording.spikes_I,repeated.recording.spikes_I) && ...
    isequal(noise.recording.final_V_E,repeated.recording.final_V_E) && ...
    isequal(noise.recording.final_V_I,repeated.recording.final_V_I) && ...
    isequaln(noise.analysis,repeated.analysis), ...
    'Repeated noise runs with identical seeds disagree.');

% 把noise率设为零，应回到blank、无背景噪声时的轨迹。
options.noise_rate_hz=0;
zero_rate=measure_ad_avalanche(options);
assert(all(zero_rate.recording.step_count_X==0) && ...
    isequal(zero_rate.recording.spikes_E,blank.recording.spikes_E) && ...
    isequal(zero_rate.recording.spikes_I,blank.recording.spikes_I), ...
    'Zero-rate noise does not match deterministic blank input.');

% natural仍应检查自己的依赖，而blank/noise已证明不需要自然图像文件。
bad=options; bad.input_mode='natural';
must_fail(@() measure_ad_avalanche(bad),'PCB:MissingDependency');
bad=options; bad.input_mode='not_a_mode';
must_fail(@() measure_ad_avalanche(bad),'PCB:InputMode');
bad=options; bad.input_mode={'blank'};
must_fail(@() measure_ad_avalanche(bad),'PCB:InputMode');
bad=options; bad.background_noise_sigma=-1;
must_fail(@() measure_ad_avalanche(bad),'');
bad=options; bad.background_noise_sigma=NaN;
must_fail(@() measure_ad_avalanche(bad),'');
bad=options; bad.noise_rate_hz=101;
must_fail(@() measure_ad_avalanche(bad),'');
bad=options; bad.noise_rate_hz=-1;
must_fail(@() measure_ad_avalanche(bad),'');
bad=options; bad.noise_rate_hz=NaN;
must_fail(@() measure_ad_avalanche(bad),'');
fprintf('test_avalanche_input_modes passed.\n');
end

% 作用：确认错误选项会被拒绝。输入是待执行操作和预期错误编号，无输出。
function must_fail(operation,expected_identifier)
caught=false;
try
    operation();
catch problem
    caught=true;
    if ~isempty(expected_identifier)
        assert(strcmp(problem.identifier,expected_identifier), ...
            'Unexpected error identifier: %s',problem.identifier);
    end
end
assert(caught,'An invalid input option was accepted.');
end

% 作用：只删除本测试创建的检查点，再删除已经清空的临时目录；不递归清理。
function remove_fixture(fixture_dir,checkpoint_file)
if isfile(checkpoint_file), delete(checkpoint_file); end
if isfolder(fixture_dir)
    entries=dir(fixture_dir);
    if ~any(~ismember({entries.name},{'.','..'})), rmdir(fixture_dir); end
end
end
