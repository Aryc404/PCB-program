% 作用：用手工事件和纯合成脉冲计数测试统计函数，不启动或训练神经网络。
% 输入：无；在本目录运行test_calculate_avalanche_dcc即可。
% 输出：results记录通过的检查；任一断言失败就停止并明确报错。
% 注意：测试中的事件数门槛及bootstrap次数会降低，只为快速验算，不代替正式1000次检验。
function results = test_calculate_avalanche_dcc()
saved_rng = rng; cleanup = onCleanup(@() rng(saved_rng)); %#ok<NASGU>
results = struct();
o = struct('bootstrap_reps',0);

% 开头和结尾的未完成事件保留，但不能进入拟合。
a = calculate_avalanche_dcc([3 0 1 2 0 0 4 5 0 6],50,.001,o);
assert(a.threshold == 0);
assert(isequal(a.events.start_bin,[1;3;7;10]));
assert(isequal(a.events.end_bin,[1;4;8;10]));
assert(isequal(a.events.size_spikes,[3;3;9;6]));
assert(isequal(a.events.duration_bins,[1;2;2;1]));
assert(isequal(a.events.duration_ms,[50;100;100;50]));
assert(isequal(a.events.complete,[false;true;true;false]));
assert(a.n_complete_events == 2 && ~a.valid);
results.edges_and_duration_units = true;

% 阈值大量重复时坚持严格大于，不把等于阈值的箱偷偷当成活跃。
a = calculate_avalanche_dcc([0 1 1 1 1 1 1 1 1 2 0],50,.001,o);
assert(a.threshold == 1 && nnz(a.active) == 1);
assert(height(a.events) == 1 && a.events.start_bin == 10 && a.events.size_spikes == 2);
results.strict_threshold_ties = true;

% size是原始总数，不扣阈值；这里阈值1，事件中3+4应为7，不是5。
q = o; q.threshold_percentile = 50;
a = calculate_avalanche_dcc([0 1 3 4 1 0],50,.001,q);
assert(a.threshold == 1 && height(a.events) == 1 && a.events.size_spikes == 7);
results.raw_spike_size_not_threshold_excess = true;

% 沉寂或所有箱相同，不伪造零DCC或临界结果。
a = calculate_avalanche_dcc(zeros(40,1),50,.001,o);
b = calculate_avalanche_dcc(ones(40,1),50,.001,o);
assert(~a.valid && isempty(a.events) && isnan(a.dcc));
assert(~b.valid && isempty(b.events) && isnan(b.dcc));
results.degenerate_records = true;

% 不足样本也可保留数学诊断；synthetic事件精确满足size=duration^2。
d = (1:12)'; multiplicity = max(1,round(300*d.^(-2)));
activity = make_activity(d,multiplicity,2);
q = struct('threshold_percentile',0,'bootstrap_reps',0,'min_events',10000,'min_unique',4, ...
    'min_events_per_duration',1,'min_scaling_durations',3);
a = calculate_avalanche_dcc(activity,50,.001,q);
assert(~a.valid && isfinite(a.raw_dcc) && a.dcc == a.raw_dcc);
assert(abs(a.beta_fit-2) < 1e-10);
assert(a.scaling.n_events == sum(multiplicity));
results.invalid_quality_keeps_raw_dcc = true;

% 从已知离散幂律构造近似频数，检查固定cutoff bootstrap、复现性和RNG不外泄。
d = (1:12)'; multiplicity = max(1,round(1500*d.^(-1.8)/sum(d.^(-1.8))));
activity = make_activity(d,multiplicity,1);
q = struct('threshold_percentile',0,'bootstrap_reps',12,'bootstrap_seed',73,'min_events',20,'min_unique',4, ...
    'min_events_per_duration',1,'min_scaling_durations',3);
rng(981,'twister'); state = rng;
a = calculate_avalanche_dcc(activity,50,.001,q);
assert(isequal(rng,state),'The analysis changed the caller random stream.');
b = calculate_avalanche_dcc(activity,50,.001,q);
assert(isequaln(a.fit_size.bootstrap_ks,b.fit_size.bootstrap_ks));
assert(numel(a.fit_size.bootstrap_ks) == 12 && numel(a.fit_duration.bootstrap_ks) == 12);
assert(a.fit_size.p_value > 0 && a.fit_size.p_value <= 1);
assert(a.fit_duration.p_value > 0 && a.fit_duration.p_value <= 1);
assert(numel(a.fit_size.bootstrap_exponents) == 12 && all(isfinite(a.fit_size.exponent_ci95)));
assert(abs(a.beta_fit-1) < 1e-10 && a.raw_dcc < 1e-7);
assert(a.fit_size.n >= q.min_events && a.fit_size.support_ratio >= 3);
results.bootstrap_and_reproducibility = true;

% 输入若不是原始整数计数，应报错而不是自动舍入。
caught = false;
try
    calculate_avalanche_dcc([0 1 NaN 0],50,.001,o);
catch err
    caught = strcmp(err.identifier,'AvalancheDCC:InvalidCounts');
end
assert(caught);
results.invalid_input_rejected = true;
results.all_passed = true;
disp(results);
end

% 作用：拼出完整的纯合成事件；每个事件前后都有0，起止边界不删失。
% 输入：durations给出每类事件的持续箱数，是列向量；multiplicity给出对应重复次数。
% 输入：power为1时size=duration，为2时size=duration^2；不是模型仿真或真实实验数据。
% 输出：activity是每箱脉冲计数的列向量，只在测试内使用，不另存实验数据文件。
function activity = make_activity(durations,multiplicity,power)
length_needed = 1+sum(multiplicity.*(durations+1));
activity = zeros(length_needed,1);
position = 2;
for j = 1:numel(durations)
    d = durations(j);
    for k = 1:multiplicity(j)
        activity(position:position+d-1) = d^(power-1);
        position = position+d+1;
    end
end
end
