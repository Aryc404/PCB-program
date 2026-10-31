% 作用：从已经分箱的E细胞原始脉冲总数中划分雪崩，并检查DCC是否可解释。
% 输入：activity记录每个时间箱内全部E细胞的脉冲总数；大小为N_bin×1，也接受行向量。
% 输入：bin_steps是一箱包含的模拟步数，例如50；本函数不会再对activity分箱。
% 输入：dt是一模拟步的秒数，例如0.001秒，不是毫秒。
% 输入：options可只填写想改变的设置；各设置及默认值见下面default_options。
% 输出：analysis保存事件表、两个分布的拟合、平均大小与持续时间的关系，以及DCC。
% 划分：activity严格大于指定百分位才算活跃；等于阈值的箱算间隔。
% 大小：将事件中的原始脉冲数相加，不减阈值，不把同一细胞多次放电压成一次。
% 边界：碰到记录开头或结尾的事件仍保存，但complete=false，不用于任何拟合。
% 拟合：离散双截断幂律用最大似然估计；KS使用完整整数支持上的CDF，不回归直方图。
% 检验：默认1000次参数bootstrap；固定已选xmin/xmax，每次重新估计指数。
% 限制：bootstrap的p值是在选定cutoff条件下的结果，不包含cutoff选择的不确定性。
% 限制：最少事件数、跨度和每种持续时间的样本数是人为QA，不是Ma论文给定的门槛。
% 解释：dcc/raw_dcc数学上能算就保留；valid=false时只是诊断数，不能据此判定临界。
% 解释：本函数不以0.2为优化目标，也不自动将任何网络标成临界或非临界。
function analysis = calculate_avalanche_dcc(activity,bin_steps,dt,options)
if nargin < 4 || isempty(options), options = struct(); end
options = parse_options(options); % 合并设置并检查，避免把拼错的选项静默忽略。
validateattributes(activity,{'numeric','logical'},{'real','2d'},mfilename,'activity');
if ~isempty(activity) && ~isvector(activity)
    error('AvalancheDCC:InvalidCounts','activity must be one vector of binned spike counts.');
end
validateattributes(bin_steps,{'numeric'},{'real','finite','scalar','integer','positive'});
validateattributes(dt,{'numeric'},{'real','finite','scalar','positive'});
activity = double(activity(:));
if any(~isfinite(activity) | activity < 0 | activity ~= floor(activity))
    error('AvalancheDCC:InvalidCounts','activity must contain finite nonnegative integer spike counts.');
end

analysis = struct('tau',NaN,'alpha',NaN,'beta_fit',NaN, ...
    'beta_predicted',NaN,'dcc',NaN,'raw_dcc',NaN,'valid',false, ...
    'reason','','threshold',NaN,'active',false(size(activity)), ...
    'events',empty_events(),'options',options);
analysis.bin_steps = bin_steps;
analysis.dt_seconds = dt;
analysis.bin_duration_ms = bin_steps*dt*1000;
analysis.n_bins = numel(activity);
analysis.n_complete_events = 0;
analysis.fit_size = empty_fit();
analysis.fit_duration = empty_fit();
analysis.scaling = empty_scaling();
analysis.interpretation = ['DCC is diagnostic unless valid=true. A valid DCC alone does not establish ' ...
    'criticality; no universal threshold is applied here.'];
analysis.uncertainty_note = ['Distribution exponent intervals are conditional fixed-cutoff percentile bootstrap intervals. ' ...
    'Paired size-duration bootstrap uncertainty for beta and DCC is not estimated in this version; ' ...
    'the reported DCC is a point estimate, not a confidence statement.'];
if isempty(activity)
    analysis.reason = '没有时间箱，不能划分雪崩。';
    return;
end

% percentile_value：按MATLAB默认prctile的中点位置定义百分位，不依赖额外工具箱。
analysis.threshold = percentile_value(activity,options.threshold_percentile);
analysis.active = activity > analysis.threshold;
analysis.active_fraction = mean(analysis.active);
analysis.threshold_tie_fraction = mean(activity == analysis.threshold);
starts = find(diff([false;analysis.active]) == 1);
ends = find(diff([analysis.active;false]) == -1);
prefix = [0;cumsum(activity)];
sizes = prefix(ends+1)-prefix(starts);
durations = ends-starts+1;
complete = starts > 1 & ends < numel(activity);
analysis.events = table(starts,ends,sizes,durations, ...
    durations*analysis.bin_duration_ms,complete, ...
    'VariableNames',{'start_bin','end_bin','size_spikes', ...
    'duration_bins','duration_ms','complete'});
analysis.n_complete_events = nnz(complete);
if ~any(complete)
    analysis.reason = '没有两侧都被间隔包围的完整事件；未完成的边界事件没有参与拟合。';
    return;
end

saved_rng = rng;
rng_cleanup = onCleanup(@() rng(saved_rng)); %#ok<NASGU>
if options.bootstrap_reps > 0, rng(options.bootstrap_seed,'twister'); end

% fit_distribution：选择足够宽且保留足够事件的区间，再做MLE与条件bootstrap。
analysis.fit_size = fit_distribution(sizes(complete),options);
analysis.fit_duration = fit_distribution(durations(complete),options);
analysis.tau = analysis.fit_size.exponent;
analysis.alpha = analysis.fit_duration.exponent;

% fit_scaling：仅按duration拟合范围取事件；不再按size拟合范围裁剪，避免扭曲条件均值。
analysis.scaling = fit_scaling(durations(complete),sizes(complete), ...
    analysis.fit_duration,options);
analysis.beta_fit = analysis.scaling.beta_fit;
if isfinite(analysis.tau) && isfinite(analysis.alpha) ...
        && abs(analysis.tau-1) > options.relation_denominator_tolerance
    analysis.beta_predicted = (analysis.alpha-1)/(analysis.tau-1);
end
if isfinite(analysis.beta_fit) && isfinite(analysis.beta_predicted)
    analysis.dcc = abs(analysis.beta_fit-analysis.beta_predicted);
    analysis.raw_dcc = analysis.dcc;
end

reasons = {};
if analysis.n_complete_events < options.min_events
    reasons{end+1} = sprintf('完整事件只有%d个，少于人为QA要求%d个。', ...
        analysis.n_complete_events,options.min_events); %#ok<AGROW>
end
if ~analysis.fit_size.valid
    reasons{end+1} = ['size拟合：' analysis.fit_size.reason]; %#ok<AGROW>
end
if ~analysis.fit_duration.valid
    reasons{end+1} = ['duration拟合：' analysis.fit_duration.reason]; %#ok<AGROW>
end
if ~analysis.scaling.valid
    reasons{end+1} = ['平均size-duration关系：' analysis.scaling.reason]; %#ok<AGROW>
end
if options.require_exponents_above_one && ...
        ~(analysis.tau > 1+options.relation_denominator_tolerance ...
        && analysis.alpha > 1+options.relation_denominator_tolerance)
    reasons{end+1} = '指数没有都大于1；不将该结果当作通常雪崩标度关系的可靠检验。'; %#ok<AGROW>
end
if ~isfinite(analysis.dcc)
    reasons{end+1} = '指数或标度斜率不可估计，或tau过于接近1。'; %#ok<AGROW>
end
analysis.valid = isempty(reasons);
if analysis.valid
    analysis.reason = '当前QA和条件幂律拟合检验通过；这不等于已经证明临界性。';
else
    analysis.reason = strjoin(reasons,' ');
end
end

% 作用：列出默认设置；所有样本数和拟合质量限制均为本次分析的QA选择。
% 输入：无。
% 输出：o提供统计设置；bootstrap_reps可在单元测试中临时调低，正式分析默认1000。
function o = default_options()
o = struct();
o.threshold_percentile = 35;
o.bootstrap_reps = 1000;
o.bootstrap_seed = 20261023;
o.gof_alpha = 0.05;
o.min_events = 100;
o.min_fit_events = []; % 空值表示沿用min_events。
o.min_unique = 6;
o.min_fit_fraction = 0.70;
o.min_support_ratio = 3;
o.max_lower_tail_fraction = 0.20;
o.max_upper_tail_fraction = 0.10;
o.max_candidate_cutoffs = 12;
o.max_support_points = 200000; % 计算保护：不是神经元或雪崩的生物学上限。
o.exponent_bounds = [0.05 8];
o.min_events_per_duration = 5;
o.min_scaling_durations = 4;
o.min_scaling_events = []; % 空值表示沿用min_events。
o.min_scaling_r_squared = 0.80;
o.relation_denominator_tolerance = 1e-6;
o.require_exponents_above_one = true;
end

% 作用：把调用者填写的设置补全，并检查取值；拼错字段会明确报错。
% 输入：provided提供想覆盖的设置，是一个结构体。
% 输出：o是实际使用的完整设置。
function o = parse_options(provided)
if ~isstruct(provided) || ~isscalar(provided)
    error('AvalancheDCC:InvalidOptions','options must be one scalar struct.');
end
o = default_options();
names = fieldnames(provided);
for j = 1:numel(names)
    if ~isfield(o,names{j})
        error('AvalancheDCC:UnknownOption','Unknown option: %s',names{j});
    end
    o.(names{j}) = provided.(names{j});
end
if isempty(o.min_fit_events), o.min_fit_events = o.min_events; end
if isempty(o.min_scaling_events), o.min_scaling_events = o.min_events; end
integer_fields = {'bootstrap_reps','bootstrap_seed','min_events','min_fit_events', ...
    'min_unique','max_candidate_cutoffs','max_support_points', ...
    'min_events_per_duration','min_scaling_durations','min_scaling_events'};
for j = 1:numel(integer_fields)
    v = o.(integer_fields{j});
    validateattributes(v,{'numeric'},{'real','finite','scalar','integer','nonnegative'});
end
if o.min_events < 2 || o.min_fit_events < 2 || o.min_unique < 2 ...
        || o.max_candidate_cutoffs < 2 || o.max_support_points < 2 ...
        || o.min_events_per_duration < 1 || o.min_scaling_durations < 2 ...
        || o.min_scaling_events < 2
    error('AvalancheDCC:InvalidOptions','Sample-count and candidate limits are too small.');
end
validateattributes(o.threshold_percentile,{'numeric'},{'real','finite','scalar','>=',0,'<=',100});
validateattributes(o.gof_alpha,{'numeric'},{'real','finite','scalar','>',0,'<',1});
validateattributes(o.min_fit_fraction,{'numeric'},{'real','finite','scalar','>',0,'<=',1});
validateattributes(o.min_support_ratio,{'numeric'},{'real','finite','scalar','>',1});
validateattributes(o.max_lower_tail_fraction,{'numeric'},{'real','finite','scalar','>=',0,'<',1});
validateattributes(o.max_upper_tail_fraction,{'numeric'},{'real','finite','scalar','>=',0,'<',1});
validateattributes(o.min_scaling_r_squared,{'numeric'},{'real','finite','scalar','>=',0,'<=',1});
validateattributes(o.relation_denominator_tolerance,{'numeric'},{'real','finite','scalar','positive'});
validateattributes(o.exponent_bounds,{'numeric'},{'real','finite','vector','numel',2});
if o.exponent_bounds(1) >= o.exponent_bounds(2)
    error('AvalancheDCC:InvalidOptions','exponent_bounds must be increasing.');
end
validateattributes(o.require_exponents_above_one,{'numeric','logical'},{'scalar','binary'});
end

% 作用：计算百分位，采用样本位于(0.5/N,1.5/N,...)的中点插值定义。
% 输入：x提供观测值，是一个向量；p是0到100之间的百分位。
% 输出：q是阈值；不对离散重复值加噪声，也不强行保证精确的活跃比例。
function q = percentile_value(x,p)
x = sort(x(:));
position = min(max(numel(x)*p/100+0.5,1),numel(x));
lo = floor(position); hi = ceil(position);
q = x(lo)+(position-lo)*(x(hi)-x(lo));
end

% 作用：先限制候选cutoff的裁剪幅度，再从合格区间中选KS最小者。
% 输入：x提供完整雪崩的size或duration，是正整数向量；o提供QA及bootstrap设置。
% 输出：fit保存指数、区间、KS、p值及有效性；找不到合格区间时仅保留全范围诊断。
function fit = fit_distribution(x,o)
x = sort(double(x(:)));
fit = empty_fit();
fit.n_total = numel(x);
if numel(unique(x)) < 2
    fit.reason = '只有一种观测值，无法识别双截断幂律指数。';
    return;
end
lower_limit = x(max(1,ceil(o.max_lower_tail_fraction*numel(x))));
upper_limit = x(max(1,ceil((1-o.max_upper_tail_fraction)*numel(x))));
lowers = thin_candidates(unique(x(x <= lower_limit)),o.max_candidate_cutoffs);
uppers = thin_candidates(unique(x(x >= upper_limit)),o.max_candidate_cutoffs);
best = [];
candidate_count = 0;
for a = 1:numel(lowers)
    for b = 1:numel(uppers)
        lo = lowers(a); hi = uppers(b);
        inside = x >= lo & x <= hi;
        n = nnz(inside);
        if hi <= lo || hi-lo+1 > o.max_support_points ...
                || n < o.min_fit_events || n/numel(x) < o.min_fit_fraction ...
                || numel(unique(x(inside))) < o.min_unique ...
                || hi/lo < o.min_support_ratio
            continue;
        end
        candidate = mle_ks(x(inside),lo,hi,o.exponent_bounds);
        candidate_count = candidate_count+1;
        if isempty(best) || candidate.ks < best.ks-1e-12 ...
                || (abs(candidate.ks-best.ks) <= 1e-12 && candidate.n > best.n)
            best = candidate;
        end
    end
end
fit.candidate_count = candidate_count;
fit.n_total = numel(x);
if isempty(best)
    fit.xmin = min(x); fit.xmax = max(x); fit.n = numel(x);
    fit.unique = numel(unique(x)); fit.fit_fraction = 1;
    fit.support_ratio = fit.xmax/fit.xmin;
    if fit.xmax-fit.xmin+1 <= o.max_support_points
        diagnostic = mle_ks(x,fit.xmin,fit.xmax,o.exponent_bounds);
        fit.exponent = diagnostic.exponent;
        fit.ks = diagnostic.ks;
    end
    fit.reason = ['没有区间同时达到人为QA的样本数、事件保留率、不同取值数和跨度要求。' ...
        '如有指数，它仅为全范围诊断；没有为凑幂律而继续缩小区间。'];
    return;
end
fields = {'exponent','xmin','xmax','n','ks'};
for j = 1:numel(fields), fit.(fields{j}) = best.(fields{j}); end
selected = x(x >= fit.xmin & x <= fit.xmax);
fit.unique = numel(unique(selected));
fit.fit_fraction = fit.n/numel(x);
fit.support_ratio = fit.xmax/fit.xmin;
fit.range_selection = ['Minimum KS among a restricted grid of observed cutoffs; ' ...
    'retention, span and distinct-value limits apply.'];
margin = 1e-5*max(1,diff(o.exponent_bounds));
if fit.exponent <= o.exponent_bounds(1)+margin ...
        || fit.exponent >= o.exponent_bounds(2)-margin
    fit.reason = 'MLE指数落在数值搜索边界附近；未将这个边界解当作可靠估计。';
    return;
end
if o.bootstrap_reps == 0
    fit.reason = 'bootstrap_reps=0，未进行幂律拟合检验；指数只用于诊断。';
    return;
end
support = (fit.xmin:fit.xmax)';
probability = powerlaw_probability(log(support),fit.exponent);
cdf = cumsum(probability); cdf(end) = 1;
fit.bootstrap_ks = nan(o.bootstrap_reps,1);
fit.bootstrap_exponents = nan(o.bootstrap_reps,1);
for j = 1:o.bootstrap_reps
    % draw_discrete：按双截断分布抽样；cutoff固定，但每次都会重新估计指数。
    simulated = support(draw_discrete(cdf,fit.n));
    surrogate = mle_ks(simulated,fit.xmin,fit.xmax,o.exponent_bounds);
    fit.bootstrap_ks(j) = surrogate.ks;
    fit.bootstrap_exponents(j) = surrogate.exponent;
end
fit.exponent_ci95 = [percentile_value(fit.bootstrap_exponents,2.5), ...
    percentile_value(fit.bootstrap_exponents,97.5)];
fit.p_value = (1+nnz(fit.bootstrap_ks >= fit.ks))/(o.bootstrap_reps+1);
fit.valid = fit.p_value >= o.gof_alpha;
if fit.valid
    fit.reason = '所选cutoff条件下，未拒绝离散双截断幂律；不等于证明幂律或临界性。';
else
    fit.reason = sprintf('所选cutoff条件下，bootstrap p=%.4g，拒绝幂律拟合。',fit.p_value);
end
end

% 作用：限制自动搜索的候选数量，但始终保留该侧允许区间的两个端点。
% 输入：x提供已排序且不重复的候选cutoff，是一个向量；limit为候选数量上限。
% 输出：values是实际搜索的候选值。
function values = thin_candidates(x,limit)
if numel(x) > limit, x = x(unique(round(linspace(1,numel(x),limit)))); end
values = x(:);
end

% 作用：在给定整数区间内求MLE，并比较经验CDF与模型CDF得到离散KS。
% 输入：x是区间内的正整数观测；xmin/xmax是固定cutoff；bounds是指数数值搜索界限。
% 输出：result保存指数和KS等；不对log直方图做最小二乘回归。
function result = mle_ks(x,xmin,xmax,bounds)
log_support = log((xmin:xmax)');
mean_log_x = mean(log(x));
score = @(a) sum(log_support.*powerlaw_probability(log_support,a))-mean_log_x;
if score(bounds(1)) <= 0
    exponent = bounds(1);
elseif score(bounds(2)) >= 0
    exponent = bounds(2);
else
    exponent = fzero(score,bounds,optimset('TolX',1e-7,'Display','off'));
end
probability = powerlaw_probability(log_support,exponent);
counts = accumarray(x(:)-xmin+1,1,[xmax-xmin+1,1]);
empirical_cdf = cumsum(counts)/numel(x);
ks = max(abs(empirical_cdf-cumsum(probability)));
result = struct('exponent',exponent,'xmin',xmin,'xmax',xmax,'n',numel(x),'ks',ks);
end

% 作用：用稳定的数值归一化计算整数支持上的幂律概率，防止上下溢出。
% 输入：log_support给出各整数取值的对数，是列向量；exponent为幂律指数。
% 输出：probability是同样长度的概率列向量，总和为1。
function probability = powerlaw_probability(log_support,exponent)
log_weights = -exponent*log_support;
weights = exp(log_weights-max(log_weights));
probability = weights/sum(weights);
end

% 作用：用二分反CDF抽样；CDF因浮点舍入出现重复值时也不会报分箱错误。
% 输入：cdf是从小到大取值的累计概率列向量；n为需要的样本数。
% 输出：indices给出抽到的整数支持位置，是n×1列向量。
function indices = draw_discrete(cdf,n)
u = rand(n,1); lo = ones(n,1); hi = repmat(numel(cdf),n,1);
while any(lo < hi)
    mid = floor((lo+hi)/2);
    move_right = cdf(mid) < u;
    lo(move_right) = mid(move_right)+1;
    hi(~move_right) = mid(~move_right);
end
indices = lo;
end

% 作用：按持续时间算平均size，再拟合log(mean size)=intercept+beta*log(duration)。
% 输入：duration提供完整事件的持续箱数；sizes提供同批事件的原始脉冲数，均为列向量。
% 输入：duration_fit提供允许的持续时间区间；o提供每个持续时间所需事件数等QA。
% 输出：scaling记录全部区间内条件均值、计数和实际拟合标记；size没有被size cutoff裁剪。
function scaling = fit_scaling(duration,sizes,duration_fit,o)
scaling = empty_scaling();
if ~isfinite(duration_fit.xmin) || ~isfinite(duration_fit.xmax)
    scaling.reason = '没有可用的duration区间。'; return;
end
keep = duration >= duration_fit.xmin & duration <= duration_fit.xmax;
duration = duration(keep); sizes = sizes(keep);
if isempty(duration), scaling.reason = 'duration区间内没有完整事件。'; return; end
[values,~,group] = unique(duration);
scaling.duration_bins = values;
scaling.mean_size = accumarray(group,sizes,[],@mean);
scaling.count = accumarray(group,1);
scaling.use_for_fit = scaling.count >= o.min_events_per_duration;
scaling.fit_duration_range = [duration_fit.xmin duration_fit.xmax];
scaling.n_events = sum(scaling.count(scaling.use_for_fit));
mask = scaling.use_for_fit;
if nnz(mask) < 2
    scaling.reason = '达到每种duration最少事件要求的持续时间类别不足2种。'; return;
end
log_d = log(values(mask)); log_s = log(scaling.mean_size(mask));
coefficients = polyfit(log_d,log_s,1);
scaling.beta_fit = coefficients(1); scaling.intercept = coefficients(2);
residual = log_s-polyval(coefficients,log_d);
total = sum((log_s-mean(log_s)).^2);
if total > eps
    scaling.r_squared = 1-sum(residual.^2)/total;
end
reasons = {};
if nnz(mask) < o.min_scaling_durations
    reasons{end+1} = '有足够事件的不同duration数量未达到人为QA。'; %#ok<AGROW>
end
if scaling.n_events < o.min_scaling_events
    reasons{end+1} = '用于条件均值回归的事件总数未达到人为QA。'; %#ok<AGROW>
end
if ~isfinite(scaling.r_squared) || scaling.r_squared < o.min_scaling_r_squared
    reasons{end+1} = 'log均值与log持续时间的直线拟合未达到人为R平方要求。'; %#ok<AGROW>
end
scaling.valid = isempty(reasons);
if scaling.valid
    scaling.reason = '条件均值回归通过当前QA；斜率并非直方图分布指数。';
else
    scaling.reason = strjoin(reasons,' ');
end
end

% 作用：建立空事件表，使无事件时也保留相同字段和数据类型。
% 输入：无。
% 输出：events是0行事件表。
function events = empty_events()
z = zeros(0,1);
events = table(z,z,z,z,z,false(0,1),'VariableNames', ...
    {'start_bin','end_bin','size_spikes','duration_bins','duration_ms','complete'});
end

% 作用：建立尚未估计的分布拟合结果，避免失败时缺少字段。
% 输入：无。
% 输出：fit包含指数、区间、样本数、KS、条件bootstrap说明及失败原因。
function fit = empty_fit()
fit = struct('exponent',NaN,'xmin',NaN,'xmax',NaN,'n',0,'ks',NaN, ...
    'p_value',NaN,'valid',false,'reason','','unique',0,'fit_fraction',NaN, ...
    'support_ratio',NaN,'n_total',0,'candidate_count',0,'range_selection','', ...
    'bootstrap_ks',zeros(0,1),'bootstrap_exponents',zeros(0,1),'exponent_ci95',[NaN NaN], ...
    'ci_method','Percentile 95% interval conditional on fixed selected cutoffs; not a joint DCC interval.', ...
    'bootstrap_method', ...
    ['Parametric bootstrap conditional on selected xmin/xmax; exponent refitted in every replicate; ' ...
    'p=(1+count(KS_boot>=KS_data))/(B+1); cutoff selection uncertainty is not included.']);
end

% 作用：建立尚未估计的平均size-duration关系结果。
% 输入：无。
% 输出：scaling保存条件均值、计数、拟合标记和斜率，回归对数采用自然对数。
function scaling = empty_scaling()
scaling = struct('duration_bins',zeros(0,1),'mean_size',zeros(0,1), ...
    'count',zeros(0,1),'use_for_fit',false(0,1),'beta_fit',NaN, ...
    'intercept',NaN,'r_squared',NaN,'n_events',0,'fit_duration_range',[NaN NaN], ...
    'valid',false,'reason','');
end
