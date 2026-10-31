% 作用：让训练好的网络在给定LGN放电率下连续运行，保存逐毫秒脉冲。
% 输入：state是固定权重和阈值；rates是每段LGN放电率，大小为512×记录分块数。
% 输入：cfg.noise_sigma单独控制原模型的背景噪声，0关闭、1沿用原强度。
% 输出：recording包含E/I脉冲、逐步E/I/LGN总脉冲数、起止电位和随机数状态。
% natural提供图像率；blank提供全零率；noise提供恒定率。这里不额外施加起始脉冲。
% 正常放电后仍把该细胞膜电位复位为0；只有换patch时不再把全网膜电位清零。

function recording = simulate_frozen_ad_stream(state,rates,cfg)
nE = numel(state.theta_E);
nI = numel(state.theta_I);
nX = size(state.W_EX,2);
nPatches = ceil(cfg.total_steps/cfg.patch_steps);
assert(size(rates,1)==nX && size(rates,2)>=nPatches, ...
    'PCB:InputSize','Input rates do not cover the requested stream.');
assert(all(isfinite(rates(:))) && all(rates(:)>=0) && ...
    all(rates(:)*cfg.dt<=1),'PCB:InputProbability','Invalid LGN probabilities.');
V_E = zeros(nE,1);
V_I = zeros(nI,1);
exp_m = exp(-cfg.dt/cfg.tau_m);
neuronsE = cell(nPatches,1); timesE = cell(nPatches,1);
neuronsI = cell(nPatches,1); timesI = cell(nPatches,1);
step_count_E = zeros(cfg.total_steps,1,'uint16');
step_count_I = zeros(cfg.total_steps,1,'uint16');
step_count_X = zeros(cfg.total_steps,1,'uint16');
started = tic;
for patch = 1:nPatches
    first_step = (patch-1)*cfg.patch_steps+1;
    last_step = min(patch*cfg.patch_steps,cfg.total_steps);
    steps_here = last_step-first_step+1;
    blockE = false(nE,steps_here);
    blockI = false(nI,steps_here);
    R_XDT = rates(:,patch)*cfg.dt;
    for local_step = 1:steps_here
        step = first_step+local_step-1;
        S_X = rand(nX,1)<R_XDT;
        step_count_X(step) = sum(S_X);
        S_E = V_E>state.theta_E;
        S_I = V_I>state.theta_I;
        V_E(S_E) = 0;
        V_I(S_I) = 0;
        blockE(:,local_step) = S_E;
        blockI(:,local_step) = S_I;
        step_count_E(step) = sum(S_E);
        step_count_I(step) = sum(S_I);

        % 六类连接都保留。W_EI是I到E的非负权重，抑制作用由减号表达。
        V_E = V_E*exp_m + state.W_EX*S_X + state.W_EE*S_E ...
            - state.W_EI*S_I + cfg.noise_sigma*randn(nE,1);
        V_I = V_I*exp_m + state.W_IX*S_X + state.W_IE*S_E ...
            - state.W_II*S_I + cfg.noise_sigma*randn(nI,1);
        V_E(V_E<cfg.V_min) = cfg.V_min;
        V_I(V_I<cfg.V_min) = cfg.V_min;
    end
    [neuronsE{patch},local_times] = find(blockE);
    timesE{patch} = local_times+first_step-1;
    [neuronsI{patch},local_times] = find(blockI);
    timesI{patch} = local_times+first_step-1;
    if cfg.progress_every_patches>0 && ...
            (mod(patch,cfg.progress_every_patches)==0 || patch==nPatches)
        fprintf('Input block %d/%d | step %d/%d | elapsed %.1f s\n', ...
            patch,nPatches,last_step,cfg.total_steps,toc(started));
    end
end
e_rows=vertcat(neuronsE{:}); e_times=vertcat(timesE{:});
i_rows=vertcat(neuronsI{:}); i_times=vertcat(timesI{:});
recording.spikes_E = sparse(e_rows,e_times,true(size(e_rows)),nE,cfg.total_steps);
recording.spikes_I = sparse(i_rows,i_times,true(size(i_rows)),nI,cfg.total_steps);
recording.step_count_E = step_count_E;
recording.step_count_I = step_count_I;
recording.step_count_X = step_count_X;
recording.final_V_E = V_E;
recording.final_V_I = V_I;
recording.initial_V_E = zeros(nE,1);
recording.initial_V_I = zeros(nI,1);
recording.final_rng_state = rng;
recording.elapsed_seconds = toc(started);
end
