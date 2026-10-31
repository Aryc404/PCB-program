% 作用：用极小的合成网络检查时间顺序、换图连续性、正常spike复位和冻结性。
% 输入：无。输出：无；任何检查失败都会报错。不会读取或改动真实训练结果。
function test_frozen_ad_stream()
old_rng=rng; cleanup=onCleanup(@() rng(old_rng));
state=struct('W_EX',1,'W_EE',0,'W_EI',0,'W_IX',0,'W_IE',0, ...
    'W_II',0,'theta_E',6,'theta_I',100);
before=state;
cfg=struct('dt',.001,'tau_m',1e6,'V_min',-10,'noise_sigma',0, ...
    'total_steps',8,'patch_steps',4,'progress_every_patches',0);
rng(71,'twister');
r=simulate_frozen_ad_stream(state,[1000 1000],cfg);
assert(nnz(r.spikes_E)==1 && r.spikes_E(1,8), ...
    'Continuity failed: resetting at step 5 would suppress the expected spike at step 8.');
assert(r.final_V_E>0 && r.final_V_E<2,'Normal spike reset was lost.');
assert(nnz(r.spikes_I)==0 && all(r.step_count_I==0),'Unexpected I spikes.');
assert(isequaln(state,before),'Input state was mutated.');

% 随机小网络逐步对照原simulate_response的公式，比较整个spike矩阵而非只比较均值。
rng(31,'twister'); nE=4; nI=2; nX=6;
s=struct('W_EX',rand(nE,nX),'W_EE',rand(nE,nE), ...
    'W_EI',rand(nE,nI),'W_IX',rand(nI,nX),'W_IE',rand(nI,nE), ...
    'W_II',rand(nI,nI),'theta_E',ones(nE,1)*2,'theta_I',ones(nI,1)*2);
cfg.total_steps=80; cfg.patch_steps=40; cfg.tau_m=.01; cfg.noise_sigma=1;
rates=rand(nX,2)*100;
rng(82,'twister'); actual=simulate_frozen_ad_stream(s,rates,cfg);
rng(82,'twister'); [expectedE,expectedI,vE,vI]=native_reference(s,rates,cfg);
assert(isequal(actual.spikes_E,sparse(expectedE)) && ...
    isequal(actual.spikes_I,sparse(expectedI)),'LIF spike histories disagree.');
assert(isequal(actual.final_V_E,vE) && isequal(actual.final_V_I,vI), ...
    'Final membrane voltages disagree.');
assert(isequal(double(actual.step_count_E),sum(expectedE,1)'), ...
    'Per-step count does not match spikes.');
fprintf('test_frozen_ad_stream passed.\n');
end

% 作用：独立逐步实现原冻结LIF公式，供测试对照；输入网络、图片率和设置，输出spikes和末尾V。
function [outE,outI,V_E,V_I]=native_reference(state,rates,cfg)
nE=numel(state.theta_E); nI=numel(state.theta_I); nX=size(rates,1);
V_E=zeros(nE,1); V_I=zeros(nI,1);
outE=false(nE,cfg.total_steps); outI=false(nI,cfg.total_steps);
exp_m=exp(-cfg.dt/cfg.tau_m);
for t=1:cfg.total_steps
    patch=ceil(t/cfg.patch_steps);
    S_X=rand(nX,1)<rates(:,patch)*cfg.dt;
    S_E=V_E>state.theta_E; S_I=V_I>state.theta_I;
    V_E(S_E)=0; V_I(S_I)=0;
    outE(:,t)=S_E; outI(:,t)=S_I;
    V_E=V_E*exp_m+state.W_EX*S_X+state.W_EE*S_E ...
        -state.W_EI*S_I+cfg.noise_sigma*randn(nE,1);
    V_I=V_I*exp_m+state.W_IX*S_X+state.W_IE*S_E ...
        -state.W_II*S_I+cfg.noise_sigma*randn(nI,1);
    V_E(V_E<cfg.V_min)=cfg.V_min; V_I(V_I<cfg.V_min)=cfg.V_min;
end
end
