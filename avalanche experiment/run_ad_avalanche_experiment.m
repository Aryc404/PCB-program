% AD网络雪崩检测入口
% 直接运行本文件。下面几个参数可自行修改；其余默认沿用cn=1的paper模型。
% 自然图像每400步换一张；换图时保留膜电位。每50步统计真实脉冲总数。
experiment_dir = fileparts(mfilename('fullpath'));
addpath(experiment_dir);

options = struct();
options.checkpoint_file = fullfile(experiment_dir,'input', ...
    'pcb_shuffle_ad_paper_cn1_run01_batch1200.mat');

% ===== 在这里输入单词，选择本次输入 =====
% 'natural'：自然图像，沿用原来的16×16 patch输入。
% 'blank'：所有LGN输入为0；不施加起始脉冲，也不自动重新激活网络。
% 'noise'：各LGN通道以固定放电率独立产生随机脉冲，不是白噪声图片。
options.input_mode = 'natural';
options.noise_rate_hz = 10;          % 只在noise下使用：每个LGN通道10Hz，可在0—100Hz内修改。
% 背景噪声与LGN输入是两回事。1保留原模型背景噪声；0关闭它。
% 若要完全没有外部驱动，选择blank并把下面这一项设为0；从零电位开始可能始终静默。
options.background_noise_sigma = 1;

options.total_steps = 200000;
options.bin_steps = 50;
options.patch_steps = 400;
options.patch_seed = 20261023;       % 只在natural下控制抽取哪些自然图像patch。
options.dynamics_seed = 20261023;    % 控制LGN脉冲和原模型高斯背景噪声。
options.threshold_percentile = 35;
options.bootstrap_samples = 1000;
options.make_figures = true;

% 三种输入都只做冻结测试：STDP、L1和自适应阈值不继续更新，E/I回路正常工作。
result = measure_ad_avalanche(options);
