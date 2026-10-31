# AD网络的连续雪崩记录与DCC初步检验

## 运行哪个文件

在MATLAB R2026b中打开并运行 `run_ad_avalanche_experiment.m`。
不需要打开并行池；这里是一条按时间连续推进的网络轨迹，不能把相邻时间段当成独立patch并行后拼接。
原来的训练程序、检查点和结果均不修改。本目录只做训练后的冻结测试。

## 在入口选择输入

只改入口的 `options.input_mode`，每次运行测试一种输入：

- `'natural'`：原来的自然图像输入，是默认值。
- `'blank'`：LGN放电率全部为0；不加起始脉冲，不在安静后自动重新刺激。
- `'noise'`：512个LGN通道分别以固定放电率独立生成随机脉冲；不是空间高斯白噪声图像。

例如：

```matlab
options.input_mode = 'blank';
options.background_noise_sigma = 1;
```

`background_noise_sigma=1`保留原模型每步对E/I细胞加入的背景高斯噪声；设为0才关闭。
因此blank默认表示“没有LGN输入”，并不表示“没有任何随机驱动”。
如果设为blank且背景噪声也为0，网络从零膜电位开始，正阈值的网络可能始终不放电。
这是该测试条件的正常结果，程序不会偷偷补脉冲，也不会把无法计算的DCC填为0。

噪声输入例如：

```matlab
options.input_mode = 'noise';
options.noise_rate_hz = 10;
options.background_noise_sigma = 1;
```

`noise_rate_hz`是每个LGN通道的放电率，允许0—100 Hz。
10 Hz只是可调的起始测试值，不是文献规定，也没有自动匹配自然图像的平均输入强度。
每1 ms生成一次独立Bernoulli脉冲，概率为放电率×0.001；不是精确连续时间Poisson过程。
512个通道都设为10 Hz时，平均总LGN输入率为5120 spikes/s，而不是每秒总共10个输入脉冲。
background_noise_sigma控制的背景扰动仍然单独存在，不能与LGN噪声混为一谈。

三种输入都使用同一AD最终网络，六类连接正常传递，权重和阈值均冻结。
没有增加重新训练、初始kick或继续开启学习的分支。
blank/noise不需要读取自然图像数据或图像patch生成程序。

## 本次固定的流程

1. 读取 `input/pcb_shuffle_ad_paper_cn1_run01_batch1200.mat`。
   加载时核对AD、cn=1、训练终点1200批，以及六类权重和E/I阈值的尺寸与数值。
   本次选用run01，训练seed为20250718；不是对多个seed求平均。
2. natural使用项目根目录已有的 `VanHateren_DoG_small.mat` 和 `generate_image_patches.m`。
   随机抽取16×16自然图像patch，沿用原程序的随机90度旋转、×70、ON/OFF分离和100 Hz上限。
   不重新做DoG，不重标定输入强度，不新增白噪声刺激。
   blank使用全零LGN率，noise使用入口指定的恒定LGN率；二者不抽图。
3. natural一次只呈现一个patch，400步后换下一个。正式设置共200000步，即500张patch。
   步长仍是1 ms，所以记录时长200秒；不是把积分步长改为50 ms。
   blank/noise的400步只用于存储分块，块与块之间不更换刺激，也不清空膜电位。
4. 保留六类连接、原来的泄漏和膜电位下限，背景高斯噪声按入口设置。
   放电后仍将该细胞膜电位复位为0；换图不把全网膜电位清零。
   不执行STDP、L1更新和自适应阈值更新。
5. 每50步统计一个bin，得到4000个bin。
   每个E神经元在bin内放几次就计几次；再跨全部400个E细胞求和，得到network activity。
   同时记录I脉冲，但本版雪崩与DCC只使用E群体。
6. 用整段E群体activity的第35百分位作阈值。
   严格大于阈值的连续bin组成一次事件；等于阈值的bin属于间隔。
   Size是事件区间内全部原始E脉冲之和，不减阈值；Duration是连续bin数量乘50 ms。
   首尾碰到记录边界的事件仍保存，但标记complete=false，不参与拟合。
7. 分别拟合Size和Duration的离散双截断幂律，得到tau和alpha。
   指数使用最大似然估计，不使用双对数直方图直线回归。
8. 按Duration分组计算平均Size，再拟合 log(mean Size) 对 log(Duration)，得到beta_fit。
   这里不会用Size拟合区间再次裁剪事件，以免人为改变条件均值。
9. beta_predicted=(alpha-1)/(tau-1)，DCC=abs(beta_fit-beta_predicted)。

## 关于初始化和原模型

训练检查点包含权重、阈值及训练RNG，但没有训练末尾的膜电位。
因此本版从V_E=0、V_I=0开始一条新的冻结测试轨迹，不能称作原训练末尾的精确续接。
按本次要求记录全部200000步，burn-in=0，没有偷偷增加或删除预热时间。
初始瞬态可能影响统计，原始1 ms脉冲已保留，后续可单独检查，不需要再跑网络。

测试的patch_seed与dynamics_seed在入口分开设置，避免改抽图方式时不知不觉改变动力学随机流。
程序也会保存全部实际使用的输入率矩阵、冻结网络和逐步脉冲，便于复核。

## 拟合范围和有效性

`calculate_avalanche_dcc.m` 的default_options集中列出了本版统计设置。
自动搜索cutoff时，最多去掉低端20%和高端10%的观测，保留至少70%的事件。
每侧最多12个候选cutoff，只在满足下列条件的区间中选KS最小者：

- 拟合事件至少100个。
- 至少6种不同观测值。
- xmax/xmin至少为3；这只是很基础的范围检查，不代表已有宽广标度区间。
- beta回归每种Duration至少5个事件、至少4种Duration、合计至少100个事件。
- beta回归的log-log R平方至少0.8。

以上都是本版为了避免极少样本/极窄区间产生假确定性的QA选择，**不是Ma等论文规定的统一门槛**。
QA未通过时，程序可能仍保存数学上能算出的指数和raw_dcc，但valid=false，并逐项说明原因。
无法估计时返回NaN，不填0，也不通过继续收窄区间去凑幂律。

正式设置每个边缘分布做1000次参数bootstrap，每次重新估计指数。
KS的p值为 `(1 + KS_boot>=KS_data的次数)/(B+1)`，p<0.05表示拒绝当前幂律拟合。
bootstrap固定已选cutoff，得到的是**给定拟合区间条件下**的p值和指数区间，没有包含cutoff选择的不确定性。
没有估计beta或DCC的配对bootstrap置信区间；当前DCC是点估计。

## 不能只凭DCC小就宣告临界

本版是借鉴Ma等2019的标度关系检验，不是逐行复现：

- 按你的要求使用原始spike总数，以及50 ms分箱；Ma原文默认40 ms且二值化表述另有口径问题。
- 200秒仅4000个bin，不等于4000个雪崩，也不等于原文4小时记录窗的统计精度。
- 自然图像每400 ms变化一次，可能带来8-bin输入节奏。程序记录换图时间、LGN驱动力和各patch相位上的平均活动/事件边界数量，不能把输入驱动的结构直接归因于复发传播。
- 在非零阈值下累计全部原始脉冲，Size包含随Duration增长的阈值基线贡献。这是明确选择的计数口径，但也可能影响size-duration斜率；不能用低DCC单独排除这种影响。
- 0.2仅作为此前讨论的参考数值，不参与拟合、参数调节或筛选图片。
- valid=true只表示当前QA和条件拟合检验通过，不代表已经证明临界。p>=0.05也不是“网络临界的概率”。

## 输出

每次运行新建 `results/AD_cn1_输入单词_日期_时间/`，例如 `AD_cn1_blank_...`，不会覆盖此前运行。
旧的 `AD_cn1_日期_时间` 自然图像结果保留原样。

- `ad_avalanche_result.mat`：完整数据。包含冻结网络、配置、原始E/I脉冲稀疏矩阵、1 ms群体计数、逐细胞逐bin计数、所有输入率、事件和统计诊断。
- `network_activity.csv`：每bin起止时间、E/I脉冲总数、输入率、真实LGN脉冲数和输入类型。只有natural有实际patch编号；blank/noise对应两列为NaN。
- `E_neuron_bin_counts.csv`：400个E细胞各自的逐bin脉冲数；每行一个bin。
- `patch_schedule.csv`：只在natural保存，每张patch的起止步数和ON/OFF平均输入率。
- `input_schedule.csv`：在blank/noise保存输入分块的时间和LGN率；分块不是换图，`is_visual_patch=false`。
- `patch_phase_diagnostics.csv`：只在natural保存，检查8-bin换图节奏；恒定的blank/noise没有这个真实周期，不生成此文件。
- `avalanches.csv`：全部事件的起止bin、Size、Duration、是否完整。
- `fit_diagnostics.csv`：Size/Duration指数、cutoff、样本数、KS、条件bootstrap p及指数区间。
- `dcc_summary.csv`：tau、alpha、两个beta、DCC、有效性与原因；另记输入类型、背景噪声强度、实际平均LGN率和真实LGN脉冲总数。只有noise填写noise_rate_hz，其他模式该列为NaN。
- 四组PNG/FIG：network_activity、avalanche_size、avalanche_duration、size_duration_scaling。白底英文，FIG可编辑。

MAT会在拟合前先保存原始记录；统计或画图出错时可以读取它继续分析，不必重跑网络。
`dcc_below_reference`只表示数值是否小于0.2，必须同时看fit_valid和fit_reason。
每种输入仍使用本次活动的第35百分位，阈值会随输入改变；不同输入的DCC不能脱离事件数、输入强度和拟合质量直接比较。
`recording.step_count_X`保存逐1 ms的真实LGN脉冲总数，blank应全部为0。

## 哪个文件负责哪一步

- `run_ad_avalanche_experiment.m`：入口，只设置本次参数。
- `measure_ad_avalanche.m`：读取、检查、分箱、保存，并调用统计与画图。
- `simulate_frozen_ad_stream.m`：原AD方程的连续冻结仿真，三种输入共用同一套动力学。
- `calculate_avalanche_dcc.m`：事件划分、MLE、KS/bootstrap、beta及DCC。
- `plot_ad_avalanche.m`：根据记录画图，不重新模拟。
- `tests/test_frozen_ad_stream.m`：8/80步合成小网络自检。
- `tests/test_calculate_avalanche_dcc.m`：纯合成统计自检，不生成冒充真实实验的CSV。
- `tests/test_avalanche_input_modes.m`：用合成检查点验证blank/noise、背景噪声、重复性及错误输入，不读取真实实验数据。

如需重做这些自检，在本目录运行：

```matlab
addpath(pwd);
addpath(fullfile(pwd,'tests'));
test_frozen_ad_stream;
test_calculate_avalanche_dcc;
test_avalanche_input_modes;
```

这些自检只使用合成数据，不会启动正式的200000步记录。

## 不重跑网络，单独重算统计

在本目录执行以下命令，替换MAT文件路径即可：

```matlab
saved = load('这里填已保存的ad_avalanche_result.mat路径','result');
options = struct('threshold_percentile',35,'bootstrap_reps',1000);
analysis = calculate_avalanche_dcc(saved.result.activity_E, ...
    saved.result.options.bin_steps,saved.result.cfg.dt,options);
```

## 文献口径

- Ma et al., 2019, Neuron, DOI: 10.1016/j.neuron.2019.08.031：雪崩标度关系与DCC。
- Marshall et al., 2016, Frontiers in Physiology, DOI: 10.3389/fphys.2016.00250：幂律和相关统计方法。
- Dalla Porta & Copelli, 2019, PLOS Computational Biology, DOI: 10.1371/journal.pcbi.1006924：Size口径、阈值依赖和更严格的指数关系检验。
