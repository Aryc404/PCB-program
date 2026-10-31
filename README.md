
# The network comprises three types of neurons:

- `X`: LGN neurons.
- `E`: Excitatory neurons.
- `I`: Inhibitory neurons.

# Weight names follow the convention of "target neuron first, source neuron second":

- `state.W_EX`: X to E (LGN neurons connecting to excitatory neurons).
- `state.W_EE`: E to E (connections between excitatory neurons).
- `state.W_EI`: I to E (inhibitory neurons connecting to excitatory neurons).
- `state.W_IX`: X to I (LGN neurons connecting to inhibitory neurons).
- `state.W_IE`: E to I (excitatory neurons connecting to inhibitory neurons).
- `state.W_II`: I to I (connections between inhibitory neurons).

# Network Scale Variables
- `cfg.N_X`: Number of pixels in a single ON or OFF input channel. Since each image patch is 16×16 pixels, the value here is 256.
- `cfg.N_E`: Number of excitatory neurons; the value here is 400.
- `cfg.N_I`: Number of inhibitory neurons; the value here is 100.
- `cfg.N_n`: Total number of training batches; the value here is 1200.
- `cfg.N_q`: Number of batches prior to formal learning during which only firing thresholds are adjusted and weights are not updated; the value here is 100.
- `cfg.N_b`: Number of image patches per batch; the value here is 100.
- `cfg.N_t`: Number of time steps executed during the presentation of each image; the value here is 400.
 
# Time and Neuronal Dynamics Parameters

- `cfg.dt`: Length of a single time step; set to 1 ms in the program.
- `cfg.T`: Total training duration for a natural image patch; set to 400 ms in the program.
- `cfg.tau_m`: Decay time constant for membrane potential or postsynaptic potential; set to 10 ms in the program.
- `cfg.tau_i`: Time constant for the short-term synaptic trace (used for symmetric STDP); set to 20 ms in the program.
- `cfg.tau_l`: Time constant for the long-term synaptic trace (used for triplet STDP); set to 50 ms in the program.
- `exp_m`: Retention ratio of membrane potential per millisecond; set to approximately 0.9048.
- `exp_i`: Retention ratio of the short-term synaptic trace per millisecond; set to approximately 0.9512.
- `exp_l`: Retention ratio of the long-term synaptic trace per millisecond; set to approximately 0.9802.
- `invtau_i`: Scaling factor for the increase in the short-term synaptic trace upon the occurrence of a new spike; set to 50 s⁻¹.
- `invtau_l`: Scaling factor for the increase in the long-term synaptic trace upon the occurrence of a new spike; set to 20 s⁻¹.
- `cfg.V_min`: Lower bound of membrane potential; set to -10 in the program (i.e., membrane potentials below this value are clipped to -10).

At each time step, the membrane potential decays according to `exp_m`, is adjusted by adding excitatory input and subtracting inhibitory input, and is further modified by Gaussian noise with a mean of 0 and a standard deviation of 1.

# Target Firing Rates and Threshold Variables

# Target Firing Rates

- `cfg.rho_E_Hz`: Target firing rate for E neurons. Set to 2 Hz in the program.
- `cfg.rho_I_Hz`: Target firing rate for I neurons. Set to 4 Hz in the program.

# Thresholds

- `cfg.theta_E0`: Initial firing threshold for all E neurons; set to 10 in the program.
- `cfg.theta_I0`: Initial firing threshold for all I neurons; set to 10 in the program.
- `state.theta_E`: Matrix representing the current firing threshold for each E neuron; matrix size: 400×1.
- `state.theta_I`: Matrix representing the current firing threshold for each I neuron; matrix size: 100×1.
- `dtheta_E`: Matrix representing the change in threshold for each E neuron per batch; matrix size: 400×1.
- `dtheta_I`: Matrix representing the change in threshold for each I neuron per batch; matrix size: 100×1.

- `cfg.rho_E*cfg.eta_tE*scale`: Upper limit of the scale of change for E neuron thresholds after a single batch is completed; set to 0.8 in the program.
- `cfg.rho_I*cfg.eta_tI*scale`: The upper bound of the scaling factor for the change in the threshold of neuron I after the completion of a single batch; set to 1.6 in the program.

# Synaptic Learning Rate Variables

- `cfg.eta_EX`: Learning rate for X-to-E weights; initial value: 0.0002.
- `cfg.eta_EE`: Learning rate for E-to-E weights; initial value: 0.0001.
- `cfg.eta_EI`: Learning rate for I-to-E weights; initial value: 0.09.
- `cfg.eta_IX`: Learning rate for X-to-I weights; initial value: 0.003.
- `cfg.eta_IE`: Learning rate for E-to-I weights; initial value: 0.04.
- `cfg.eta_II`: Learning rate for I-to-I weights; initial value: 0.06.
- `cfg.eta_tE`: Learning rate for E-neuron adaptive thresholds; initial value: 1.
- `cfg.eta_tI`: Learning rate for I-neuron adaptive thresholds; initial value: 1.

All the above learning rates are halved after the 400th batch, and halved again after the 800th batch.

# STDP Parameters

- `A_i`: Coefficient for symmetric STDP; value used in the program: 0.5.
- `A_p`: Coefficient for the potentiation term in triplet STDP; value used in the program: 1.
- `A_d`: Coefficient for the depression term in triplet STDP; value used in the program: 2.


# L1 Weight Constraints

These values ​​limit the total weight of a specific type of connection received by each postsynaptic neuron.

- `cfg.L1_EX`: Upper limit on the sum of all X-to-E weights received by a single E neuron; value used in the program: 100.
- `cfg.L1_EE`: Upper limit on the sum of all E-to-E weights received by a single E neuron; value used in the program: 10.
- `cfg.L1_EI`: Upper limit on the sum of all I-to-E weights received by a single E neuron; value used in the program: 120.
- `cfg.L1_IX`: Upper limit on the sum of all X-to-I weights received by a single I neuron; value used in the program: 80.
- `cfg.L1_IE`: Upper limit on the sum of all E-to-I weights received by a single I neuron; value used in the program: 240.
- `cfg.L1_II`: Upper limit on the sum of all I-to-I weights received by a single I neuron; value used in the program: 120.

# Weight Matrices

- `state.W_EX`: Matrix representing synaptic weights from X to E; size: 400×512.
- `state.W_EE`: Matrix representing synaptic weights from E to E; the diagonal is set to 0 because self-connections among E neurons are not permitted; size: 400×400.
- `state.W_EI`: Matrix representing synaptic weights from I to E; size: 400×100.
- `state.W_IX`: Matrix representing synaptic weights from X to I; size: 100×512.
- `state.W_IE`: Matrix representing synaptic weights from E to I; size: 100×400.
- `state.W_II`: Matrix representing synaptic weights from I to I; the diagonal is set to 0 because self-connections among I neurons are not permitted; size: 100×100.

# Weight and Threshold Update Increments

- `stats.dW_EX`: Matrix of accumulated changes in X-to-E synaptic weights for a single batch; size: 400×512.
- `stats.dW_EE`: Matrix of accumulated changes in E-to-E synaptic weights for the current batch; size: 400×400.
- `stats.dW_EI`: Matrix of accumulated changes in I-to-E synaptic weights for the current batch; size: 400×100.
- `stats.dW_IX`: Matrix of accumulated changes in X-to-I synaptic weights for the current batch; size: 100×512.
- `stats.dW_IE`: Matrix of accumulated changes in E-to-I synaptic weights for the current batch; size: 100×400.
- `stats.dW_II`: Matrix of accumulated changes in I-to-I synaptic weights for the current batch; size: 100×100.

# Membrane Potential, Spiking, and Pulse Variables

- `V_E`: Matrix representing the change in membrane potential for 400 E-neurons in every time step during stimulation by 100 images (processed in parallel as a batch); size: 400×100.
- `V_I`: Matrix representing the change in membrane potential for 100 I-neurons in every time step during stimulation by 100 images; size: 100×100.
- `S_X`: Matrix indicating whether input neurons generate a spike within the current 1-millisecond interval (1 for spike, 0 for no spike); size: 512×100.
- `S_E`: Matrix indicating whether E-neurons generate a spike within the current 1-millisecond interval; size: 400×100.
- `S_I`: Matrix indicating whether I-neurons generate a spike within the current 1-millisecond interval; size: 100×100.
- `R_X`: Matrix of ON/OFF input firing rates corresponding to the 100 images, in Hertz. Values ​​exceeding 100 Hz are capped at 100 Hz; size: 512×100.

# Synaptic Trace Variables

Synaptic traces are used to record recent spiking history, ensuring that weight updates depend not only on the current moment but also on activity over a preceding period.

- `U_Xi`: Short-term synaptic trace matrix for X neurons; time constant: 20 ms; size: 512×100.
- `U_Ei`: Short-term synaptic trace matrix for E neurons; time constant: 20 ms; size: 400×100.
- `U_El`: Long-term synaptic trace for E neurons; time constant: 50 ms; size: 400×100.
- `U_Ii`: Short-term synaptic trace for I neurons; time constant: 20 ms; size: 100×100.

=======================================================================


##  Functions of each program:
# main.m
    The original main program used in the Ruslim paper,it is primarily responsible for network initialization, spiking simulation, synaptic learning, and simulating the adaptive threshold mechanism.

# generate_image_patches.m
     extract patches from a set of natural images —already processed by center-surround receptive field filter— to be used for training the network.
     
# VanHateren_DoG_small.mat
    A set of 8 images processed with a center-surround receptive field filter; each image measures 1024 x 1536 pixels.

# run_pcb_shuffle_ei_balance_cn1.m
    The entry program for training ablation experiments with a network size of cn=1.

# run_pcb_shuffle_all.m
    Primary responsibilities include: reading the experimental mode and configuration parameters; determining the operating conditions, network size (cn), and number of repetitions based on the configuration; and checking for existing checkpoints to skip tasks that have already been completed.

# run_pcb_shuffle_experiment.m
    Load condition parameters and execution settings. Invoke the `run_ps_factorial_condition.m` script to translate conditions—such as AD and AE into specific settings for I→E transmission, I→E plasticity, dynamic thresholds, and L1 normalization. Configure network size and random seeds for each condition; finally, review and organize the results, saving them as CSV and MAT files.

# run_ps_factorial_condition.m
    This is the core simulation program executed for a single experimental condition. Modified from the main program of the original paper, it serves as the core routine currently used for shuffle experiments.
    Based on the provided conditions, it:
    - Initializes the network, neuronal states, and synaptic weights.
    - Inputs natural images and simulates the membrane potentials and spiking activity of E/I neurons.
    - Executes STDP, dynamic threshold adjustment, and L1 weight normalization.
    - Completes the specified training batches and saves checkpoints.
    - Measures metrics such as PS, LS, RF correlation, and E/I balance. 


# save_pcb_final_diagnostics.m
    Used for saving and visualizing E-neuron metrics after training.

# save_pcb_ei_balance_summary.m
    Saves the loose- and tight-balance measurements produced after training and exports the corresponding summary data and figures.

# pcb_shuffle_conditions.m
    Defines the A--G ablation switches and combines them into the twelve experimental conditions used by the shuffle experiments.

# plot_pcb_shuffle_existing_results.m
    Reads the existing cn=1--3 result files and generates the twelve condition-comparison figures for RF correlation, LS, and PS.

# run_cn1_functional_rf_analysis.m
    analysis program for use after network training is complete.（Mainly results related to grating testing）

## 2026-10-05：动态阈值下限与相对路径

- `shuffle_study/run_ps_factorial_condition.m`：当前消融实验的训练核心。每次动态阈值更新后，把E、I神经元中小于0的阈值截断为0；关闭动态阈值的条件不执行该更新。还保存训练检查点对应的逐神经元自然图像放电率。
- `shuffle_study/pcb_results_paths.m`：统一计算本项目的结果目录，使用相对位置，不依赖原作者的用户名或盘符。
- `shuffle_study/run_pcb_shuffle_ei_balance_cn1.m`：当前文件夹提供的cn=1训练入口，运行全部12种条件，每种条件1个种子，并在训练后测量E/I平衡。
- `run_pcb_shuffle_all.m`、`run_pcb_shuffle_experiment.m`和`run_pcb_shuffle_ei_balance_cn1.m`已同步结果保存路径。原论文的`main.m`未替换；它不是阈值截断实验的入口。

在MATLAB中运行`shuffle_study/run_pcb_shuffle_ei_balance_cn1.m`可启动cn=1的12条件E/I平衡实验。程序优先使用并行计算，工具箱不可用时会提示并改用串行。旧说明中的`run_ae_threshold_floor_training.m`不在当前文件夹内。

新训练结果保存在本项目的`shuffle_study/pcb_shuffle_study/results/01_实验数据/网络规模实验/cn_1`、`cn_2`和`cn_3`中。cn=1的E/I平衡入口保存在`01_实验数据/cn1_EI平衡`中，感受野分析程序能读取这一布局。

本次只同步程序，不复制Nutstore中的结果数据，不启动训练。续跑会检查本项目目录的同名结果和检查点；不要混入未使用阈值截断规则的旧数据。

## 2026-10-31：文件夹改名后的路径与运行说明

- 现有程序目录为 `shuffle_study`、`result analysis`、`avalanche experiment`。程序使用自身位置计算项目路径；文件夹名称中的空格和中文可以保留。
- `main.m`从它所在目录读取`VanHateren_DoG_small.mat`，无需把MATLAB当前工作目录切换到项目根目录。
- `shuffle_study`的9个MATLAB文件只在文件开头保留用途说明，内部注释已清理。
- `plot_pcb_shuffle_existing_results`优先使用本项目的结果目录，缺失时仍兼容原Nutstore和D盘历史目录；外部或改名后的历史结果可通过第二个参数指定，例如`plot_pcb_shuffle_existing_results('english','D:/实验结果/results')`。该程序需要历史cn=1十种子数据以及cn=2、cn=3相关结果。
- 当前副本未包含`shuffle_study/pcb_shuffle_study/results`训练结果目录。结果分析和历史绘图仍需先训练或补齐所需数据；缺失数据不属于文件夹改名造成的路径错误。
- 雪崩实验的`input`中已有AD检查点，入口仍为`avalanche experiment/run_ad_avalanche_experiment.m`。

## 2026-11-01：标准实验入口

训练程序仅支持`paper`标准实验。保留原有函数调用方式，例如`run_pcb_shuffle_all('paper',options)`或`run_pcb_shuffle_experiment('paper','A','D',options)`；其他模式会在开始训练前报错。

cn=1默认使用16×16图像、400个E神经元和100个I神经元，每批100张图像、每张呈现400 ms，共训练1200批。训练指标按0:100:1200记录；默认E/I平衡候选数和测量设置沿用标准实验。

Treves–Rolls稀疏性计算已作为局部函数放入`run_ps_factorial_condition.m`，不再依赖单独的`modified_treves_rolls.m`文件，计算公式和静默响应处理保持一致。
