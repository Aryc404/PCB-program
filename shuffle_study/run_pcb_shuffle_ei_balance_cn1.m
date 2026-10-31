% 用途：启动cn=1的全部12种消融条件训练，每种条件运行一次，在第400、800、1200批保存检查点并测量最终E/I平衡。
simulation_mode = 'paper';
code_dir = fileparts(mfilename('fullpath'));
addpath(code_dir);

options = struct();
options.inhibitory_selection = 'ALL';
options.homeostatic_selection = 'ALL';
options.network_scales = 1;
options.runs_per_scale = 1;
options.balance_only_scales = 1;
paths = pcb_results_paths();
options.output_root = paths.balance;
options.use_parallel = true;
options.max_workers = 8;
options.resume = true;

options.training_checkpoint_batches = [400 800 1200];
options.training_progress_every = 100;

options.measure_ei_balance = true;
options.balance_seed = 3202603;
options.balance_neuron_count = 40;
options.balance_example_candidates = 1000;
options.balance_population_candidates = 1000;
options.balance_scatter_candidates = 100;
options.balance_tight_candidates = 1000;
options.balance_tight_top_k = 10;
options.balance_transient_ms = 100;
options.balance_max_lag_ms = 50;

summary = run_pcb_shuffle_all(simulation_mode,options);
