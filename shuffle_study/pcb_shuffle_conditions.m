function output = pcb_shuffle_conditions(inhibitory_code, homeostatic_code)
%PCB_SHUFFLE_CONDITIONS Central definitions for conditions A--G.
%   definitions = pcb_shuffle_conditions()
%   condition = pcb_shuffle_conditions('A','D')
%
% Inhibitory conditions:
%   A: I->E transmission ON,  I->E plasticity ON
%   B: I->E transmission ON,  I->E plasticity OFF
%   C: I->E transmission OFF, I->E plasticity OFF
%
% Homeostatic conditions:
%   D: adaptive thresholds ON,  L1 normalization ON
%   E: adaptive thresholds ON,  L1 normalization OFF
%   F: adaptive thresholds OFF, L1 normalization ON
%   G: adaptive thresholds OFF, L1 normalization OFF

definitions.inhibitory = struct( ...
    'code', {'A','B','C'}, ...
    'display_name', { ...
        'I->E transmission ON; I->E plasticity ON', ...
        'I->E transmission ON; I->E plasticity OFF', ...
        'I->E transmission OFF; I->E plasticity OFF'}, ...
    'transmission_enabled', {true,true,false}, ...
    'plasticity_enabled', {true,false,false});

definitions.homeostatic = struct( ...
    'code', {'D','E','F','G'}, ...
    'display_name', { ...
        'Adaptive thresholds ON; L1 normalization ON', ...
        'Adaptive thresholds ON; L1 normalization OFF', ...
        'Adaptive thresholds OFF; L1 normalization ON', ...
        'Adaptive thresholds OFF; L1 normalization OFF'}, ...
    'threshold_enabled', {true,true,false,false}, ...
    'l1_normalization_enabled', {true,false,true,false});

% Reliability repetitions disabled: every condition x cn runs once.
% Previous 10-seed configuration (kept here only for possible restoration):
% definitions.seeds.training = ...
%     [20250718 20250729 20250811 20250823 20250907, ...
%      20250919 20251003 20251017 20251101 20251113];
% definitions.seeds.eval_patch = 12025 + (0:9)*1000;
% definitions.seeds.eval_dynamics = 22025 + (0:9)*1000;
% definitions.seeds.eval_grating = 1202603 + (0:9)*1000;

% One fixed RNG configuration is retained solely for reproducibility.
definitions.seeds.training = 20250718;
definitions.seeds.eval_patch = 12025;
definitions.seeds.eval_dynamics = 22025;
definitions.seeds.eval_grating = 1202603;

% Ten matched RNG configurations for the cn=1 reliability experiment.
% Every condition uses the same seed set so differences can be compared
% without confounding condition identity with random initialization.
definitions.reliability_seeds.training = ...
    [20250718 20250729 20250811 20250823 20250907, ...
     20250919 20251003 20251017 20251101 20251113];
definitions.reliability_seeds.eval_patch = 12025 + (0:9)*1000;
definitions.reliability_seeds.eval_dynamics = 22025 + (0:9)*1000;
definitions.reliability_seeds.eval_grating = 1202603 + (0:9)*1000;

definitions.valid_inhibitory_codes = ["A","B","C"];
definitions.valid_homeostatic_codes = ["D","E","F","G"];

% Network-size multipliers from PCB Methods 2.8.5.
definitions.network_scales = [1 2 3 4];

% The four condition groups in the supervisor's final 12-chart layout.
% Each group produces one chart for LS, PS and RF correlation.
definitions.figure_groups = { ...
    ["AD","AE","AF","AG"], ...
    ["AD","BD","CD"], ...
    ["BE","BF","BG"], ...
    ["CE","CF","CG"]};
definitions.figure_group_names = { ...
    'keep_inhibition_A', ...
    'keep_homeostasis_D', ...
    'remaining_B_conditions', ...
    'remaining_C_conditions'};

if nargin == 0
    output = definitions;
    return;
end
if nargin ~= 2
    error('Provide both an A/B/C code and a D/E/F/G code.');
end

inhibitory_code = upper(string(inhibitory_code));
homeostatic_code = upper(string(homeostatic_code));
if ~isscalar(inhibitory_code) || ...
        ~any(inhibitory_code == definitions.valid_inhibitory_codes)
    error('Inhibitory condition must be A, B or C.');
end
if ~isscalar(homeostatic_code) || ...
        ~any(homeostatic_code == definitions.valid_homeostatic_codes)
    error('Homeostatic condition must be D, E, F or G.');
end

inhibitory = definitions.inhibitory( ...
    strcmp({definitions.inhibitory.code}, char(inhibitory_code)));
homeostatic = definitions.homeostatic( ...
    strcmp({definitions.homeostatic.code}, char(homeostatic_code)));

output.id = [char(inhibitory_code) char(homeostatic_code)];
output.inhibitory_code = char(inhibitory_code);
output.homeostatic_code = char(homeostatic_code);
output.display_name = sprintf('%s | %s', ...
    inhibitory.display_name, homeostatic.display_name);
output.inhibitory_transmission_enabled = ...
    inhibitory.transmission_enabled;
output.inhibitory_plasticity_enabled = inhibitory.plasticity_enabled;
output.threshold_enabled = homeostatic.threshold_enabled;
output.l1_normalization_enabled = ...
    homeostatic.l1_normalization_enabled;
end
