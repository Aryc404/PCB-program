% This script runs a demo of the biologically-grounded spiking model that
% learns V1 cells
% Authors: Marko A. Ruslim and Martin J. Spencer
% Date: 01/11/2024
% Note that a smaller image dataset is used here (8)
% Also note that learning with background/spontaneous firing 
% has been omitted in this demo

%% Parameter initialisation

clear; clc; close all;
tic;

rng('shuffle');

N_X = 16 ^ 2; % number of input neurons
N_E = 20 ^ 2; % number of output excitatory neurons
N_I = N_E / 4; % number of output inhibitory neurons

N_n = 1200; % number of batches for main learning
N_q = 100; % number of batches for updating thresholds with fixed weights
N_b = 100; % batch size
N_t = 400; % (ms) stimulus presentation duration

dt = 1e-3; % (s / ms) time step
tau_m = 10e-3; % (s) membrane time constant for PSP
tau_i = 20e-3; % (s) STDP time constant
tau_l = 50e-3; % (s) long STDP time constant for triplet rule

% L1 norm of weights
L1_EX = 100; L1_IX = 80;
L1_EE = 10; L1_IE = 240; 
L1_EI = 120; L1_II = 120;

rho_E = 2; % (Hz) target output excitatory firing rate
rho_I = rho_E * 2; % (Hz) target output inhibitory firing rate

V_min = -10;

% Learning rates
eta_EX = 2e-4; eta_EE = 1e-4; eta_EI = 9e-2; eta_tE = 1e-0;
eta_IX = 3e-3; eta_IE = 4e-2; eta_II = 6e-2; eta_tI = 1e-0;

theta_E0 = 10; % initial output excitatory threshold value
theta_I0 = 10; % initial output inhibitory threshold value
sigma = 1;

raster_show = 10;

%% Variable initialisation

T = N_t * dt; % Duration of stimulus presentation
p_E = rho_E; p_I = rho_I;

rho_E = rho_E * T; % Convert Hz to spikes / image 变成放电率(hz)
rho_I = rho_I * T; % Convert Hz to spikes / image

exp_m = exp(-dt / tau_m); % decay factor at each time

A_i = 0.5 ; % coefficient of symmetrical STDP
exp_i = exp(-dt / tau_i); % decay factor at each time step

A_p = 1; % potentiation coefficient factor of triplet STDP
exp_l = exp(-dt / tau_l); % decay factor at each time step

A_d = p_E; % depression coefficient factor of triplet STDP
invtau_i = 1/tau_i; invtau_l = 1/tau_l;

% Initialise weights
p = 0.2;
W_EX = normrnd(1, 0.5, [N_E, 2*N_X]);
I = (rand([N_E, 2*N_X]) > p);
W_EX(I) = W_EX(I) / 100; W_EX(W_EX<0) = 0;
W_EX = W_EX ./ sum(W_EX, 2) * L1_EX;
W_EE = normrnd(1, 0.5, [N_E, N_E]);
I = (rand([N_E, N_E]) > p);
W_EE(I) = W_EE(I) / 100; W_EE(W_EE<0) = 0;
W_EE = W_EE - diag(diag(W_EE));
W_EE = W_EE ./ sum(W_EE, 2) * L1_EE;
W_EI = normrnd(1, 0.5, [N_E, N_I]);
I = (rand([N_E, N_I]) > p);
W_EI(I) = W_EI(I) / 100; W_EI(W_EI<0) = 0;
W_EI = W_EI ./ sum(W_EI, 2) * L1_EI;
W_IX = normrnd(1, 0.5, [N_I, 2*N_X]);
I = (rand([N_I, 2*N_X]) > p);
W_IX(I) = W_IX(I) / 100; W_IX(W_IX<0) = 0;
W_IX = W_IX ./ sum(W_IX, 2) * L1_IX;
W_IE = normrnd(1, 0.5, [N_I, N_E]);
I = (rand([N_I, N_E]) > p);
W_IE(I) = W_IE(I) / 100; W_IE(W_IE<0) = 0;
W_IE = W_IE ./ sum(W_IE, 2) * L1_IE;
W_II = normrnd(1, 0.5, [N_I, N_I]);
I = (rand([N_I, N_I]) > p);
W_II(I) = W_II(I) / 100; W_II(W_II<0) = 0;
W_II = W_II - diag(diag(W_II)); 
W_II = W_II ./ sum(W_II, 2) * L1_II;

W0.EX = W_EX; W0.EE = W_EE; W0.EI = W_EI;
W0.IX = W_IX; W0.IE = W_IE; W0.II = W_II;

% Initialise spiking thresholds
theta_E = theta_E0 * ones(N_E, 1);
theta_I = theta_I0 * ones(N_I, 1);

% Weight change and threshold change variables
dW_EX = zeros(N_E, 2*N_X); dW_EE = zeros(N_E, N_E); 
dW_EI = zeros(N_E, N_I); dW_IX = zeros(N_I, 2*N_X);
dW_IE = zeros(N_I, N_E); dW_II = zeros(N_I, N_I);
dtheta_E = zeros(N_E, 1); dtheta_I = zeros(N_I, 1);

% Firing rate and membrane potential variables
R_X = zeros(2*N_X, N_b); R_E = zeros(N_E, N_b); R_I = zeros(N_I, N_b);
V_E = zeros(N_E, N_b);
V_I = zeros(N_I, N_b);

% Synaptic trace variables
U_Xi = zeros(2*N_X, N_b);
U_Ei = zeros(N_E, N_b); U_El = zeros(N_E, N_b);
U_Ii = zeros(N_I, N_b); U_Il = zeros(N_I, N_b);

% Spike variables
S_X = false(2*N_X, N_b); S_E = false(N_E, N_b); S_I = false(N_I, N_b);

rho_Eeta_tE = rho_E*eta_tE; rho_Ieta_tI = rho_I*eta_tI;

fprintf('Weights and other variables initialised... '); toc; tic;

load VanHateren_DoG_small.mat;
IMAGES = IMAGES_DoG; clear IMAGES_DoG;

fprintf('Image dataset loaded... '); toc; tic;

%% Update spiking thresholds before learning weights

fprintf("Updating spiking thresholds with weights fixed " + ...
    "(%d iterations)...\nIter: ", N_q);

for i_T = 1 : N_q
    
    % Reset variables
    V_E = zeros(N_E, N_b); V_I = zeros(N_I, N_b);
    R_E = zeros(N_E, N_b); R_I = zeros(N_I, N_b); 
    R_X = generate_image_patches(IMAGES, N_X, N_b);

    R_X(R_X>100) = 100; % ##############

    R_XDT = R_X * dt;
    
    for i_t = 1 : N_t
        S_X = rand(2*N_X, N_b) < R_XDT; % Poisson process for LGN
        % Neuron dynamics
        S_E = V_E > theta_E; S_I = V_I > theta_I;
        V_E(S_E) = 0; V_I(S_I) = 0;
        V_E = V_E * exp_m + W_EX * S_X + W_EE * S_E - W_EI * S_I + ...
            normrnd(0, 1, [N_E, N_b]);
        V_I = V_I * exp_m + W_IX * S_X + W_IE * S_E - W_II * S_I + ...
            normrnd(0, 1, [N_I, N_b]);

        V_E(V_E<V_min) = V_min; V_I(V_I<V_min) = V_min;

        R_E = R_E + S_E; R_I = R_I + S_I;
    end
    % Update spiking thresholds
    dtheta_E = (mean(R_E, 2) - rho_E) * eta_tE;
    dtheta_I = (mean(R_I, 2) - rho_I) * eta_tI;
    dtheta_E = min(dtheta_E, rho_Eeta_tE); %约束每次阈值变化的最大值
    dtheta_I = min(dtheta_I, rho_Ieta_tI);
    theta_E = theta_E + dtheta_E;
    theta_I = theta_I + dtheta_I;
    
    fprintf('%d ', i_T)
    if mod(i_T, 10) == 0
        fprintf('\n');
    end
    drawnow limitrate;
end

fprintf('Spiking thresholds updated... '); toc; tic;

%% Spike raster figure

spike_figure = figure;
spike_axes = gca;

%% Main loop

fprintf("Running main simulation " + ...
    "(%d iterations)...\nIter: ", N_n);

for i_T = N_q+1 : N_n

    % Reset variables
    V_E = zeros(N_E, N_b); V_I = zeros(N_I, N_b);
    R_E = zeros(N_E, N_b); R_I = zeros(N_I, N_b); R_X2 = zeros(N_X*2, N_b);
    U_Xi = zeros(2*N_X, N_b);
    U_Ei = zeros(N_E, N_b); U_El = zeros(N_E, N_b);
    U_Ii = zeros(N_I, N_b); U_Il = zeros(N_I, N_b);
    dW_EX = zeros(N_E, 2*N_X); dW_EE = zeros(N_E, N_E); 
    dW_EI = zeros(N_E, N_I); dW_IX = zeros(N_I, 2*N_X);
    dW_IE = zeros(N_I, N_E); dW_II = zeros(N_I, N_I);
    spikes_e = []; spikes_i = []; % save spike history for raster plot
    
    R_X = generate_image_patches(IMAGES, N_X, N_b);

    R_X(R_X>100) = 100;

    R_XDT = R_X * dt;

    for i_t = 1 : N_t
        S_X = rand(2*N_X, N_b) < R_XDT;
        S_E = V_E > theta_E; S_I = V_I > theta_I;
        V_E(S_E) = 0; V_I(S_I) = 0;
        R_E = R_E + S_E; R_I = R_I + S_I; R_X2 = R_X2 + S_X;
        V_E = V_E * exp_m + W_EX * S_X + W_EE * S_E - W_EI * S_I + ...
            normrnd(0, 1, [N_E, N_b]);
        V_I = V_I * exp_m + W_IX * S_X + W_IE * S_E - W_II * S_I + ...
            normrnd(0, 1, [N_I, N_b]);
        
        V_E(V_E<V_min) = V_min; V_I(V_I<V_min) = V_min;

        if mod(i_T, raster_show) == 0
            for i = find(S_E(:,1))'
                spikes_e(end+1, :) = [i_t, i];
            end
            for i = find(S_I(:,1))'
                spikes_i(end+1, :) = [i_t, i];
            end
        end
        
        % Update synaptic traces (decay), U
        U_Xi = U_Xi * exp_i;
        U_Ei = U_Ei * exp_i; U_El = U_El * exp_l;
        U_Ii = U_Ii * exp_i; U_Il = U_Il * exp_l;
        
        % Update change in weights, dW
        % X -> E 3STDP
        dW_EX = dW_EX + A_p*(S_E .* U_El) * U_Xi' - ...
                A_d*U_Ei * S_X';
        % E -> E 3STDP
        dW_EE = dW_EE + A_p * (S_E .* U_El) * U_Ei' - (A_d .* U_Ei) * S_E';
        % I -> E sSTDP
        dW_EI = dW_EI + A_i * S_E * U_Ii' + A_i * U_Ei * S_I' + ...
            S_E * S_I' * A_i; % accounts for synchronous firing
        % X -> I sSTDP
        dW_IX = dW_IX + A_i*S_I * U_Xi' + A_i*U_Ii * ...
                S_X' + S_I * S_X' * A_i;
        % E -> I sSTDP
        dW_IE = dW_IE + A_i * S_I * U_Ei' + A_i * U_Ii * S_E' + ...
            S_I * S_E' * A_i; % accounts for synchronous firing
        % I -> I sSTDP
        dW_II = dW_II + A_i * U_Ii * S_I' + A_i * S_I * U_Ii' + ...
            S_I * S_I' * A_i; % accounts for synchronous firing

        % Update synaptic traces (new spikes), U
        U_Xi(S_X) = U_Xi(S_X) + invtau_i;
        U_Ei(S_E) = U_Ei(S_E) + invtau_i; U_El(S_E) = U_El(S_E) + invtau_l;
        U_Ii(S_I) = U_Ii(S_I) + invtau_i; U_Il(S_I) = U_Il(S_I) + invtau_l;
    end

    R_E = mean(R_E, 2); R_I = mean(R_I, 2);
    
    dW_EX = dW_EX / N_b * eta_EX;
    dW_EE = dW_EE / N_b * eta_EE;
    dW_EI = dW_EI / N_b * eta_EI;
    dW_IX = dW_IX / N_b * eta_IX;
    dW_IE = dW_IE / N_b * eta_IE;
    dW_II = dW_II / N_b * eta_II;
    dtheta_E = (R_E - rho_E) * eta_tE; 
    dtheta_I = (R_I - rho_I) * eta_tI;
    dtheta_E = min(dtheta_E, rho_Eeta_tE);
    dtheta_I = min(dtheta_I, rho_Ieta_tI);

    % Update weights and normalise
    W_EX = W_EX + dW_EX; W_EX(W_EX<0) = 0;
    tmp = sum(W_EX, 2);
    W_EX = W_EX + (-tmp + min(tmp, L1_EX)) / 2/N_X;
    W_EX(W_EX<0) = 0;
    tmp = sum(W_EX, 2);
    W_EX = W_EX ./ tmp .* min(tmp, L1_EX);
    W_EE = W_EE + dW_EE; W_EE(W_EE<0) = 0;
    W_EE = W_EE - diag(diag(W_EE));
    tmp = sum(W_EE, 2);
    W_EE = W_EE + (-tmp + min(tmp, L1_EE)) / N_E;
    W_EE(W_EE<0) = 0;
    tmp = sum(W_EE, 2);
    W_EE = W_EE ./ tmp .* min(tmp, L1_EE);
    W_EI = W_EI + dW_EI; W_EI(W_EI<0) = 0;
    tmp = sum(W_EI, 2);
    W_EI = W_EI + (-tmp + min(tmp, L1_EI)) / N_I;
    W_EI(W_EI<0) = 0;
    tmp = sum(W_EI, 2);
    W_EI = W_EI ./ tmp .* min(tmp, L1_EI);
    
    W_IX = W_IX + dW_IX; W_IX(W_IX<0) = 0;
    tmp = sum(W_IX, 2);
    W_IX = W_IX + (-tmp + min(tmp, L1_IX)) / 2/N_X;
    W_IX(W_IX<0) = 0;
    tmp = sum(W_IX, 2);
    W_IX = W_IX ./ tmp .* min(tmp, L1_IX);
    W_IE = W_IE + dW_IE; W_IE(W_IE<0) = 0;
    tmp = sum(W_IE, 2);
    W_IE = W_IE + (-tmp + min(tmp, L1_IE)) / N_E;
    W_IE(W_IE<0) = 0;
    tmp = sum(W_IE, 2);
    W_IE = W_IE ./ tmp .* min(tmp, L1_IE);
    W_II = W_II + dW_II; W_II(W_II<0) = 0;
    W_II = W_II - diag(diag(W_II));
    tmp = sum(W_II, 2);
    W_II = W_II + (-tmp + min(tmp, L1_II)) / N_I;
    W_II(W_II<0) = 0;
    tmp = sum(W_II, 2);
    W_II = W_II ./ tmp .* min(tmp, L1_II);
    theta_E = theta_E + dtheta_E; theta_I = theta_I + dtheta_I;
    
    if mod(i_T, raster_show) == 0
        cla(spike_axes);
        if ~isempty(spikes_e)
            scatter(spike_axes, spikes_e(:,1), spikes_e(:,2), '.b');
            hold(spike_axes, 'on');
        end
        if ~isempty(spikes_i)
            scatter(spike_axes, spikes_i(:,1), 1-spikes_i(:,2), '.r');
            hold(spike_axes, 'off');
        end
    end
        
    if i_T == 400 || i_T == 800
        eta_EX=eta_EX/2; eta_EE=eta_EE/2; eta_EI=eta_EI/2;
        eta_IX=eta_IX/2; eta_IE=eta_IE/2; eta_II=eta_II/2;
        eta_tE = eta_tE/2; eta_tI = eta_tI/2;
    end
    fprintf('%d ', i_T);
    if mod(i_T, 10) == 0
        fprintf('\n')
    end
    drawnow limitrate;
end

W = struct();
W.EX = W_EX; W.EE = W_EE; W.EI = W_EI;
W.IX = W_IX; W.IE = W_IE; W.II = W_II;
theta = struct(); theta.E = theta_E; theta.I = theta_I;

clearvars -except W0 W theta;

fprintf('Simulation completed... '); toc;
