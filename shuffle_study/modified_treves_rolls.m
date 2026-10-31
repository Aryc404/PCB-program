function [mean_ps, ps_each, silent_fraction] = modified_treves_rolls( ...
        response, finite_population_correction)
%MODIFIED_TREVES_ROLLS Population sparseness used in Ruslim et al. (2025).
% response is neurons x stimuli. A completely silent stimulus has undefined
% sparseness and is therefore reported as NaN instead of being called sparse.
% Finite-population correction divides by (1-1/N), where N is the number
% of rows. It is enabled by default for the current PCB experiments.

if nargin < 2 || isempty(finite_population_correction)
    finite_population_correction = true;
end

mean_r = mean(response, 1);
mean_r2 = mean(response .^ 2, 1);
ps_each = 1 - (mean_r .^ 2) ./ mean_r2;
ps_each(mean_r2 == 0) = NaN;

if finite_population_correction
    population_size = size(response, 1);
    if population_size <= 1
        error('Finite-population correction requires at least two rows.');
    end
    ps_each = ps_each / (1 - 1 / population_size);
end

valid = ~isnan(ps_each);
if any(valid)
    mean_ps = mean(ps_each(valid));
else
    mean_ps = NaN;
end
silent_fraction = mean(~valid);
end
