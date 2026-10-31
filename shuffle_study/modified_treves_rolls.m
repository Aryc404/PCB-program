% 用途：计算响应矩阵的Treves–Rolls稀疏性、逐刺激稀疏性和静默比例，可选择有限群体校正。
function [mean_ps, ps_each, silent_fraction] = modified_treves_rolls( ...
        response, finite_population_correction)

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
