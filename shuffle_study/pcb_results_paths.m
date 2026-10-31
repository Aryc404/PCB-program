% 作用：统一管理PCB实验结果目录；只返回路径，不创建目录，也不运行实验。
% 输入：requested_root是指定的保存目录；kind是默认实验类别，通常不用填写。
% 输出：paths列出整理后的各类目录；output_root是本次实际使用的保存目录。
% 已知旧目录会自动映射到新位置，自定义目录和quick缓存目录保持原样。
function [paths,output_root] = pcb_results_paths(requested_root,kind)
if nargin < 1, requested_root = ''; end
if nargin < 2, kind = 'scaling'; end
code_dir = fileparts(mfilename('fullpath'));
paths.root = fullfile(code_dir,'pcb_shuffle_study','results');
paths.scaling = fullfile(paths.root,'01_实验数据','网络规模实验');
paths.balance = fullfile(paths.root,'01_实验数据','cn1_EI平衡');
paths.cn1_legacy = fullfile(paths.root,'01_实验数据','cn1_十种子旧版');
paths.cn3_legacy = fullfile(paths.root,'01_实验数据','cn3_三种子旧版');
% 新的十种子实验与旧版数据分开，防止新规则续跑时误用旧结果。
paths.reliability = fullfile(paths.root,'01_实验数据','cn1_十种子');
paths.figures = fullfile(paths.root,'03_汇总图表','网络规模对比12图');
paths.figures_en = fullfile(paths.root,'03_汇总图表','网络规模对比12图_英文');
paths.archive = fullfile(paths.root,'04_历史归档');
paths.predictions = fullfile(paths.root,'02_模拟预测');
paths.unverified = fullfile(paths.root,'05_待核实数据');
if strlength(string(requested_root)) == 0
    assert(isfield(paths,char(kind)),'Unknown PCB result category: %s',kind);
    output_root = paths.(char(kind));
    return;
end
output_root = char(requested_root);
old_roots = {paths.root, ...
    fullfile(paths.root,'不同cn'), ...
    fullfile(paths.root,'ei_balance_cn1'), ...
    fullfile(paths.root,'cn3_5seeds'), ...
    fullfile(paths.root,'reliability_cn1_10seeds'), ...
    fullfile(paths.root,'archive','legacy_cn1_10seeds')};
new_roots = {paths.scaling,paths.scaling,paths.balance, ...
    paths.cn3_legacy,paths.reliability,paths.cn1_legacy};
normalized = regexprep(strrep(output_root,'\','/'),'/+$','');
for index = 1:numel(old_roots)
    old_normalized = regexprep(strrep(old_roots{index},'\','/'),'/+$','');
    if strcmpi(normalized,old_normalized)
        output_root = new_roots{index};
        return;
    end
end
end
