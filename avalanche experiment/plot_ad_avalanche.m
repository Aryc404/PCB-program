% 作用：画图程序
function plot_ad_avalanche(result,output_dir)
if nargin<2, output_dir=result.output_dir; end
if ~isfolder(output_dir), mkdir(output_dir); end
a=result.analysis;
bin_seconds=result.options.bin_steps*result.cfg.dt;
f=new_figure();
stairs(((1:numel(result.activity_E))'-0.5)*bin_seconds,result.activity_E, ...
    'Color',[.1 .35 .65],'LineWidth',.8);
hold on; yline(a.threshold,'--',sprintf('%g-percentile threshold', ...
    result.options.threshold_percentile),'Color',[.7 .2 .1]);
xlabel('Time (s)'); ylabel(sprintf('E-population spike count / %g ms',bin_seconds*1000));
title('Network activity'); export_figure(f,output_dir,'network_activity');
events=a.events(a.events.complete,:);
plot_distribution(events.size_spikes,a.fit_size,1,'Avalanche size', ...
    'Size (spikes)',output_dir,'avalanche_size');
plot_distribution(events.duration_bins,a.fit_duration,bin_seconds*1000,'Avalanche duration', ...
    'Duration (ms)',output_dir,'avalanche_duration');
f=new_figure(); s=a.scaling;
if ~isempty(s.duration_bins)
    loglog(s.duration_bins*bin_seconds*1000,s.mean_size,'o', ...
        'Color',[.6 .6 .6]); hold on;
    mask=s.use_for_fit;
    loglog(s.duration_bins(mask)*bin_seconds*1000,s.mean_size(mask),'o', ...
        'Color',[.1 .35 .65],'MarkerFaceColor',[.1 .35 .65]);
    if isfinite(a.beta_fit)
        x=s.duration_bins(mask);
        y=exp(s.intercept)*x.^a.beta_fit;
        loglog(x*bin_seconds*1000,y,'-','Color',[.8 .25 .1],'LineWidth',1.5);
    end
end
xlabel('Duration (ms)'); ylabel('Mean avalanche size (spikes)');
title('Size-duration scaling');
text(.98,.96,sprintf('beta = %.3g\nDCC = %.3g\nfit valid = %d', ...
    a.beta_fit,a.dcc,a.valid),'Units','normalized','HorizontalAlignment','right', ...
    'VerticalAlignment','top','Interpreter','none');
export_figure(f,output_dir,'size_duration_scaling');
end

% 作用：画原始事件的概率质量和选定区间的MLE曲线。factor只把duration换成毫秒。
function plot_distribution(values,fit,factor,heading,x_label,output_dir,file_name)
f=new_figure();
if ~isempty(values)
    [x,~,group]=unique(double(values));
    p=accumarray(group,1)/numel(values);
    loglog(x*factor,p,'.','Color',[.1 .35 .65],'MarkerSize',12); hold on;
    if isfinite(fit.exponent) && fit.xmax>=fit.xmin && fit.xmax-fit.xmin<2e6
        support=(fit.xmin:fit.xmax)';
        log_p=-fit.exponent*log(support); log_p=log_p-max(log_p);
        model_p=exp(log_p); model_p=model_p/sum(model_p);
        fraction=mean(values>=fit.xmin & values<=fit.xmax);
        loglog(support*factor,model_p*fraction,'-','Color',[.8 .25 .1],'LineWidth',1.3);
    end
end
xlabel(x_label); ylabel('Probability mass'); title(heading);
text(.98,.96,sprintf('Exponent = %.3g\nKS bootstrap p = %.3g\nfit valid = %d', ...
    fit.exponent,fit.p_value,fit.valid),'Units','normalized', ...
    'HorizontalAlignment','right','VerticalAlignment','top','Interpreter','none');
export_figure(f,output_dir,file_name);
end

% 作用：新建统一白底坐标轴；输出窗口句柄，不改变MATLAB全局默认样式。
function f=new_figure()
f=figure('Color','w','Visible','off','Position',[100 100 1100 600]);
ax=axes(f); set(ax,'Color','w','FontName','Arial','FontSize',12,'LineWidth',1, ...
    'Box','off','XColor','k','YColor','k');
end

% 作用：保存一张PNG和可编辑FIG，然后关闭该窗口。
function export_figure(f,output_dir,file_name)
cleanup=onCleanup(@() close(f));
% plot/loglog可能重置坐标轴样式；最后再次固定白底黑字，避免受MATLAB深色主题影响。
set(f,'Color','w','InvertHardcopy','off');
ax=findall(f,'Type','axes');
set(ax,'Color','w','XColor','k','YColor','k','ZColor','k','FontName','Arial');
set(findall(f,'Type','text'),'Color','k');
exportgraphics(f,fullfile(output_dir,[file_name '.png']),'Resolution',200,'BackgroundColor','white');
savefig(f,fullfile(output_dir,[file_name '.fig']));
end
