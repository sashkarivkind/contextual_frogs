% plot_results.m
clear; clf; hold on;

% resultsDir = './antero_10runsExact';
 resultsDir = '/homes/ar2342/frogs_project/COIN/results/antero_results/antero_r10_p1000_mSy';
%resultsDir = '/homes/ar2342/frogs_project/COIN/results/antero_results/antero_r10_p100_mSyCTRLMEX_';

%resultsDir = './antero10runs1000ptcls';
paradigms  = {'AB0','AB1','AB2','AB3','AB4','AB5'};  % etc.
colors     = lines(numel(paradigms));
lineHandles = gobjects(numel(paradigms),1);
one_by_one = true;

for ip = 1:numel(paradigms)
    paradigm = paradigms{ip};
    files = dir(fullfile(resultsDir, sprintf('%s_*.mat', paradigm)));
    if isempty(files), continue; end

    % load motor_output
    data = cell(numel(files),1);
    for k = 1:numel(files)
        S = load(fullfile(resultsDir,files(k).name),'res');
        data{k} = S.res.runs{1}.motor_output(:)';
    end

    if one_by_one
        for k = 1:numel(files)
            subplot(8,5,k);
            plot(data{k});
            hold on;
        end
    else

        % align to end, pad with NaN
        Ls   = cellfun(@numel,data);
        Tmax = max(Ls);
        M    = NaN(numel(files),Tmax);
        for k=1:numel(files)
            v = data{k}; L= numel(v);
            M(k,Tmax-L+1:end) = v;
        end

        mu  = nanmean(M,1);
        sem = nanstd (M,0,1) ./ sqrt(sum(~isnan(M),1));
        t   = -(Tmax-1):0;

        % draw patch but turn its legend‐handle off
        patch( [t fliplr(t)], [mu-sem fliplr(mu+sem)], colors(ip,:), ...
            'FaceAlpha',0.3, 'EdgeColor','none', 'HandleVisibility','off');

        % plot mean and capture its handle
        lineHandles(ip) = plot(t, mu, 'LineWidth',2, 'Color',colors(ip,:));
    end
end

xlabel('trials');
ylabel('adaptation');
legend(lineHandles, paradigms,'Location','Best');
title('anterograde interference (??)');
grid on;
