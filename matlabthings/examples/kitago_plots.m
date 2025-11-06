% plot_results.m
clear;
figure;
clf; hold on;

kitago_norm = true;
% resultsDir = './antero_10runsExact';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/antero_results/antero_r10_p100_mSy';
%resultsDir = '/homes/ar2342/frogs_project/COIN/results/herzf_r10_p1000_';
%resultsDir = '/homes/ar2342/frogs_project/COIN/results/frogs_r10_p1000_b_';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/frogsV2_r10_p1000_b_';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/all_A_r100_p1000_b_';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/all_A_r100_p1000_b_';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/all_r100_p1000_b_';
% resultsDir = '/homes/ar2342/frogs_project/COIN/results/all_r10_p1000_';
resultsDirs = {'/homes/ar2342/frogs_project/COIN/results/all_r10_p1000b_',...
    '/homes/ar2342/frogs_project/COIN/results/all_r10_p1000_'};
for kk = 1:length(resultsDirs)
    resultsDir = resultsDirs{kk};

%resultsDir = './antero10runs1000ptcls';
% paradigms  = {'herzfeld_z_1'};%','herzfeld_z_0_5','herzfeld_z_0_1'};% {'AB0','AB1','AB2','AB3','AB4','AB5'};  % etc.
% paradigms  = {'herzfeld_z_0_9','herzfeld_z_0_5','herzfeld_z_0_1'};% {'AB0','AB1','AB2','AB3','AB4','AB5'};  % etc.
% paradigms  = {'frogs','anti_frogs'};% {'AB0','AB1','AB2','AB3','AB4','AB5'};  % etc.
% paradigms  = {'toy__savings'}
% paradigms  =  {'AB0','AB1','AB2','AB3','AB4','AB5'};  % etc.
% paradigms  =  {'lng_AB0','lng_AB1','lng_AB2','lng_AB3','lng_AB4','lng_AB5'};  % etc.
paradigms = {'Kitago'} %,'KitagoX2','KitagoX0p5'}
colors     = lines(5*numel(paradigms));
lineHandles = gobjects(numel(paradigms),1);

one_by_one = false;
single_trial_result = false; % used for Herzfeld
invT = false;

for ip = 1:numel(paradigms)
    paradigm = paradigms{ip};
    files = dir(fullfile(resultsDir, sprintf('%s_*.mat', paradigm)));
    if isempty(files), continue; end

    % load motor_output
    data = cell(numel(files),1);
    for k = 1:numel(files)
        S = load(fullfile(resultsDir,files(k).name),'res');
        data{k} = S.res.runs{1}.motor_output(:)';
        if single_trial_result
            d = data{k};
            before = (45:45:1125)-2;
            after = before + 2;
            data{k} = d(after) - d(before);
        end
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
        if kitago_norm
            M = M ./ nanmean(M(:,101:120),2);
        end
        
        mu  = nanmean(M,1);
        sem = nanstd (M,0,1) ./ sqrt(sum(~isnan(M),1));
        if invT
        t   = -(Tmax-1):0;
        else
            t= 1:Tmax;
        end

        % draw patch but turn its legend‐handle off
        patch( [t fliplr(t)], [mu-sem fliplr(mu+sem)], colors(ip+kk-1,:), ...
            'FaceAlpha',0.3, 'EdgeColor','none', 'HandleVisibility','off');

        % plot mean and capture its handle
        lineHandles(ip) = plot(t, mu, 'LineWidth',2, 'Color',colors(ip+kk-1,:));
    end
end
end

% xlabel('trials');
% ylabel('adaptation');
% legend(lineHandles, paradigms,'Location','Best');
% title('anterograde interference (??)');
legend('with bias', 'without bias')
if kitago_norm
    ylabel('adaptation [fraction of P+ ADAPTATION]')
else
ylabel('adaptation [fraction of P+ perturbation]')
end
xlabel('trails')
grid on;
