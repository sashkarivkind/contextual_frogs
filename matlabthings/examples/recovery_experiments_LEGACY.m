%paradigm from COIN paper
% load('../fitted parameters/COIN_params_table.mat')
legacy_mode = false;
Pplus=1;
Pminus=-1;
P0=0;
Pchannel = NaN;
compareToExp = true;
stimuli = struct( ...
    'spontaneous', [ P0*ones(1,50),   Pplus*ones(1,125),   Pminus*ones(1,15),   Pchannel*ones(1,150) ], ...
    'evoked',     [ P0*ones(1,50),   Pplus*ones(1,125),   Pminus*ones(1,15),   Pchannel*ones(1,2), ...
    Pplus*ones(1,2), Pchannel*ones(1,146) ] ...
    );

alias_LUT = struct('spontaneous','S','evoked','E');
if ~legacy_mode
startingIDoffset_LUT = struct('spontaneous',8,'evoked',0);
else
   startingIDoffset_LUT_p1 = struct('spontaneous',0,'evoked',8);
   startingIDoffset_LUT_p2 = struct('spontaneous',8,'evoked',0);
end 

results = {};
experiments = {};
negLogLiHo = {};
skipto=-1;
paradigms = fields(stimuli);
cnt = 0;
for i_paradigm = 1:length(paradigms)
    paradigm = paradigms{i_paradigm};
    for i_subj=1:8
        cnt = cnt + 1;

        if cnt <skipto
            disp('skip')
            continue
        end

        model = COIN;
        model.perturbations = stimuli.(paradigm);
        % model.sigma_sensory_noise = 0.01;
        %         model = assign_coin_params(model,[alias_LUT.(paradigm),num2str(i_subj)],paramsTable);
        if ~legacy_mode
            model = assign_coin_params_v2(model,[startingIDoffset_LUT.(paradigm)+i_subj],POpt);
        else
            % model = assign_coin_params(model,[alias_LUT.(paradigm),num2str(i_subj)],mappedTable);
            i_subj_upd = subj_order_vec(i_subj + startingIDoffset_LUT_p1.(paradigm)) - startingIDoffset_LUT_p2.(paradigm);
            model = assign_coin_params(model,[alias_LUT.(paradigm),num2str(i_subj_upd)],paramsTable);

        end            


        model.runs = 10;
        model.particles = 100;
        model.max_cores = feature('numcores');
        model.plot_average_state = true;
        %         model.plot_state_given_context = true;
        %         model.plot_predicted_probabilities = true;
        %         model.plot_state = true;
        %         model.plot_global_transition_probabilities = true;
        %         model.plot_local_transition_probabilities = true;
        %         model.plot_responsibilities = true;

        % if compareToExp
        measuredData = load(['/homes/ar2342/frogs_project/data/COIN_data/',paradigm,'_recovery_participant',num2str(i_subj),'.mat']);
        % measuredData = load(['/homes/ar2342/frogs_project/data/COIN_data/',paradigm,'_recovery_participant',num2str(inv_subj_order_vec(8*i_paradigm+i_subj)),'.mat']);
        experiments{end+1} = measuredData.TrialData.Adaptation;
        if compareToExp
        model.adaptation = measuredData.TrialData.Adaptation*sign(nanmean(measuredData.TrialData.Adaptation(50:125)));

        end
        results{end+1} = model.simulate_COIN;
if compareToExp
        nll = model.objective_COIN;
        negLogLiHo{end+1} = nll;
end
        % close all;
    end
end

%%
% figure(1); clf;
figure;
for ii=1:length(results)
    these_results = results{ii};
    these_exp = experiments{ii};
    subplot(4,4,ii);
    %     plot(0);
    %     hold on;
    plot(these_results.plots.average_state);
    hold on;
    plot(sign(nanmean(these_exp(50:125)))*these_exp,'*')
    ylim([-1.3,1.3]);
    xlim([0,340]);
    if ii<=8
        text(0,-1,['myS',num2str(ii)]);
    else
        text(0,-1,['myE',num2str(ii-8)]);
    end
end