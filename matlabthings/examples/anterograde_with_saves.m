% make sure there’s a place to save
runname = '/homes/ar2342/frogs_project/COIN/results/antero_results/antero_r10_p100_mSyCTRLMEX_'; % TSTantero_10runsExact'
if ~exist(runname,'dir')
    mkdir(runname);
end

% your stimuli definition goes here...
Pplus=1; Pminus=-1; P0=0;
TaN=150; TaB=120;
TaN=160; TaB=115;
stimuli = struct( ...
  'AB0', [ P0*ones(1,TaN),           Pminus*ones(1,TaB) ], ...
  'AB1', [ P0*ones(1,TaN),   Pplus*ones(1,13),  Pminus*ones(1,TaB) ], ...
  'AB2', [ P0*ones(1,TaN),   Pplus*ones(1,41),  Pminus*ones(1,TaB) ], ...
  'AB3', [ P0*ones(1,TaN),  Pplus*ones(1,112),  Pminus*ones(1,TaB) ], ...
  'AB4', [ P0*ones(1,TaN),  Pplus*ones(1,230),  Pminus*ones(1,TaB) ], ...
  'AB5', [ P0*ones(1,TaN),  Pplus*ones(1,369),  Pminus*ones(1,TaB) ]  ...
);

% build subject list
subject_alias = {};
for i=1:8,  subject_alias{end+1} = ['S',num2str(i)];  end
for i=1:8,  subject_alias{end+1} = ['E',num2str(i)];  end
for i=1:24, subject_alias{end+1} = ['M',num2str(i)];  end

paradigms = fields(stimuli);

% loop paradigms × subjects
for ip = 1:numel(paradigms)
    paradigm = paradigms{ip};
    for isub = 1:numel(subject_alias)
        subj = subject_alias{isub};
        
        % filename for this run
        fname = fullfile(runname, sprintf('%s_%s.mat', paradigm, subj));
        
        % skip if already done
        if exist(fname,'file')
            fprintf('Skipping %s / %s (already exists)\n', paradigm, subj);
            continue;
        end
        
        % otherwise simulate and save
        fprintf('Running %s / %s ...\n', paradigm, subj);
        model = COIN;
        model.perturbations = stimuli.(paradigm);
%         model = assign_coin_params(model, subj, paramsTable);
        model = assign_coin_params(model, subj, mappedTable);
        
        model.runs = 10;
        model.particles=100;
        model.max_cores = feature('numcores');
        model.plot_average_state = true;
%         model.plot_state_given_context           = true;
%         model.plot_predicted_probabilities       = true;
%         model.plot_state                         = true;
%         model.plot_global_transition_probabilities = true;
%         model.plot_local_transition_probabilities  = true;
%         model.plot_responsibilities              = true;
        
        res = model.simulate_COIN;
        
        save(fname, 'res');
        close all;
    end
end
