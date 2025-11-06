function model = assign_coin_params(model, subj, paramsTable)
% ASSIGN_COIN_PARAMS  Load one subject’s row from paramsTable into a COIN object.
%
%   model = ASSIGN_COIN_PARAMS(model, subj, paramsTable)
%   finds the row where paramsTable.participant==subj, then overwrites:
%     sigma_process_noise, prior_mean_retention,
%     prior_precision_retention, prior_precision_drift,
%     alpha_context, rho_context,
%     sigma_motor_noise, alpha_cue.
%


  % find matching row
  if iscell(paramsTable.participant)
    mask = strcmp(paramsTable.participant, subj);
  else
    mask = paramsTable.participant == subj;
  end
  idx = find(mask,1);
  if isempty(idx)
    error("Subject '%s' not found in paramsTable.", subj);
  end

  % now pull out each field
  sq    = paramsTable{idx, '\sigma_q'};   % process‐noise SD
  mua   = paramsTable{idx, '\mu_a'};      % prior mean retention
  siga  = paramsTable{idx, '\sigma_a'};   % SD retention prior
  sigd  = paramsTable{idx, '\sigma_d'};   % SD drift prior
  alp   = paramsTable{idx, '\alpha'};      % alpha_context
  rho   = paramsTable{idx, '\rho'};        % rho_context
  sigm  = paramsTable{idx, '\sigma_m'};   % motor‐noise SD
  alpe  = paramsTable{idx, '\alpha_e'};   % alpha_cue

  % assign into model
  model.sigma_process_noise       = sq;
  model.prior_mean_retention      = mua;
  model.prior_precision_retention = 1./(siga.^2);
  model.prior_precision_drift     = 1./(sigd.^2);
  model.alpha_context             = alp;
  model.rho_context               = rho;
  model.sigma_motor_noise         = sigm;
  model.alpha_cue                 = alpe;
end
