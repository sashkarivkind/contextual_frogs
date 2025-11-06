function model = assign_coin_params_v2(model, subj_idx, paramsStruct)
% ASSIGN_COIN_PARAMS_V2

  i= subj_idx;
  % assign into model
  model.sigma_process_noise       = paramsStruct.qsd(i);
  model.prior_mean_retention      = paramsStruct.a(i);
  model.prior_precision_retention = paramsStruct.aprecsd(i)^2;
  model.prior_precision_drift     = paramsStruct.dprecsd(i).^2;
  model.alpha_context             = paramsStruct.alpha(i);
  model.rho_context               = paramsStruct.rho(i);
  model.sigma_motor_noise         = paramsStruct.motorsd(i);
  model.alpha_cue                 = paramsStruct.alphaQ(i);
end
