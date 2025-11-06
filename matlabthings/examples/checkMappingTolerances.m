function [tolSummary, relErrs] = checkMappingTolerances(paramsTable, POpt, matchTbl, baseParam, tol)
% CHECKMAPPINGTOLERANCES  Given a “base” mapping (e.g. sigma_a) checks
%   relative‐errors across *all* parameters for the known 40 subjects.
%
%   [tolSummary, relErrs] = ...
%       CHECKMAPPINGTOLERANCES(paramsTable, POpt, matchTbl, baseParam, tol)
%
% Inputs:
%   paramsTable : 43×9 table with vars:
%                 participant, '\sigma_q','\mu_a',...,'\alpha_e'
%   POpt        : struct with fields
%                 alphaQ, a, qsd, aprecsd, alpha, motorsd, rho, dprecsd
%   matchTbl    : 40×8 table from buildMatchTable, rows in the order
%                 S1…S8, E1…E8, M1…M24, with VarNames matching
%                 matlab.lang.makeValidName of the eight param names.
%   baseParam   : string, one of the original column names, e.g. '\sigma_a'
%   tol         : scalar or 1×8 vector of allowed rel-error (e.g. 0.05)
%
% Outputs:
%   tolSummary  : 8×5 table with columns
%                 Param       MeanRelErr   MaxRelErr   PctWithinTol   Tol
%   relErrs     : 40×8 matrix of abs((table-value/POpt-value)–1),
%                 rows in S1…M24 order

  %--- define canonical subject order
  subjList = [ ...
    arrayfun(@(i) sprintf('S%d',i),1:8,'uni',false), ...
    arrayfun(@(i) sprintf('E%d',i),1:8,'uni',false), ...
    arrayfun(@(i) sprintf('M%d',i),1:24,'uni',false) ...
  ]';
  nSub = numel(subjList);

  %--- parameter names and POpt field names
  colNames   = {'\sigma_q','\mu_a','\sigma_a','\sigma_d', ...
                '\alpha','\sigma_m','\rho','\alpha_e'};

POptFields = {'qsd',  'a',     'aprecsd_inv1',    'dprecsd_inv1', ...
    'alpha',   'motorsd','rho',    'alphaQ'};
  validNames = matlab.lang.makeValidName(colNames);
  nParam     = numel(colNames);

  %--- expand scalar tol
  if isscalar(tol)
    tol = tol * ones(1,nParam);
  elseif numel(tol)~=nParam
    error('tol must be scalar or length %d',nParam)
  end

  %--- figure out which column of matchTbl gives our “base” mapping
  vBase = matlab.lang.makeValidName(baseParam);
  colIdx = find(strcmp(validNames,vBase),1);
  if isempty(colIdx)
    error('baseParam ''%s'' not one of %s', baseParam, strjoin(colNames,', '))
  end
  mapping = matchTbl{:, validNames{colIdx}};   % should be nSub×1

  %--- preallocate output
  relErrs = nan(nSub,nParam);

  %--- loop over each subject in canonical order
  for i = 1:nSub
    subj = subjList{i};
    % find row in paramsTable
    if iscell(paramsTable.participant)
      ridx = find(strcmp(paramsTable.participant, subj),1);
    else
      ridx = find(paramsTable.participant==subj,1);
    end
    if isempty(ridx)
      error('Subject %s not found in paramsTable', subj)
    end

    % and the POpt index coming from our mapping
    pidx = mapping(i);

    % compute per‐param rel‐error
    for p = 1:nParam
      v_tab = paramsTable{ridx, colNames{p}}; 
      v_pop = POpt.(POptFields{p})(pidx);
      relErrs(i,p) = abs(v_tab / v_pop - 1);
    end
  end

  %--- build summary stats
  meanErr   = mean(relErrs,1);
  maxErr    = max(relErrs,[],1);
  pctWithin = sum(relErrs <= tol,1) ./ nSub * 100;

  tolSummary = table( colNames', meanErr', maxErr', pctWithin', tol', ...
    'VariableNames', {'Param','MeanRelErr','MaxRelErr','PctWithinTol','Tol'} );
end


