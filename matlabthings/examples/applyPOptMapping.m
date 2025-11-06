function mappedTable = applyPOptMapping(paramsTable, POpt, matchTbl, baseParam)
%APPLYPOPTMAPPING  Replace parameters by POpt via matchTbl.
%   If baseParam is provided, uses that single mapping column.
%   If baseParam is omitted, determines mapping per row by majority vote
%   across all mapping columns, warns if majority <75%, and errors
%   if mapping is not one-to-one.
%
%   mappedTable = applyPOptMapping(paramsTable,POpt,matchTbl[,baseParam])

  %--- parameter names and corresponding POpt fields
  colNames   = {'\sigma_q','\mu_a','\sigma_a','\sigma_d', ...
                '\alpha','\sigma_m','\rho','\alpha_e'};
  POptFields = {'qsd','a','aprecsd_inv1','dprecsd_inv1', ...
                'alpha','motorsd','rho','alphaQ'};
  validNames = matlab.lang.makeValidName(colNames);

  %--- determine mapping vector
  if nargin < 4 || isempty(baseParam)
    % Majority vote mapping
    mapVarNames = validNames;  % expects matchTbl vars matching these
    missing = setdiff(mapVarNames, matchTbl.Properties.VariableNames);
    if ~isempty(missing)
      error('applyPOptMapping:missingVars', ...
            'matchTbl missing variables: %s', strjoin(missing,', '));
    end
    mapMat = matchTbl{:, mapVarNames};
    [nSub, nMapParam] = size(mapMat);
    mapping = nan(nSub,1);
    for i = 1:nSub
      [uniqVals,~,idx] = unique(mapMat(i,:));
      counts = histcounts(idx,1:(numel(uniqVals)+1));
      [maxCount,mi] = max(counts);
      if maxCount < 0.75 * nMapParam
        warning('Row %d: majority count %d of %d < 75%%', i, maxCount, nMapParam);
      end
      mapping(i) = uniqVals(mi);
    end
    if numel(unique(mapping)) < numel(mapping)
      error('applyPOptMapping:nonUniqueMapping', ...
            'Majority-based mapping is not one-to-one.');
    end
  else
    % Single-column mapping via baseParam
    baseValid = matlab.lang.makeValidName(baseParam);
    if ~ismember(baseValid, validNames)
      error('applyPOptMapping:badParam', ...
            'baseParam ''%s'' invalid. Choose from: %s', ...
            baseParam, strjoin(colNames,', '));
    end
    mapVar = ['x_' baseValid];
    if ~ismember(mapVar, matchTbl.Properties.VariableNames)
      error('applyPOptMapping:badParam', ...
            'matchTbl lacks variable ''%s''.', mapVar);
    end
    mapping = matchTbl{:, mapVar};
  end

  %--- assemble replacement values
  subjList = [ ...
    arrayfun(@(i)sprintf('S%d',i),1:8,'uni',false), ...
    arrayfun(@(i)sprintf('E%d',i),1:8,'uni',false), ...
    arrayfun(@(i)sprintf('M%d',i),1:24,'uni',false)
  ]';
  nSub = numel(subjList);
  nParam = numel(validNames);
  replacement_values = nan(nSub,nParam);

  for i = 1:nSub
    subj = subjList{i};
    if iscell(paramsTable.participant)
      ridx = find(strcmp(paramsTable.participant,subj),1);
    else
      ridx = find(paramsTable.participant==subj,1);
    end
    if isempty(ridx)
      error('applyPOptMapping:missingSubject', 'Subject %s not found.', subj);
    end
    pidx = mapping(i);
    for p = 1:nParam
      replacement_values(i,p) = POpt.(POptFields{p})(pidx);
    end
  end

  %--- build output table
  T = array2table(replacement_values, ...
                'RowNames', subjList, ...
                'VariableNames', colNames);
  T.participant = T.Properties.RowNames;
  T = movevars(T,'participant','Before',1);
  T.Properties.RowNames = {};
  mappedTable = T;
end
