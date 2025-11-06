function matchTbl = buildMatchTable(paramsTable, POpt, toreturn)
% BUILDMATCHTABLE   For each subject and each COIN parameter, find which
%   POpt vector‐index best matches the value in paramsTable.
%
%   matchTbl = BUILDMATCHTABLE(paramsTable, POpt)
%
% Inputs:
%   paramsTable : 43×9 table with vars:
%                 participant (cell of 'S1'…'M24'), '\sigma_q','\mu_a',...
%   POpt        : struct with fields
%                 alphaQ, a, qsd, aprecsd, alpha, motorsd, rho, dprecsd
%
% Output:
%   matchTbl    : 40×8 table.  RowNames = {'S1',…,'M8','E1',…,'E8','M1',…,'M24'},
%                 VariableNames = {'sigma_q','mu_a','sigma_a','sigma_d',...
%                                  'alpha','sigma_m','rho','alpha_e'}
%                 matchTbl{i,p} = index j ∈ 1:40 so that
%                   abs( paramsTable(subj_i,p) / POpt.(field_p)(j) - 1 )
%                 is minimized.

% list your 40 subjects in the exact order you want
subs = [ ...
    arrayfun(@(i) sprintf('S%d',i), 1:8, 'uni',0), ...
    arrayfun(@(i) sprintf('E%d',i), 1:8, 'uni',0), ...
    arrayfun(@(i) sprintf('M%d',i), 1:24,'uni',0) ...
    ];
nSub = numel(subs);

% the eight table‐column names (with backslashes) and corresponding POpt fields
colNames   = {'\sigma_q','\mu_a','\sigma_a','\sigma_d', ...
    '\alpha','\sigma_m','\rho','\alpha_e'};
POptFields = {'qsd',  'a',     'aprecsd_inv1',    'dprecsd_inv1', ...
    'alpha',   'motorsd','rho',    'alphaQ'};
%   POptFields = {'alphaQ',  'a',     'aprecsd',    'dprecsd', ...
%                 'alpha',   'motorsd','rho',    'aprecsd'};

% preallocate idx‐matrix
idx = nan(nSub, numel(colNames));

for i = 1:nSub
    % find the row in paramsTable for this subject
    row = find( strcmp(paramsTable.participant, subs{i}), 1 );
    if isempty(row)
        error("Subject %s not found in paramsTable.", subs{i});
    end

    for p = 1:numel(colNames)
        v_tab = paramsTable{row, colNames{p}};        % scalar
        v_pop = POpt.(POptFields{p})(:);              % 40×1
        [tolbest, jbest] = min( abs(v_tab./v_pop - 1) );    % best match
        if strcmp(toreturn,'argmins')
            idx(i,p) = jbest;
        elseif strcmp(toreturn,'mins')
            idx(i,p) = tolbest;
        end
    end
end

% build output table
matchTbl = array2table(idx, ...
    'RowNames',        subs, ...
    'VariableNames',   matlab.lang.makeValidName(colNames) ...
    );
end
