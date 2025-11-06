function scenarios = generate_herzfeld_scenarios(z_list, n_blocks, Pplus, Pminus, P0)
% GENERATE_HERZFELD_SCENARIOS  Create merged Herzfeld‐block sequences
%
%   scenarios = generate_herzfeld_scenarios(z_list, n_blocks, Pplus, Pminus, P0)
%
%   Returns a struct whose fields are of the form "herzfeld_z_x_y" (where
%   x_y is the probability z with the decimal point replaced by an underscore),
%   and whose values are long column vectors obtained by concatenating
%   `n_blocks` calls to `herzfeld_block` (using the default T_pert=30,
%   T_wash=10 and tau=1).
%
%   Inputs:
%     z_list    – vector of “p_stay” probabilities (e.g. [0.1,0.5,0.9])
%     n_blocks  – number of blocks to concatenate per z
%     Pplus     – P1 argument for herzfeld_block
%     Pminus    – P2 argument for herzfeld_block
%     P0        – P0 (washout level) argument for herzfeld_block
%
%   Output:
%     scenarios – struct with fields:
%                   scenarios.herzfeld_z_{z}
%                 each a column vector of length n_blocks * block_length

    %--- input validation ---
    if nargin < 5
        error('Usage: generate_herzfeld_scenarios(z_list, n_blocks, Pplus, Pminus, P0)');
    end
    if ~isvector(z_list) || isempty(z_list)
        error('z_list must be a nonempty vector of probabilities.');
    end
    if ~isscalar(n_blocks) || n_blocks < 1 || floor(n_blocks)~=n_blocks
        error('n_blocks must be a positive integer scalar.');
    end
    
    scenarios = struct();
    
    for i = 1:numel(z_list)
        z = z_list(i);
        % create a valid field name, e.g. 'herzfeld_z_0_1' for z=0.1
        z_str   = regexprep(sprintf('%.3g', z), '\.', '_');
        fldname = ['herzfeld_z_' z_str];
        
        % build and concatenate n_blocks of herzfeld_block
        merged_seq = [];
        for b = 1:n_blocks
            hz = herzfeld_block(z, [], [], Pplus, Pminus, P0);  % uses default T_pert, T_wash, tau
            merged_seq = [merged_seq; hz(:)];                   % ensure column‐vector
        end
        
        % assign to struct
        scenarios.(fldname) = merged_seq;
    end
end
