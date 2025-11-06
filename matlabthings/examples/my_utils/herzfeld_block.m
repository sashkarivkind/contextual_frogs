function hz = herzfeld_block(z, T_pert, T_wash, P1, P2, P0, tau)
% HERZFELD_BLOCK  Build a Herzfeld perturbation‐washout sequence
%
%   hz = herzfeld_block(z) uses defaults T_pert=30, T_wash=10, and tau=1,
%   but requires P1, P2 and P0 to be provided (or added to the call).
%
%   hz = herzfeld_block(z, T_pert, T_wash, P1, P2, P0, tau)
%   explicitly sets all parameters.
%
%   Inputs:
%     z      – probability p_stay for flip_sequence
%     T_pert – length of the perturbation epoch (default 30)
%     T_wash – length of the washout epoch (default 10)
%     P1     – outcome level for flip_sequence’s “up” state
%     P2     – outcome level for flip_sequence’s “down” state
%     P0     – baseline outcome level during washout
%     tau    – repetition factor for each element in the final vector
%
%   Output:
%     hz     – (column) vector containing:
%                [flip_sequence; NaN; NaN; P0×ones(T_wash,1); NaN; P1; NaN],
%              with each element repeated tau times.

    % --- handle defaults ---
    if nargin < 7 || isempty(tau),     tau     = 1;   end
    if nargin < 6 || isempty(P0),      error('P0 must be specified'); end
    if nargin < 5 || isempty(P2),      error('P2 must be specified'); end
    if nargin < 4 || isempty(P1),      error('P1 must be specified'); end
    if nargin < 3 || isempty(T_wash),  T_wash  = 10;  end
    if nargin < 2 || isempty(T_pert),  T_pert  = 30;  end

    % --- build the block ---
    seq      = flip_sequence(z, T_pert, 1, P1, P2);       % perturbation
    washout  = P0 * ones(T_wash, 1);                      % washout
    tail     = [nan; P1; nan];                            % final 3 samples
    
    % concatenate as a column vector
    hz_block = [ seq; 
                 nan(2,1);      % two NaNs
                 washout; 
                 tail ];
    
    % repeat each element tau times
    hz = repelem(hz_block, tau);
end


function seq = flip_sequence(p_stay, Tmax, tau, P1, P2)
% FLIP_SEQUENCE  Generate a sequence of binary flips with hold‐times
%
%   seq = flip_sequence(p_stay, Tmax) uses defaults tau=1, P1=0, P2=1.
%   seq = flip_sequence(p_stay, Tmax, tau) uses defaults P1=0, P2=1.
%   seq = flip_sequence(p_stay, Tmax, tau, P1, P2) sets all parameters.
%
%   Inputs:
%     p_stay – probability of remaining in the same state
%     Tmax   – total number of timesteps (must be a multiple of tau)
%     tau    – number of sub‐timesteps per flip block
%     P1     – value for “polarity = 1” (default 0)
%     P2     – value for “polarity = 0” (default 1)
%
%   Output:
%     seq    – a column vector of length Tmax, containing P1/P2 values
%              that flip according to p_stay and the tau constraint.

    % --- handle defaults ---
    if nargin < 5 || isempty(P2),   P2     = 1;   end
    if nargin < 4 || isempty(P1),   P1     = 0;   end
    if nargin < 3 || isempty(tau),  tau    = 1;   end

    % check arguments
    if mod(Tmax, tau) ~= 0
        error('flip_sequence:badArgs', ...
              'Tmax must be an integer multiple of tau');
    end
    if p_stay < 0 || p_stay > 1
        error('flip_sequence:badArgs', ...
              'p_stay must be in [0, 1]');
    end

    % number of independent flips
    nBlocks = Tmax / tau;

    % draw random flips: true means “flip state” at that block
    flips = rand(nBlocks, 1) < (1 - p_stay);

    % cumulative XOR to build polarity (0→1→0→1 as flips occur)
    polarity = false(nBlocks, 1);
    for k = 1:nBlocks
        if k == 1
            polarity(k) = flips(k);
        else
            polarity(k) = xor(polarity(k-1), flips(k));
        end
    end

    % expand each block by tau
    polarity_full = repelem(polarity, tau);
    polarity_full = xor(rand<0.5,polarity_full);
    % map logical polarity to values P1, P2
    seq = (P1 * double(polarity_full) + P2 * double(~polarity_full));
end

