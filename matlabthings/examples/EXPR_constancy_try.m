function OUTPUT_WM=EXPR_constancy_try(zz,n,n_tries)

% paradigm based on Keisler, A. & Shadmehr, R. A shared resource between declarative memory and motor memory. J. Neurosci. 30, 14817–14823 (2010).

obj_WM = COIN;
T=100;
a = 30;
% obj_WM.perturbations = [zeros(1,192) ones(1,192) zeros(1,192) ones(1,192) NaN(1,192)];
% obj_WM.perturbations = [zeros(1,T) ones(1,T) zeros(1,T),...
%                         0.5*ones(1,T)  zeros(1,T) ones(1,T) NaN(1,T)];

% obj_WM.perturbations = [zeros(1,T) ones(1,T) zeros(1,T),...
%                         1-1/a*(1:a) zeros(1,T-a)  zeros(1,T) ones(1,T) NaN(1,T)];


obj_WM.perturbations = [];
sandwich = [nan,1,nan];
for i = 1:n_tries
    new_segment = [nan, nan, flipsequence(n,zz), zeros(1,10), sandwich];
    obj_WM.perturbations = [obj_WM.perturbations, new_segment];
end

obj_WM.runs = 10;
obj_WM.max_cores = feature('numcores');
obj_WM.plot_state_given_context = true;
obj_WM.plot_predicted_probabilities = true;
obj_WM.plot_state = true;

% for a working memory task performed between trials 596 and 597, set the
% predicted probabilities on trial 597 to the stationary context 
% probabilities
obj_WM.stationary_trials = 597;

OUTPUT_WM = obj_WM.simulate_COIN;

% hold on; plot(250-100* OUTPUT_WM.runs{1,2}.state_feedback,'g --', LineWidth=3)


end 
function x=flipsequence(n,z)
stays = rand(1,n) < z;
stays = -1+2*stays;
x =  cumprod(stays);
end
