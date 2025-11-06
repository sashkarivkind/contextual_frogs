max_tries = 10000;
funs ={@randnumtable, @randnumtablem};
fun_aliases = {'mex','matlab'};
results = struct();
% paramsets = {{0,50},{2,80},{10,100},{3,200},{20,30},{5,200},{10,250},{15,300},{15,350}};
paramsets = {{5,20},{20,30},{10,100},{3,200},{10,250},{5,300},{15,300},{10,350},{30,350}};

for ff = 1:length(funs)
    fun = funs{ff};
    fun_alias = fun_aliases{ff}
    results.(fun_alias) = {};
    for pp = 1:length(paramsets)
        paramset = paramsets{pp};
        p1 = paramset{1};
        p2 = paramset{2};
        results.(fun_alias){end+1} = [];

        for tt = 1:max_tries
                results.(fun_alias){end}(end+1) = fun(p1,p2);
        end
    end
 end

%%


% for 
bin_ctrs = 0:100;
figure;
    for pp = 1:length(paramsets)

subplot(3,3,pp)
o_mex = hist(results.mex{pp}, bin_ctrs-0.5); 
o_mat = hist(results.matlab{pp}, bin_ctrs-0.5); 

stem(bin_ctrs,o_mex/max_tries,'x')
hold on;
stem(bin_ctrs,o_mat/max_tries,'o')
text(15,0.1,sprintf('w=%d, m=%d\n',paramsets{pp}{:}));
% ylim([0,0.5]);
    end
    subplot(3,3,1)
legend(fun_aliases{:})
subplot(3,3,7)
xlabel('num table')
ylabel('est probability')

figure;
    for pp = 1:length(paramsets)

subplot(3,3,pp)
o_mex = hist(results.mex{pp}, bin_ctrs-0.5); 
o_mat = hist(results.matlab{pp}, bin_ctrs-0.5); 

semilogy(bin_ctrs,o_mex/max_tries,'x')
hold on;
semilogy(bin_ctrs,o_mat/max_tries,'o')
xlim([0,max(bin_ctrs)])
    end
    subplot(3,3,1)
legend(fun_aliases{:})

subplot(3,3,7)
xlabel('num table')
ylabel('est probability')

% figure; hist(results.mex{1},50); 
% hold on; 
% figure;
% hist(results.matlab{1},500,'r');