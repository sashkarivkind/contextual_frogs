z_vec = [0.1,0.5,0.9];
n=30;
n_tries=25;
err_sen = {};

for ii = 1: length(z_vec)
    z = z_vec(ii);
    OUTPUT_WM = EXPR_constancy_try(z,n,n_tries);
    d = n+15;
    err_sen{ii} = OUTPUT_WM.plots.average_state(d:d:end)-OUTPUT_WM.plots.average_state(d-2:d:end);
end

figure(954);

for ii = 1: length(z_vec)
    plot(err_sen{ii})
    hold on;
end