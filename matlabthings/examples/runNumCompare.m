f1=91;
f2=92;


% figure(f1); clf;
% figure(f2); clf;
% for ii=1:length(results)
%     these_results = results{ii};
%     figure(f1);
%     subplot(4,4,ii);
% %     hold on;
%  plot(these_results.plots.average_state);
%      figure(f2);
%     subplot(4,4,ii);
% %  plot(these_results.plots.predicted_probabilities);
% %  plot(these_results.plots.known_context_responsibilities);
% %  plot(these_results.plots.known_context_responsibilities);
% end


figure(f1); clf;
% figure(f2); clf;
for ii=1:length(results)
    these_results100 = results{ii};
    these_results10 = ref_res.results{ii};
    figure(f1);
    subplot(4,4,ii);
%     hold on;
 hold on;

 plot(these_results100.plots.average_state,'r','linewidth',2);
 plot(these_results10.plots.average_state,'b','linewidth',1);

 if ii<9
 i_py = ii+8;
 else
     i_py=ii-8;
 end

%  foo = csvread(['../python_results/coin_pythonOutputStats_subj_',num2str(i_py),'_model.csv']);
%     errorInBetween(foo(:,1), mean(foo(:,2:11),2),std(foo(:,2:11),[],2),'linewidth',1);
% %     plot(foo(:,1), min(foo(:,2:11),2));
    ylim([-1.2,1.2]);
    xlim([0,340])
end
subplot(4,4,1)
legend('new results','reference')
subplot(4,4,13)
xlabel('trials')
ylabel('adaptation')
