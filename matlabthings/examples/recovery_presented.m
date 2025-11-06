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
    these_results = results{ii};
    figure(f1);
    subplot(4,4,ii);
%     hold on;
 plot(these_results.plots.average_state);
 hold on;
%  plot(these_results.runs{1}.motor_output);
%  plot(these_results.runs{2}.motor_output);
 plot(mean([these_results.runs{1}.motor_output'; these_results.runs{2}.motor_output']),'.');
%      figure(f2);
%     subplot(4,4,ii);
%  plot(these_results.plots.predicted_probabilities);
%  plot(these_results.plots.known_context_responsibilities);
%  plot(these_results.plots.known_context_responsibilities);
end

% figure(f1); clf;
% figure(f2); clf;
% for ii=1:8 %length(results)
for ii=9:16 %length(results)
    these_results = results{ii};
    figure;
%     subplot(4,4,ii);
%     hold on;
 plot(these_results.plots.average_state, '.');

 plot(these_results.plots.average_state, '.');
 xlim([0 340])
 ylim([-1.6,1.6])
%  text(0,-1,['myS',num2str(ii)]);
 text(0,-1,['myE',num2str(ii-8)]);
 hold on;
 set(gcf,'InnerPosition', [488 660.2000 214.6000 101.8000]);
 set(gcf, 'OuterPosition', [481 653 228.8000 191.2000]);
%  plot(these_results.runs{1}.motor_output);
%  plot(these_results.runs{2}.motor_output);
%  plot(mean([these_results.runs{1}.motor_output'; these_results.runs{2}.motor_output']),'.');
%      figure(f2);
%     subplot(4,4,ii);
%  plot(these_results.plots.predicted_probabilities);
%  plot(these_results.plots.known_context_responsibilities);
%  plot(these_results.plots.known_context_responsibilities);
end

%InnerPosition: [488 660.2000 214.6000 101.8000]
%OuterPosition: [481 653 228.8000 191.2000]