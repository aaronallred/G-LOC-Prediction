% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: 1) count number of unique trials and subjects, 2) identify
% and count unique events, 3) identify and count unique validated events,
% 4) identify number of G-LOCs and number of G-LOCs per subject, 5) count
% number of spin types
% chooseID
% INPUTS:   T_full: full, uncleaned dataset
% OUTPUTS:  none - displays of final counts to command window

function [] = trialOverview(T, chooseID, chooseVar)
% X axis variable
xVar = "Time_s_";
% Acceleration y variable
accVar = "DoubleI_O_actualGr_Centrifuge";

% Split data into just this trial, trial i
Ti = T(T.trial_id == chooseID,:);
% Find key times throughout the trial - ie anytime event is not 'NO VALUE'
% What are the possible values of "event_validated"?
TiEventValid(:,1) = unique(Ti.event_validated);
TiEventValid(:,2) = num2cell(groupcounts(Ti.event_validated));
fprintf("Validated Events for %s:      # Occurrences:\n",...
    chooseID)
disp(TiEventValid)
% Checking how many string compare true values:
% num2cell(groupcounts(strcmp(Ti.event_validated, 'NO VALUE')))
keyTimesInd = find(~strcmp(Ti.event_validated, 'NO VALUE'));
% Index into Ti with key time indices
keyTimes.time = Ti.Time_s_(keyTimesInd);
keyTimes.event_validated = Ti.event_validated(keyTimesInd);
keyTimes.Spin_Centrifuge = Ti.Spin_Centrifuge(keyTimesInd);
keyTimes = struct2table(keyTimes);
% Truncate after GLOC occurrence to disregard return to consciousness times
% N = find(strcmp(keyTimes.event_validated, 'GLOC'));
% keyTimes = keyTimes(1:N,:);

% Create visualization
figure
hold on
title(strcat("Trial Visualization for ID ",chooseID))

% Plot key trial times - GOR start, GOR end, ROR start, GLOC
if any(contains(keyTimes.event_validated,'begin GOR'))
    % Find which index has begin GOR
    N = find(strcmp(keyTimes.event_validated, 'begin GOR'));
    % Plot begin GOR line
    xline(table2array(keyTimes(N,1)),'--g','LineWidth',2)
end
if any(contains(keyTimes.event_validated,'end GOR'))
    % Find which index has end GOR
    N = find(strcmp(keyTimes.event_validated, 'end GOR'));
    % Plot end GOR line
    xline(table2array(keyTimes(N,1)),'--b','LineWidth',2)
end
if any(contains(keyTimes.event_validated,'begin ROR'))
    % Find which index has begin ROR
    N = find(strcmp(keyTimes.event_validated, 'begin ROR'));
    % Plot begin ROR line
    xline(table2array(keyTimes(N,1)),'--g','LineWidth',2)
end
if any(contains(keyTimes.event_validated,'GLOC'))
    % Find which index has GLOC
    N = find(strcmp(keyTimes.event_validated, 'GLOC'));
    % Plot GLOC line
    xline(table2array(keyTimes(N,1)),'--r','LineWidth',2)
end
if any(contains(keyTimes.event_validated,'return to consciousness'))
    % Find which index has return to consciousness
    N = find(strcmp(keyTimes.event_validated, 'return to consciousness'));
    % Plot return to consciousness line
    xline(table2array(keyTimes(N,1)),'--m','LineWidth',2)
end
if any(contains(keyTimes.event_validated,'resume task'))
    % Find which index has resume task
    N = find(strcmp(keyTimes.event_validated, 'resume task'));
    % Plot resume task line
    xline(table2array(keyTimes(N,1)),'--y','LineWidth',2)
end

xlabel("Aligned Time [s]")
% Left axis: chosen variable above, 'chooseVar'
yyaxis left
ylabel(chooseVar)
plot(Ti,xVar,chooseVar,'LineWidth',2)
% Right axis: acceleration ground truth, 'accVar'
yyaxis right
ylabel("Centrifuge Acceleration [G]")
plot(Ti,xVar,accVar,'LineWidth',2)
legend(keyTimes.event_validated)
hold off

end