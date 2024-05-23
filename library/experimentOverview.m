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

function [] = experimentOverview(T_full)
% 1) Trial_ID Analysis
% Number of trials
fprintf("Number of unique trial IDs: %1.0f\n",numel(unique(T_full.trial_id)))

% Number of subjects
fprintf("Number of unique subjects: %1.0f\n",numel(unique(T_full.subject)))

% Number of trials per subject
tPerS = zeros(length(unique(T_full.subject)),2);
% Create unique trials of type double by splitting unique trial ids and
% converting to double
unqTrials = str2double(strrep(split(unique(T_full.trial_id),"-"),',','.'));
tPerS(:,1) = unique(T_full.subject);
for i = 1:numel(unique(T_full.subject))
    % Number per subject is number of rows of unique trial ids for this
    % subject id
    tPerS(i,2) = size(unqTrials(find(unqTrials(:,1)==i),:),1);
end
disp("Number of trials per subject:")
disp("Subject:   # Trials:")
disp(tPerS)
% We have 13 subjects. Each subject performed 6 trials, besides subject 11,
% who performed 3 trials

% 2) Event Column Analysis
% What are the possible values of "event"?
unqEvent(:,1) = unique(T_full.event);
unqEvent(:,2) = num2cell(groupcounts(T_full.event));
disp("Event Identifier:             # Occurrences:")
disp(unqEvent)

% 3) Event Validated Analysis
% What are the possible values of "event_validated"?
unqEventValid(:,1) = unique(T_full.event_validated);
unqEventValid(:,2) = num2cell(groupcounts(T_full.event_validated));
disp("Event_Validated Identifier:           # Occurrences:")
disp(unqEventValid)

% 4) G-LOC Analysis
% We know from Event Validated Analysis that 45 subjects GLOCed
% Which subjects did during which trials?
ind = find(strcmp(T_full.event_validated, 'GLOC'));
TGLOC.trial_id = T_full.trial_id(ind);
TGLOC.subject = T_full.subject(ind);
TGLOC.trial = T_full.trial(ind);
TGLOC.spin = T_full.Spin_Centrifuge(ind);
TGLOC.time = T_full.Time_s_(ind);
TGLOC = struct2table(TGLOC);
% Number of GLOCs per subject
glocPerSub = zeros(length(unique(T_full.subject)),2);
% Create unique trials of type double by splitting unique trial ids and
% converting to double
unqGLOC = str2double(strrep(split(unique(TGLOC.trial_id),"-"),',','.'));
glocPerSub(:,1) = unique(TGLOC.subject);
for i = 1:numel(unique(TGLOC.subject))
    % Number per subject is number of rows of unique trial ids for this
    % subject id
    glocPerSub(i,2) = size(unqGLOC(find(unqGLOC(:,1)==i),:),1);
end
disp("Subject:   # G-LOCs:")
disp(glocPerSub)
% Note that this doesn't include the seemingly erroneous second GLOC on
% trial 03-01

% 5) Spin Centrifuge Analysis
% Number of spin types
fprintf("Number of unique spin types: %1.0f\n", ...
    numel(unique(T_full.Spin_Centrifuge)))
disp(unique(T_full.Spin_Centrifuge))
unqSpin(:,1) = unique(T_full.Spin_Centrifuge);
unqSpin(:,2) = num2cell(groupcounts(T_full.Spin_Centrifuge));

end