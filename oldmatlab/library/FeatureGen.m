% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: Provides an overview plot per trial
% INPUTS:   T: partitioned dataset
%           chooseID: inputted IDs to analyze
%           chooseVar: variable to plot for overview. Chosen in wrapper
%           saveFlag: binary 1/0 for if plots should be saved
%           outPath: path for figure saving
% OUTPUTS:  none. Saves .png files for each trial overview. Can save .fig
% files as well by uncommenting those lines of code at the bottom of the
% fcn

function [PredictorTable] = FeatureGen(T, chooseID, chooseVar,binFreq, width,gap)
for i = 1:length(chooseID)
    clear keyTimes
    clear TiEventValid
    Ti = T(T.trial_id == chooseID(i),:);
    % X axis variable
    xVar = "Time_s_";
    % Acceleration y variable
    accVar = "DoubleI_O_actualGr_Centrifuge";
    % Find key times throughout the trial - ie anytime event is not 'NO VALUE'
    % What are the possible values of "event_validated"?
    TiEventValid(:,1) = unique(Ti.event_validated);
    TiEventValid(:,2) = num2cell(groupcounts(Ti.event_validated));
    fprintf("Validated Events for %s:      # Occurrences:\n",chooseID(i))
    disp(TiEventValid)
    % Checking how many string compare true values:
    keyTimesInd = find(~strcmp(Ti.event_validated, 'NO VALUE'));
    % Index into T with key time indices
    keyTimes.time = Ti.Time_s_(keyTimesInd);
    keyTimes.event_validated = Ti.event_validated(keyTimesInd);
    keyTimes.Spin_Centrifuge = Ti.Spin_Centrifuge(keyTimesInd);
    keyTimes = struct2table(keyTimes);


    % Assign GLOC Labels Based on Events
    GLOC_labels = zeros(length(Ti.Time_s_),1);
   
    if any(contains(keyTimes.event_validated,'GLOC'))
        % Find which index has GLOC
        N = find(strcmp(keyTimes.event_validated, 'GLOC'));
        % Plot GLOC line
        GLOCstart = keyTimes.time(N);
        GLOC_labels(Ti.Time_s_>=GLOCstart(1)) = 1;
        % xline(table2array(keyTimes(N,1)),'--r','LineWidth',2)
    end

    if any(contains(keyTimes.event_validated,'return to consciousness'))
        % Find which index has return to consciousness
        N = find(strcmp(keyTimes.event_validated, 'return to consciousness'));
        % Plot return to consciousness line
        GLOCend = keyTimes.time(N);
        GLOC_labels(Ti.Time_s_>=GLOCend(1)) = 0;
        % xline(table2array(keyTimes(N,1)),'--m','LineWidth',2)
    end
    
    % Grab Predictor from Subject Trial Table, Ti
    Predictor = Ti{:,chooseVar};

    % Let's now Baseline the predictor
    Rest = 10; % first seconds of trial is rest
    BL = mean(Predictor(Ti.Time_s_<Rest));
    Predictor = Predictor/BL;

    % Now let's compute binned predictors and labels
    sampleFreq = 25;
    Frequency = binFreq*sampleFreq;
    binSize = round(length(Ti.Time_s_)/Frequency);
    
    predictor = zeros(binSize,1);
    label = predictor;
    for b = 1:binSize
        pstartspot = (b-1)*Frequency-round(width*sampleFreq/2);
        pendspot   = (b-1)*Frequency+round(width*sampleFreq/2);
        lstartspot = pstartspot + gap*sampleFreq;
        lendspot   = pendspot + gap*sampleFreq;
        
        % Trim Bins so that they do not exceed array limits
        [pstartspot, pendspot] = ...
            trimBin(pstartspot, pendspot, length(Predictor));
        [lstartspot, lendspot] = ...
            trimBin(lstartspot, lendspot, length(GLOC_labels));

        % Compute predictor as the mean value in the bin
        predictor(b) = mean(Predictor(pstartspot:pendspot));

        % Assign label if GLOC is present in bin
        if sum(GLOC_labels(lstartspot:lendspot)==1) > 0
            occur = 1;
        else
            occur = 0;
        end
        label(b) = occur;
    end

    trial_id = repmat(chooseID(i),[binSize 1]);
    if i == 1
        PredictorTable = table(predictor,label,trial_id);
    else
        NextTrialTable = table(predictor,label,trial_id);
        PredictorTable = [PredictorTable; NextTrialTable];
    end

% end chooseID loop
end
% end function
end