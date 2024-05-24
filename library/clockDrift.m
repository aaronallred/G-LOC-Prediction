% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: visualizes all time-related variables in one plot
% INPUTS:   T: partitioned dataset
%           chooseID: inputted IDs to analyze
% OUTPUTS: 

% THOUGHTS FOR TOMORROW: Plot mean as a line. Plot std as well. Threshold
% for when acceleration goes up by XYZ value (for now, 0.1G, spin up rate?)
% Plot for each trial
% Plot for each sensor
% In plotting, zoom in to see just GOR and just ROR. Effectively plot twice
% and save each of these. 
% Check out Kieran's matlab snippet

%function [] = clockDrift(T,vars,chooseID)
%for i = 1:length(chooseID)
chooseID = "01-01";

    clear keyTimes
    % Split data
    Ti = T(T.trial_id == chooseID,:);
    % Checking how many string compare true values:
    keyTimesInd = find(~strcmp(Ti.event_validated, 'NO VALUE'));
    % Index into T with key time indices
    keyTimes.time = Ti.Time_s_(keyTimesInd);
    keyTimes.event_validated = Ti.event_validated(keyTimesInd);
    keyTimes.Spin_Centrifuge = Ti.Spin_Centrifuge(keyTimesInd);
    keyTimes = struct2table(keyTimes);
    % Find when GOR and ROR occur in this data
    GORstart = 0;
    RORstart = 0;
    for j=1:height(keyTimes)
    if contains(keyTimes{j,2}, 'begin GOR')
        GORstart = keyTimes{j,1};
    end
    if contains(keyTimes{j,2}, 'begin ROR')
        RORstart = keyTimes{j,1};
    end
    end
    if GORstart == 0
        disp("No listed GOR start time in this data")
    end
    if RORstart == 0
        disp("No listed ROR start time in this data")
    end

    % Split variables to plot
    % Variables to plot against - truth acceleration and truth time
    truthAcc = Ti.magnitude_Centrifuge;
    t = Ti.Time_s_;
    % First plot experimental acceleration
    expAcc = Ti.magnitude_Tobii;

    % Normalize variables
    ZtruthAcc = (truthAcc - min(truthAcc))/(max(truthAcc)-min(truthAcc));
    expAcc = (expAcc - min(expAcc))/(max(expAcc)-min(expAcc));

    % Smooth variables with Savitz-Golay to remove noise
    framelen = 51;
    truthFilt = sgolayfilt(ZtruthAcc, 3, framelen);
    % remove baseline Tobii noise?
    expAcc = expAcc-mean(expAcc(1:100));
    expAccFilt = sgolayfilt(expAcc, 3, framelen);

    % Calculate Moving Average Mean 
    % 1 second averaging and std calculation
    % Define moving average time window in seconds
    tWindow = 1; % [sec]
    % Find number of samples that represents window time
    N = find(t==tWindow);

    % Instead of looping through, make more efficient by reshaping and
    % taking mean and std of the columns
    % First, if length isn't a multiple of N, chop off the end. We
    % will cut off the end of the data since end of data doesn't matter due
    % to its occurrence after the last validated acceleration correlation
    % point
    shortExpAcc = expAccFilt(1:end-rem(length(expAccFilt),N));
    shortTruthAcc = truthFilt(1:end-rem(length(expAccFilt),N));
    shortt = t(1:end-rem(length(expAccFilt),N));

    % Now reshape into a matrix of N rows by (length of short vector)/N
    matrixExpAcc = reshape(shortExpAcc,N,[]);
    matrixTruthAcc = reshape(shortTruthAcc,N,[]);
 
    % Calculate mean and std for each reshaped matrix
    avgExpAcc = mean(matrixExpAcc);
    stdExpAcc = std(matrixExpAcc);
    avgTruthAcc = mean(matrixTruthAcc);
    stdTruthAcc = std(matrixTruthAcc);

     % Keep every N values of t
    plott = linspace(min(shortt),max(shortt),length(avgExpAcc));

    % Identify when acceleration goes above threshold value
    % TBD

%% Plotting results
    figure
    hold on 
    title("Tobii vs Centrifuge Acceleration Comparison")
    xlabel("Aligned Time [s]")
    ylabel("Measured Acceleration [G]")
    plot(t,expAccFilt)
    plot(t,truthFilt)
    errorbar(plott, avgExpAcc,stdExpAcc)
    errorbar(plott, avgTruthAcc,stdTruthAcc)
    xline(GORstart, '-r','LineWidth',2)
    xline(RORstart, '-b','LineWidth',2)
    legend("Tobii Acceleration","Centrifuge Ground Truth","Averaged Tobii Acceleration",...
        "Averaged Truth Acceleration","GOR Start","ROR Start","Location","northwest")
    hold off

%end
%end