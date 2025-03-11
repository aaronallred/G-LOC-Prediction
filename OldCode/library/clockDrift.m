% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: visualizes all time-related variables in one plot
% INPUTS:   T: partitioned dataset
%           chooseID: inputted IDs to analyze

%function [] = clockDrift(T,vars,chooseID)
chooseID = "01-01";
for i = 1:length(chooseID)


    clear keyTimes
    % Split data
    Ti = T(T.trial_id == chooseID(i),:);
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
    sensorAcc = Ti.magnitude_Tobii;

    % Normalize variables
    ZtruthAcc = (truthAcc - min(truthAcc))/(max(truthAcc)-min(truthAcc));
    sensorAcc = (sensorAcc - min(sensorAcc))/(max(sensorAcc)-min(sensorAcc));

    % Smooth variables with Savitz-Golay to remove noise
    framelen = 51;
    truthFilt = sgolayfilt(ZtruthAcc, 3, framelen);
    % remove baseline Tobii noise?
    sensorAcc = sensorAcc-mean(sensorAcc(1:100));
    sensorAccFilt = sgolayfilt(sensorAcc, 3, framelen);

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
    shortSensorAcc = sensorAccFilt(1:end-rem(length(sensorAccFilt),N));
    shortTruthAcc = truthFilt(1:end-rem(length(sensorAccFilt),N));
    shortt = t(1:end-rem(length(sensorAccFilt),N));

    % Now reshape into a matrix of N rows by (length of short vector)/N
    matrixSensorAcc = reshape(shortSensorAcc,N,[]);
    matrixTruthAcc = reshape(shortTruthAcc,N,[]);
 
    % Calculate mean and std for each reshaped matrix
    avgSensorAcc = mean(matrixSensorAcc);
    stdSensorAcc = std(matrixSensorAcc);
    avgTruthAcc = mean(matrixTruthAcc);
    stdTruthAcc = std(matrixTruthAcc);

     % Keep every N values of t
    plott = linspace(min(shortt),max(shortt),length(avgSensorAcc));

    % Identify when acceleration goes above threshold value
    % TBD

    % Calculate slope of averaged data
    sensorSlope = gradient(avgSensorAcc);
    truthSlope = gradient(avgTruthAcc);

% %% Full Plot 
%     figure
%     hold on 
%     title("Tobii vs Centrifuge Acceleration Comparison")
%     xlabel("Aligned Time [s]")
%     ylabel("Measured Acceleration [G]")
%     plot(t,sensorAccFilt)
%     plot(t,truthFilt)
%     errorbar(plott, avgSensorAcc,stdSensorAcc)
%     errorbar(plott, avgTruthAcc,stdTruthAcc)
%     plot(plott, sensorSlope)
%     plot(plott, truthSlope)
%     xline(GORstart, '-r','LineWidth',2)
%     xline(RORstart, '-b','LineWidth',2)
%     legend("Tobii Acceleration","Centrifuge Ground Truth",...
%         "Averaged Tobii Acceleration","Averaged Truth Acceleration",...
%         "Sensor Slope","Truth Slope","GOR Start","ROR Start",...
%         "Location","northwest")
%     hold off
%% GOR Plots
% find X axes of 5 seconds surrounding GOR
GORxAxisStart = GORstart-2.5;
GORxAxisEnd = GORstart+2.5;
figure
    hold on 
    title(strcat...
        ("Tobii vs Centrifuge Acceleration Comparison: GOR, Trial ",...
        chooseID(i),", Averaged Per ",string(tWindow)," Second(s)"))
    xlabel("Aligned Time [s]")
    ylabel("Measured Acceleration [G]")
    plot(t,sensorAccFilt)
    plot(t,truthFilt)
    errorbar(plott, avgSensorAcc,stdSensorAcc)
    errorbar(plott, avgTruthAcc,stdTruthAcc)
    plot(plott, sensorSlope)
    plot(plott, truthSlope)
    xline(GORstart, '-r','LineWidth',2)
    legend("Tobii Acceleration","Centrifuge Ground Truth",...
        "Averaged Tobii Acceleration","Averaged Truth Acceleration",...
        "Sensor Slope","Truth Slope","GOR Start",...
        "Location","northwest")
    xlim([GORxAxisStart GORxAxisEnd])
    ylim([-0.01 0.05])
    hold off
    figGOR = gcf;

%% ROR Plots
% find X axes of 5 seconds surrounding ROR
RORxAxisStart = floor(RORstart)-3;

RORxAxisEnd = floor(RORstart)+2;
figure
    hold on 
    title(strcat...
        ("Tobii vs Centrifuge Acceleration Comparison: ROR, Trial ",...
        chooseID(i),", Averaged Per ",string(tWindow)," Second(s)"))
    xlabel("Aligned Time [s]")
    ylabel("Measured Acceleration [G]")
    plot(t,sensorAccFilt)
    plot(t,truthFilt)
    errorbar(plott, avgSensorAcc,stdSensorAcc)
    errorbar(plott, avgTruthAcc,stdTruthAcc)
    plot(plott, sensorSlope)
    plot(plott, truthSlope)
    xline(RORstart, '-b','LineWidth',2)
    legend("Tobii Acceleration","Centrifuge Ground Truth",...
        "Averaged Tobii Acceleration","Averaged Truth Acceleration",...
        "Sensor Slope","Truth Slope","ROR Start",...
        "Location","northwest")
    xlim([RORxAxisStart RORxAxisEnd])
    ylim([-0.02 0.05])
    hold off
    figROR = gcf;

 %% Figure Saving
 % Save figure if flag is set to 1
if saveFlag == 1
    % Set the path and name of the figure
    figNameGOR = strcat(chooseID(i),"_GOR");
    figNameROR = strcat(chooseID(i),"_ROR");
    OS = ispc;
    if OS == 0 % if Mac
        mkdir(strcat(outPath,"/clockDrift/Tobii"))
        saveNameGORPNG = strcat(outPath,"/clockDrift/Tobii/",...
            figNameGOR,".png");
        saveNameRORPNG = strcat(outPath,"/clockDrift/Tobii/",...
            figNameROR,".png");
%         saveNameFIG = strcat(outPath,"/clockDrift/Tobii/",...
%             figName,".fig");
    elseif OS == 1 % if Microsoft or Linux
        mkdir(strcat(outPath,"\clockDrift\",sensor))
        saveNameGORPNG = strcat(outPath,"\clockDrift\Tobii\",...
            figNameGOR,".png");
        saveNameRORPNG = strcat(outPath,"\clockDrift\Tobii\",...
            figNameROR,".png");
%         saveNameFIG = strcat(outPath,"\clockDrift\Tobii\",...
%             figName,".fig");
    end
    saveas(figGOR,saveNameGORPNG)
    saveas(figROR,saveNameRORPNG)
    %saveas(gcf,saveNameFIG)
end

end
%end