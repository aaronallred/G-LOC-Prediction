% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: 
% INPUTS:   T_full: full, uncleaned dataset
% OUTPUTS:  none - displays of final counts to command window

function [] = visualizeTime(T,vars, chooseID)

for 
% Create pattern to search for - we want all time related variables
pattern = "Time_s";
% Find names of time variables (caps sensitive)
timeNames = stringVars(contains(vars(:,:),pattern));
% Find where time variables exist (caps sensitive)
[~,timeCols] = find(contains(vars(:,:),pattern));
% Plot all together
warning('off','all')
figure
hold on 
title("Native Sensor Time Comparison")
xlabel("Aligned Time [s]")
ylabel("Individual Sensor Time [s]")
plot(T, "Time_s_", timeNames)
legend("Location","Northwest")
hold off
% It appears that none of the sensors dictate when trial begins - in other
% words, no individual sensor drives elapsed time. Elapsed time likely
% comes from a set amount of time prior to GOR spin up
% What is the first value of the centrifuge time variable?
centTimesReal = T{~isnan(T.Time_s__Centrifuge),1};
disp("First non-NaN centrifuge time [s]:")
disp(centTimesReal(1))

end