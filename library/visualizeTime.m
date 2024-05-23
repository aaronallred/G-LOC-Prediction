% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: 
% INPUTS:   T_full: full, uncleaned dataset
% OUTPUTS:  none - displays of final counts to command window

function [] = visualizeTime(T,vars,chooseID,saveFlag,outPath)
for i = 1:length(chooseID)
    fprintf("Analyzing trial ID %s...\n",chooseID(i))
    % Split 
    Ti = T(T.trial_id == chooseID(i),:);
    % Create pattern to search for - we want all time related variables
    pattern = "Time_s";
    % Find names of time variables (caps sensitive)
    timeNames = vars(contains(vars(:,:),pattern));
    % Find where time variables exist (caps sensitive)
    [~,timeCols] = find(contains(vars(:,:),pattern));
    % Plot all together
    warning('off','all')
    figure
    hold on 
    title(strcat("Native Sensor Time Comparison: Trial ",chooseID(i)))
    xlabel("Aligned Time [s]")
    ylabel("Individual Sensor Time [s]")
    plot(Ti, "Time_s_", timeNames)
    legend("Location","Northwest")
    
    % Save figure if flag is set to 1
    if saveFlag == 1
        % Set the path and name of the figure
        figName = chooseID(i);
        OS = ispc;
        if OS == 0 % if Mac
            mkdir(strcat(outPath,"/timeViz/"))
            saveNamePNG = strcat(outPath,"/timeViz/",...
                figName,".png");
            saveNameFIG = strcat(outPath,"/timeViz/",...
                figName,".fig");
        elseif OS == 1 % if Microsoft or Linux
            mkdir(strcat(outPath,"\timeViz\",chooseVar))
            saveNamePNG = strcat(outPath,"\timeViz\",...
                figName,".png");
            saveNameFIG = strcat(outPath,"\timeViz\",...
                figName,".fig");
        end
        saveas(gcf,saveNamePNG)
        %saveas(gcf,saveNameFIG)
    end
% end chooseID loop
end
% end function
end