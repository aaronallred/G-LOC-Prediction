% AUTHOR: Victoria Hurd
% DATE CREATED: 5/20/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: Wrapper function to perform relevant analyses on base G-LOC
% data. Analyses include data visualizations, identification and 
% quantification of missing values, . 
% The primary purpose of the wrapper is to input which trials to query. The
% given G-LOC data is large, and it is advantageous to be able to only 
% analyze one trial if necessary while being able to visualize and analyze 
% all trials. The wrapper also contains the ability to output a smaller
% portion of the total data that corresponds with the user inputs. 
% INPUTS: The only input for the wrapper script is the trial IDs being
% queried. Options are currently "[ID here]" or "all". Must be in either
% string or numerical form. 
% OUTPUTS: All outputs are optional. Output 1 is saved figures, both matlab
% type and jpg. Output 2 is partitioned data based on given ID inputs. 
% 
% FUTURE WORK: input by subject. if input is numerical, then analyze trial
% n or the inputted series, ex. trials 1 through 10. Saving of partitioned
% data. Saving outputted images into [wrapperOutput<Date>] divided into
% subfolders by analysis. Each folder will have both matlab and jpg figure
% saved per subject. 
% 
% (Created on Mac M1 ARM chip)

%% Housekeeping
clear;clc;close all
OS = ispc;

%% User Inputs
chooseID = "01-01";

%% Data Read
if OS == 0 % if Mac
    load("./data/flow_data.mat")
    load("./data/mask.mat")
    outPath = "./movieOutputs/";
elseif OS == 1 % if Microsoft or Linux
    load(".\data\flow_data.mat")
    load(".\data\mask.mat")
    outPath = ".\movieOutputs\";
end

