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
% saved per subject. Split the cleaning and splitting functions?
% 
% (Created on Mac M1 ARM chip)

%% Housekeeping
% Clean up entire workspace
clear;clc;close all

%% User Inputs
% Name of file to be analyzed
fileName = "all_trials_25_hz_stacked_null_str_filled.csv";
% Trial ID to analyze 
% Options are "all" or individual trial. Note that individual trial must 
% be in "[two digit subject number]-[two digit trial number]" format
chooseID = ["01-01" "01-02" "01-03"];

% Saving plots flag
% Set to 1 to save, set to 0 to not save
saveFlag = 1;

%% Data Read
% Read in full datafile and define input and output paths based on OS and
% current date
OS = ispc;
date = string(datetime("today"));
if OS == 0 % if Mac
    inPath = strcat("./data/",fileName);
    outPath = strcat("./outputs/",date,"/");
elseif OS == 1 % if Microsoft or Linux
    inPath = strcat(".\data\",fileName);
    outPath = strcat(".\outputs\",date,"\");
end
if saveFlag == 1
    [status, msg] = mkdir(outPath);
    if ~isempty(msg)
       fprintf("Output directory already exists. Name: %s\n",outPath)
    else
       fprintf("Output directory successfully created. Name: %s\n",outPath)
    end
end
% Read entire data stream into table form
T_full = readtable(inPath); % takes about 100 sec 
% Prep chooseID if set to all
if chooseID == "all"
    chooseID = unique(T_full.trial_id);
end

%% Data Preparation
% Call function to 1) create easily indexable subject and trial columns and
% 2) collect and clean table variable names. 
% Note that this function takes ~25 seconds to run
% Full dataset name: T_full
% Partioned dataset name: T
% Trials to analyze: chooseID
% Data variables: vars

% Function call
[vars,T_full] = cleanData(T_full);

%% Data Splitting
% Split data by desired trials/subjects for easier indexing, as dictated by
% chooseID.
T = T_full(matches(T_full.trial_id,chooseID),:);

%% Full Dataset Assessments
% Analyze number of occurrences of trials per subject, GLOCs per subject,
% unique events and validated events and count their occurrences

% Function call
experimentOverview(T_full);

%% Trial Overview Plots
% Choose which variable to analyze over the course of each 
%chooseVar = "HR_bpm__Equivital";
chooseVar = "AF4_delta_EEG";
% chooseVar = "P1_delta_EEG";
% chooseVar = "Cz_delta_EEG";
% chooseVar = "TP10_delta_EEG";
% chooseVar = "P8_delta_EEG";

% Function call
trialOverview(T, chooseID, chooseVar, saveFlag, outPath)

%% Time Visualizations


%% Acceleration Analysis

%% Equivital Analysis

%% fNIRS Analysis

%% EEG Analysis

%% Eye-Tracking Analysis

%% Clock Drift Calculations

