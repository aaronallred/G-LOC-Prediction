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
% Must be in "[two digit subject number]-[two digit trial number]" format
chooseID = "01-01";

%% Data Read
% Read in full datafile and define input and output paths based on OS
OS = ispc;
if OS == 0 % if Mac
    inPath = strcat("./data/",fileName);
    outPath = "./outputs/";
elseif OS == 1 % if Microsoft or Linux
    inPath = strcat(".\data\",fileName);
    outPath = ".\outputs\";
end
% Read entire data stream into table form
T_full = readtable(inPath); % takes about 100 sec 

%% Data Prep and Split 
% Call function to 1) create easily indexable subject and trial columns, 
% 2) collect and clean table variable names, and 3) split this prepared 
% data by desired trials/subjects for easier indexing, as dictated by
% chooseID. Note that this function takes ~25 seconds to run
% Full dataset name: T_full
% Partioned dataset name: T
% Trials to analyze: chooseID
% Data variables: vars

% Function call
[vars,T,T_full] = cleanSplitData(T_full,chooseID);

%% Full Dataset Assessments
% Analyze number of occurrences of trials per subject, GLOCs per subject,
% unique events and validated events and count their occurrences


