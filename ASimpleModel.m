%% Housekeeping
% Clean up entire workspace
clear;clc;close all
addpath("./library/")

%% User Inputs
% Name of file to be analyzed
fileName = "all_trials_25_hz_stacked_null_str_filled.csv";
% Trial ID to analyze 
% Options are "all" or individual trial. Note that individual trial must 
% be in "[two digit subject number]-[two digit trial number]" format
chooseID = ["01-01" "01-02" "03-01"];
% chooseID = "all";

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

%% Prep chooseID if set to all
if chooseID == "all"
    chooseID = string(unique(T_full.trial_id));
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

%% A Simple Linear Model
chooseVar = "HR_bpm__Equivital";

binFreq = 5; % How many seconds apart are bins?

% Bin Width Size and gap between predictor and label bins
width = 15; gap = width+1; 


% Created new FeatureGen to take varialbe stream in Table T and create
% Engineered Features for use in our regression model
[PredictorTable] = FeatureGen(T, chooseID, chooseVar, binFreq, width, gap);

 mdl = fitglm(PredictorTable.predictor,PredictorTable.label,...
              'linear','distr','binomial','link','logit');
 
 plotSlice(mdl)
 plotDiagnostics(mdl)
 plotResiduals(mdl,'probability')