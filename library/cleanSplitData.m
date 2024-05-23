% AUTHOR: Victoria Hurd
% DATE CREATED: 5/23/2024
% DATE LAST MODIFIED: 5/23/2024
% PROJECT: G-LOC Prediction Modeling
% DESCRIPTION: 1) create easily indexable subject and trial columns, 
% 2) collect and clean table variable names, and 3) split this prepared 
% data by desired trials/subjects for easier indexing, as dictated by
% chooseID
% INPUTS:   T_full: full, uncleaned dataset
%           chooseID: desired trials/subjects
% OUTPUTS:  vars: variable names of table
%           T: Partitioned dataset that includes only desired data
%           T_full: Full dataset that includes new subject/trial columns

function [vars,T,T_full] = cleanSplitData(T_full,chooseID)
% 1) Create Subject and Trial Columns
% Subject and trial are currently stored in a string within a cell, which
% isn't particularly easy to index. Make new subject and trial columns of
% type double
% Split trial_id string using dash as a delimiter
splitStr = split(T_full.trial_id,"-");
% Create subject column
T_full.subject = splitStr(:,1);
% Convert subject column to type double
T_full.subject = str2double(strrep(T_full.subject,',','.'));
% Create trial column
T_full.trial = splitStr(:,2);
% Convert trial column to type double
T_full.trial = str2double(strrep(T_full.trial,',','.'));

% 2) Collect table variable names for future use and put into indexable format
originalVars = T_full.Properties.VariableNames;
% Create string in non cell form
vars = strings(1,length(originalVars));
for i=1:length(originalVars)
    vars(1,i) = originalVars{1,i};
end

% 3) Split the data so that we're now only analyzing desired trials 
T = T_full(T_full.trial_id == chooseID,:);

end