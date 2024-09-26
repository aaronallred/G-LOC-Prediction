clc
clear
load fisheriris % Load example dataset
% Remove observations from two classes to simulate one majority class and 
% two minority classes
I = [1:50 51:75 101:125]; % Exclude some samples to create an unbalanced dataset
meas = meas(I,:);
species = categorical(species(I));
% Prepare some variables for use in plotting results of examples
us = unique(species);
c = 'rbg';
Example 1: Balance Dataset
Use SMOTE to balance dataset (using default 5 k-nearest neighbors)
rng(42)



[meas1,species1,new_meas,new_species] = smote(meas, [], 'Class', species);
K = groupcounts(species);
figure,hold on
legend_str = {};
for ii=1:numel(us)
    I = species==us(ii);
    plot(meas(I,1),meas(I,4),'.','Color',c(ii));
    legend_str{end+1} = sprintf('%s (N=%d)', us(ii) ,K(ii));
    I = new_species==us(ii);
    if sum(I)>0
        plot(new_meas(I,1),new_meas(I,4),'o','Color',c(ii));
        legend_str{end+1} = sprintf('Synthesized %s (N=%d)', us(ii), K(ii));
    end
end
legend(legend_str)
title('Example 1: Balanced dataset (by oversampling minority classes)')