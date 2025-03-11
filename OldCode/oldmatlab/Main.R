library(vroom)
library(ggplot2)
library(dplyr)
library(tidyverse)

Data <- vroom("./data/all_trials_25_hz_stacked_null_str_filled.csv")

trial = '03-02'
trialid <- Data[Data$trial_id == trial ,c("trial_id")]
HRdata <- Data[Data$trial_id == trial ,c("Time (s)", "HR (bpm) - Equivital")]

p <- ggplot(HRdata,aes(x=`Time (s)`, y= `HR (bpm) - Equivital`,group=1)) + geom_line() + geom_point()
p