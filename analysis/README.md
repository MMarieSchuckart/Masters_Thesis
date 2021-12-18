# Analysis Scripts for Merle's Master's Thesis
(last updated: December 18th 2021)

This repository is used to collect all the scripts I wrote for my Master's thesis.

&nbsp;  

## Overview 
This repository only contains a [script for reading in / preprocessing](/old_scripts/read_and_preproc_data.py) test data I got from Julius and another one for the [stats part](/old_scripts/stats.py). The scripts are super long so I'm currently trying to split them up a bit, turn the parts into functions and execute them from a main script. This way I can run the parts independently.

This is work in progress.&nbsp;  🐢

The data that I use to write the scripts right now are from a pilot study with a patient with Parkinson's disease, it's only a single file that ends somewhere in block 1 (so there's quite a bit missing). There are no real datasets yet, so I haven't tested my script yet.

&nbsp;  

## Functions I call from the [main script](/main) 

### [read_in_EEG(working_directory)](EEG_read_xdf):
* for each participant...
    * read in xdf file containing EEG data
    * create MNE-Raw-Object containing the EEG data with triggers and automatically detected eog events ('bad blinks') as annotations
    * save each file in the working directory you used as the function's argument

### [read_in_GSS(working_directory)](GSS_read_xdf):
* for each participant...
    * read in xdf file containing GSS data
    * create MNE-Raw-Object containing the GSS data with triggers as annotations
    * save each file in the working directory you used as the function's argument

(I'll add preprocessing functions and functions for the stats parts later)


## Contents of the old scripts

### Reading in and preprocessing data:
* for each participant...
    * read in xdf file with the EEG and GSS data
    * create MNE-Raw-Object containing the EEG data and GSS data and triggers as annotations
    * blink detection: 
        * detect blinks and show segments as annotations
        * fit ICA using eog channels
        * plot ICA & bad blink segments to check if everything worked and they overlap
        * exclude first ICA component to get rid of blinks
        * filter EEG data (FIR bandpass (5-35 Hz) zero-phase hamming filter)
        * divide data into blocks, get epochs from blocks 1 & 2 (exclude training) for -1.5 - 7 s around the onset of each trial 
--> use interval of -1.5 - 0 s before trial onset as baseline for baseline correction
    * save epochs and MNE-Raw-Object

:warning: &nbsp; TO DO: Filtering & epoching for GSS data 
--> delete GSS data from annotations and create another MNE-Raw-Object instead
&nbsp;

### Stats: 
* for each participant...
    * read in epochs-object
    * get power spectral density (PSD) for each epoch at each channel, then extract power at each frequency --> save as dataframe
     
     
:warning: &nbsp; TO DO: ANOVAs, Analysis of the GSS data
