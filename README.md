# Merle's Master's Thesis Repo 
(last updated: October 6th 2021)

This repository is used to collect all the scripts I wrote for my Master's thesis.

&nbsp;  

## Overview 
This repository only contains a [script for reading in / preprocessing](/read_and_preproc_data.py) test data I got from Julius and another one for the [stats part](/stats.py).

This is work in progress (e.g. the analysis of the GSS data is still missing).&nbsp;  🐢

The data that I use to write the scripts right now are from a pilot study with a patient with Parkinson's disease, it's only a single file that ends somewhere in block 1 (so there's quite a bit missing). There are no real datasets yet, so I haven't tested my script yet.

&nbsp;  

## Contents of the Scripts

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
