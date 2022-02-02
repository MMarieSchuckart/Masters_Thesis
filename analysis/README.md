# :closed_book: Analysis Scripts for Merle's Master's Thesis
(last updated: January 13th 2022)

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


### [EEG_filter_epoching( ... )](EEG_preproc):
* for each participant...
    * read in fif file containing EEG data
    * pick channels (as specified in the function call arguments)
    * run ICA to get rid of blinks
    * filter EEG data (as specified in the function call arguments)
    * change trigger names so they contain information on block + feedback, sfb & sfc conditions
    * epoch data (as specified in the function call arguments), exclude data from training and block 3
    * apply baseline correction
    * save as .fif file in the working directory you used as the function's argument


### [GSS_filter_epoching( ... )](GSS_preproc):
* for each participant...
    * read in fif file containing GSS data
    * compute IMFs using Empirical Mode Decomposition (EMD)
    * create MNE Raw object containing IMFs as channels
    * run ICA on IMFs to get rid of motor & heart artifacts
    * (idk yet: somehow convert data back to 1 channel?!)
    * filter GSS data (as specified in the function call arguments)
    * change trigger names so they contain information on block + feedback, sfb & sfc conditions
    * epoch data (as specified in the function call arguments), exclude data from training and block 3
    * apply baseline correction
    * save as .fif file in the working directory you used as the function's argument


### [EEG_stats( ... )](EEG_stats):
* for each participant...
    * read in fif file containing EEG data
    * for each epoch & channel, compute PSD
    * extract Power value for each epoch, channel & frequency
    * add information on participant, sfb, sfc and feedback condition and save as df
* for each participant, channel and frequency, compute OLS regression: power ~ feedback + sfc
    * get betas for feedback & sfc, save in df with information on participant, channel and frequency
* for each channel & frequency, compute t-test against 0 (1 t-test for feedback, 1 t-test for sfc)
    * save p- & T-vales in df with information on channel & frequency
    * apply false detection rate correction on the p-values
* TO DO: plot the p-values
* TO DO: save as .csv file in the working directory you used as the function's argument

(I'll add functions for the GSS stats parts later)


## Scripts with "TESTDATEN" in the filename:
I only have a single test dataset for writing my scripts so Julius sent me a few datasets from another study (?) that also contain tremor data. I used those to test the GSS scripts. All scripts with TESTDATEN in front are scripts for reading them in or preprocessing them.


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


### Stats: 
* for each participant...
    * read in epochs-object
    * get power spectral density (PSD) for each epoch at each channel, then extract power at each frequency --> save as dataframe
    * get beta coefficients for each participant at each channel & for each frequency
    * run t-test on beta coefficients for each channel & frequency (aggregate over participants) 
     

