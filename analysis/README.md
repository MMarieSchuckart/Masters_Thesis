# :closed_book: Analysis Scripts for Merle's Master's Thesis
(last updated: February 10th 2022)

This repository is used to collect all the scripts I wrote for my Master's thesis.

&nbsp;  

## Overview 
The data that I use to write the scripts right now are from a pilot study with a patient with Parkinson's disease, it's only a single file that ends somewhere in block 1 (so there's quite a bit missing). There are no real datasets yet, so I haven't tested my script yet.

&nbsp;  

## Functions I call from the [main script](/main) 

### [read_in_EEG(working_directory)](EEG_read_xdf):
* for each participant...
    * read in xdf file containing EEG data
    * create MNE-Raw-Object containing the EEG data with triggers and automatically detected eog events ('bad blinks') as annotations
    * save each file in the working directory you used as the function's argument

### [read_in_GSS(working_directory)](GSS_read_xdf):
* for each participant, read in xdf file containing GSS data as Epochs:
    * read in xdf file containing GSS data
    * cut into epoched data
    * exclude epochs that are >= 1s too short
    * change trigger names so they contain information on block + feedback, sfb & sfc conditions
    * create dataframe containing information on sfc, sfb, feedback, 
      epoch nr and ID of participant for each epoch
    * find matching trigger for each epoch
    * exclude epochs you couldn't find a matching trigger for
    * exclude epochs from training block & (passive) block 3
    * make sure epochs have the same length
    * create MNE-Epochs-Object containing the GSS data
    * save Epochs as .fif files and dataframe with epoch info as .csv 
      in the working directory you used as the function's argument


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
    * read in fif file containing epoched GSS data
    * apply Hampel filter for imputation of missing samples
    * compute IMFs using Empirical Mode Decomposition (EMD)
    * exclude last IMF
    * convert IMFs back to 1 signal
    * filter GSS data (disabled this part)
    * save as .fif file in the working directory you used as the function's argument


### [EEG_stats_ttests( ... )](EEG_stats_part1):
* for each participant...
    * read in fif file containing epoched & filtered EEG data
    * for each epoch & channel, compute PSD
    * extract Power value for each epoch, channel & frequency
    * add information on participant, sfb, sfc and feedback condition and save as df
* for each participant, ROI and frequency band, compute OLS regression: power ~ feedback + sfc
    * get betas for feedback & sfc, save in df with information on participant, ROI and frequency band
* for each ROI & frequency band, compute 2 1-sample t-tests against 0 (1 t-test for feedback, 1 t-test for sfc)
    * save p- & T-vales in df with information on ROI & frequency band
    * apply false detection rate correction on the p-values
* save betas and t-test results as .csv files in the working directory you used as the function's argument


### [EEG_stats_coherences( ... )](EEG_stats_part2): 
* for each participant...
    * read in fif file containing epoched & filtered EEG data
    * for each epoch, ROI & frequency band, compute coherences
* get average coherences 
* save results as .csv files in the working directory you used as the function's argument
