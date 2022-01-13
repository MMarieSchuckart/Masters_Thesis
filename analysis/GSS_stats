""" Stats script for Merle's Master's Thesis

Stats part for GSS data

Part of Merle's Master Thesis
Version 1: 13.01.2022

"""

#%%

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
#gss_psd_tmin = 1
#gss_psd_tmax = 4 
#gss_psd_sfreq = 500 
#gss_psd_fmin = 4 
#gss_psd_fmax = 36 
#gss_psd_n_overlap = 0 
#gss_psd_n_per_seg = None 
#gss_psd_n_jobs = 1
#gss_psd_average = 'mean' 
#gss_psd_window = 'hamming'

# create function for running EEG stats script
def GSS_stats(working_directory, 
              gss_psd_tmin = 1, 
              gss_psd_tmax = 4, 
              gss_psd_sfreq = 500, 
              gss_psd_fmin = 4, 
              gss_psd_fmax = 36, 
              gss_psd_n_overlap = 0, 
              gss_psd_n_per_seg = None, 
              gss_psd_n_jobs = 1, 
              gss_psd_average = 'mean', 
              gss_psd_window = 'hamming'):
    
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.

    """ settings """

    """ 1.1 load packages """

    # os for setting working directory
    import os

    # glob for getting all files in a directory
    import glob

    # Python MNE for eeg data analysis / processing
    # put this into your terminal to install python nme
        # pip install mne
    # NME should be active by default and should already have the 
    # necessary dependencies (numpy, scipy, and matplotlib).    
    import mne

    # for rounding frequency values
    import numpy as np

    # for dataframes as in R
    #import pandas as pd

    # for computing the multiple linear regression 
    import statsmodels.formula.api as smf
    
    # import function for t-test for 2 related samples
    from scipy.stats import ttest_rel

    # function for false detection rate (FDR) correction
    #from statsmodels.stats.multitest import fdrcorrection as fdr

    # for plotting
    import matplotlib.pyplot as plt


#%%
    """ 2. read in data """
    
    # 2.1 set path to working directory
    os.chdir(working_directory)
    
    # 2.2 get list of all files in the directory that end with "epo.fif"
    file_list = glob.glob(working_directory + "gss_" + "*epo.fif")
    
    
#%%
    """ 3.1  Loop participants"""
    for filename in file_list:
        
        # get participant number from file name string
        file = filename[-11:-8] # participant numbers can be 1 - 999
        
        # correct the number if it has < 3 digits
        if file[0] == "n":
            file = file[-1]
        elif file[0] == "t":
            file = file[-2:len(file)]
            
        # 3.1 Read in the epoched data
        epochs = mne.read_epochs("eeg_participant"+file+"_epo.fif")
    
        # save name of participant (yes I know I'm using the exact same value twice here)
        part_nr = file
         
#%%
        """ loop epochs """
        for pick_epoch in range(0, epochs.__len__()):   
            
            # get single epoch, crop to get time interval from second 1 - second 4 
            single_epoch = epochs[pick_epoch].crop(tmin = gss_psd_tmin, tmax = gss_psd_tmax)
            
            # save information converning the trial:
            # number of epoch 
            epoch_nr = pick_epoch
            # feedback condition (= ov, oa, av)
            feedback = single_epoch._metadata["feedback"].values[0]
            # sfb condition (= scaling of feedback)
            sfb = single_epoch._metadata["sfb"].values[0]
            # sfc condition (= % of max. grip strength)
            sfc = single_epoch._metadata["sfc"].values[0]

            # set title of epoch
            #title = "participant " + file + ", epoch " + str(pick_epoch) + " of " + str(len(epochs.picks))
            
            # plot psd (no clue which Method this is, the MNE docs don't want me to know this)
            #single_epoch.plot_psd(fmin = 4, fmax = 12, spatial_colors = True)
             
            
            
            
            
            """ TO DO: Change the following part so it fits the GSS data """
            # There are too many channels in the epoch object, this doesn't make sense.
            
            # get data for the current channel
            channel_epoched_data = single_epoch._data[0]
            
            """compute power spectral density (PSD) analysis using Welch's Method """
            # the following function returns... 
            # ...the PSD values for each channel (--> saved in object "psds")... 
            # ...for all frequencies (--> saved in object "freqs")... 
            # ...for the current epoch
            psds, freqs = mne.time_frequency.psd_array_welch(channel_epoched_data,
                                                             sfreq = gss_psd_sfreq, 
                                                             fmin = gss_psd_fmin, 
                                                             fmax = gss_psd_fmax, 
                                                             n_fft = len(channel_epoched_data), 
                                                             n_overlap = gss_psd_n_overlap, 
                                                             n_per_seg = gss_psd_n_per_seg, 
                                                             n_jobs = gss_psd_n_jobs, 
                                                             average = gss_psd_average, 
                                                             window = gss_psd_window)
            # round frequencies
            freqs = np.round(freqs, 1)

            """ loop frequencies """
            for freq_val in range(0,len(freqs)-1):

                # if freq value is an integer...
                if freqs[freq_val].is_integer():
                    # get corresponding value from psds array and add to df
                    freq = freqs[freq_val] 
                    psd_val = psds[freq_val]                  
 
                    # save nr of participant, nr of epoch, metadata,
                    # nr of channel, frequency & corresponding power value 
                    # as temporary 1 row df:
                    tmp_df.loc[0] = [part_nr, epoch_nr, feedback, sfb, sfc, channel_nr, freq, psd_val]
                    
                    # append as new row to dataframe containing the values for all participants:
                    power_vals_all = power_vals_all.append(tmp_df)
                
            # END loop frequencies    
        # END loop epochs
        print(" -- finished computing betas for participant " + str(file) + " -- ")
    # END loop participants
