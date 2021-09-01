#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Stats script for analyzing EEG & grip strength sensor (gss) data """

Part of Merle's Master Thesis

Version 1: 1.9.2021

"""
#-------------------------------------------------

""" settings """

""" 1.1 load packages """

# os for setting working directory
import os

# Python MNE for eeg data analysis / processing
# put this into your terminal to install python nme
    # pip install mne
# NME should be active by default and should already have the 
# necessary dependencies (numpy, scipy, and matplotlib).    
import mne

# for 
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for dataframes as in R
import pandas as pd

#-------------------------------------------------

""" 1.2 set variables for power spectral density (PSD) analysis using Welch's Method """ 

# settings for cropping the epoch:
psd_tmin = 1 
psd_tmax = 4

# settings for running the PSD:
# sampling rate of eeg    
psd_sfreq = 500
# # freq range we'd like to look at (I need 5-35, so I added 1 freq at each end)
psd_fmin = 4
psd_fmax = 36 

# psd_n_fft = 256*6 #??? # 256 is default, but only produces values for each 2nd frequency. If you multiply by 2, you get roughly each frequency, I guess if you go high enough, you get perfekt INTs as freqs
# If you set n_fft as == len(samples), you get more values aka more freqs
psd_n_overlap = 0 
psd_n_per_seg = None 
psd_n_jobs = 1
psd_average = 'mean'
psd_window = 'hamming'



#-------------------------------------------------

""" 2. read in data """

# 2.1 set path to working directory
os.chdir("/Users/merle/Desktop/Masterarbeit/Master_Testdaten/")

# 2.2 get list of all files in the directory that start with "eeg_epochs" and end with ".fif"
file_list = glob.glob("/Users/merle/Desktop/Masterarbeit/Master_Testdaten/eeg_epochs*.fif")

# 2.3 create df for power values
power_vals_all = pd.DataFrame(columns = ["participant", "epoch", "channel", "frequency", "power value"])
tmp_df = pd.DataFrame(columns = ["participant", "epoch", "channel", "frequency", "power value"])

""" 2.4 Loop participants """
for filename in file_list:
    
    file = filename[-7:-4] # participant numbers can be 1 - 999
    
    # correct the number if it has < 3 digits
    if file[0] == "n":
        file = file[-1]
    elif file[0] == "t":
        file = file[-2:len(file)]
        
    # 2.4.1 Read in the epoched data
    epochs = mne.read_epochs("eeg_epochs_participant"+file+".fif")


    # save name of participant (yes I know I'm using the exact same value twice here)
    part_nr = file
    
    
    # 2.4.2 get power spectrum for each electrode & each epoch, then extract power at each freq
    
    """ loop epochs """
    for pick_epoch in range(0, epochs.__len__()):
        
        # save number of epoch 
        epoch_nr = pick_epoch
        
        # get single epoch, crop to get time interval from second 1 - second 4 
        single_epoch = epochs[pick_epoch].crop(tmin = psd_tmin, tmax = psd_tmax)
        
        # get channel names
        channels = single_epoch.ch_names
      
        # set title of epoch
        title = "participant " + file + ", epoch " + str(pick_epoch) + " of " + str(len(epochs.picks))
        
        # plot psd (no clue which Method this is, the MNE docs don't want me to know this)
        #single_epoch.plot_psd(fmin = 5, fmax = 35)
        
        """ Loop channels """
        for channel in range(0, len(channels)):    
            # get data for the current channel
            channel_epoched_data = single_epoch._data[0][channel]

            # save channel name in df
            channel_nr = single_epoch.ch_names[channel]
            
            """compute power spectral density (PSD) analysis using Welch's Method """
            # the following function returns... 
            # ...the PSD values for each channel (--> saved in object "psds")... 
            # ...for all frequencies (--> saved in object "freqs")... 
            # for the current epoch
            psds, freqs = mne.time_frequency.psd_array_welch(channel_epoched_data,
                                                             sfreq = psd_sfreq, 
                                                             fmin = psd_fmin, 
                                                             fmax = psd_fmax, 
                                                             n_fft = len(channel_epoched_data), 
                                                             n_overlap = psd_n_overlap, 
                                                             n_per_seg = psd_n_per_seg, 
                                                             n_jobs = psd_n_jobs, 
                                                             average = psd_average, 
                                                             window = psd_window)
            # round frequencies
            freqs = np.round(freqs, 1)

            """ loop frequencies """
            for freq_val in range(0,len(freqs)):

                # if freq value is an integer...
                if freqs[freq_val].is_integer():
                    # get corresponding value from psds array and add to df
                    freq = freqs[freq_val] 
                    psd_val = psds[freq_val]                  
 
                    # save nr of participant, nr of epoch, 
                    # nr of channel, frequency & corresponding power value 
                    # as temporary 1 row df:
                    tmp_df.loc[0] = [part_nr, epoch_nr, channel_nr, freq, psd_val]
                    
                    # append as new row to dataframe containing the values for all participants:
                    power_vals_all = power_vals_all.append(tmp_df)
                

            """ TO DO:"""
            # Which PSD Method (Welch or Multitaper)? 
            # nonparametric PSD method: Periodogram as direct transformation of signal (--> Welch method)?
            # compute ANOVA, get Betas
            # ANOVA again over all subjects
