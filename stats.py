#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Stats script for Merle's Master's Thesis """

    
# look at the TFR, is there a linear relationship between the 
# conditions (sfb, sfc & feedback) and the TFR amplitudes
# The question is: Are there TFR correlates of different processing stages?
# This script computes the Stats on Betas across Participants


"""

Stats part for EEG data

Part of Merle's Master Thesis
Version 1: 25.8.2021

"""
#-------------------------------------------------

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

# for 
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for dataframes as in R
import pandas as pd

# for computing the ANOVAs
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# for getting effect sizes
# run the following bit in your terminal:

import scipy.sparse
import pingouin as pg 
# This doesn't work. Why tho.

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
tmp_df = pd.DataFrame(columns = ["participant", "epoch", "feedback", "sfb", "sfc", "channel", "frequency", "power_value"])

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
                
        # get single epoch, crop to get time interval from second 1 - second 4 
        single_epoch = epochs[pick_epoch].crop(tmin = psd_tmin, tmax = psd_tmax)
        
        # save information converning the trial:
        # number of epoch 
        epoch_nr = pick_epoch
        # feedback condition (= ov, oa, av)
        feedback = single_epoch._metadata["feedback"].values[0]
        # sfb condition (= scaling of feedback)
        sfb = single_epoch._metadata["sfb"].values[0]
        # sfc condition (= % of max. grip strength)
        sfc = single_epoch._metadata["sfc"].values[0]
        
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
        # END loop channels
    # END loop epochs
# END loop participants




""" Stats """ 
    
""" aggregate dataframe """
# get the mean amplitude at each channel for each freq

# include sfb
#power_vals_agg = power_vals_all.groupby(["participant", "sfb", "sfc", "feedback", "channel", "frequency"], as_index = False)["power_value"].mean()

# don't include sfb
power_vals_agg = power_vals_all.groupby(["participant", "channel", "freq", "sfc_F", "sfc_degr_freedom", "sfc_p", "feedback_F", "feedback_degr_freedom", "feedback_p"], as_index = False)["power_value"].mean()

# create empty dfs for ANOVA results
tmp_df = pd.DataFrame(columns = ["participant", "channel", "freq", "sfc_F", "sfc_degr_freedom", "sfc_p", "sfc_eta", "feedback_F", "feedback_degr_freedom", "feedback_p", "feedback_eta"])
Anova1_res = pd.DataFrame(columns = ["participant", "channel", "freq", "sfc_F", "sfc_degr_freedom", "sfc_p", "sfc_eta", "feedback_F", "feedback_degr_freedom", "feedback_p", "feedback_eta"])

# loop participants
for participant in np.unique(power_vals_agg["participant"]):
    # only get aggregated data of the current participant
    df_participant = power_vals_agg[power_vals_agg["participant"] == participant]
    
    # loop channels:
    for channel in np.unique(df_participant["channel"]): 
        # get subset of df with data for the current channel
        df_channel = df_participant[df_participant["channel"] == channel]
    
        # loop frequencies:
        for freq in np.unique(df_channel["frequency"]): 
            # get subset of df with data for the current frequency
            df_freq = df_channel[df_channel["frequency"] == freq]
    
            # TO DO: test conditions: sphericity & normality of distribution
            # --> if not given, rank transform data   
    
                
            """ Anova: power ~ feedback + sfc """

            # fit model (ols = ordinary least squares; an OLS regression is the same as a linear regression)
            mod1 = ols(formula = "power_value ~ feedback + sfc", data = df_freq).fit()
            # compute ANOVA using OLS model
            anova1_res = anova_lm(mod1)

            """ get results for sfc """
            sfc_F = anova1_res["F"]["sfc"]
            sfc_degr_freedom = anova1_res["df"]["sfc"]
            sfc_p = anova1_res["PR(>F)"]["sfc"]
            # get effect size eta (1 as placeholder, change this later)
            sfc_eta = 1 
            
            """ get results for feedback """    
            feedback_F = anova1_res["F"]["feedback"]
            feedback_degr_freedom = anova1_res["df"]["feedback"]
            feedback_p = anova1_res["PR(>F)"]["feedback"]
            # get effect size eta (1 as placeholder, change this later)
            feedback_eta = 1
            
            # append to df
            tmp_df.loc[0] = [participant, channel, freq, sfc_F, sfc_degr_freedom, sfc_p, sfc_eta, feedback_F, feedback_degr_freedom, feedback_p, feedback_eta]
            Anova1_res = Anova1_res.append(tmp_df) 
            
            
            
            
            
            """ get Eta effect sizes """ 
            
            # --> Eta squared (η²) = sum of squares of the effect we're looking at divided by the total sum of squares
            # η² = SSeffect / SStotal
            
            #eta_sfc = anova1_res["sum_sq"]["sfc"] / anova1_res["mean_sq"]["sfc"]
            #eta_feedback = 



             
            """ TO DO:"""
            # Which PSD Method (Welch or Multitaper)? 
            # nonparametric PSD method: Periodogram as direct transformation of signal (--> Welch method)?
            # compute ANOVA, get Betas
            # ANOVA again over all subjects
            
