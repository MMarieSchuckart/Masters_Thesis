""" Main Script for Merle's Master's Thesis """
# Scripts by Merle Schuckart, written Sept. - Dec. 2021 
# Github: MMarieSchuckart
# Email: merle.schuckart@gmx.de

# Use this script to run the whole analysis 
# for Merle's Thesis or just run parts of it.


""" Before using this script...

Please make sure the packages MNE, pyxdf and emd can be installed 
on your machine if you have never used them before.
You might need to pip install them in your terminal:
     pip install pyxdf
     pip install mne
     pip install emd
  
If you can't pip install the latest stable pyxdf version directly, 
try installing the latest developer version from github 
by running this command in your terminal:
    pip install git+https://github.com/xdf-modules/pyxdf.git
    
I found this solution here: https://github.com/xdf-modules/pyxdf
(BTW: The internet says you should install PyPDF2 to be able to import pyxdf, 
but that doesn't work.)

"""  

#%%
""" set working directory """

import sys
# Choose file containing all analysis scripts as wd
sys.path.append("/Users/merle/Desktop/Masterarbeit/")


# Choose file containing your data files.
# This file should also contain all other analysis scripts with the functions
# we execute herein.
data_file = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# where do you want to save plots?
plot_file = "/Users/merle/Desktop/Masterarbeit/Plots/"

#%%
""" ------ get functions from the other scripts ------ """

from EEG_read_xdf import read_in_EEG

from GSS_read_xdf import read_in_GSS

from EEG_preproc import EEG_filter_epoching

from GSS_preproc import GSS_filter_epoching

from EEG_stats import EEG_stats

from GSS_stats import GSS_stats


#%%
""" ------------ Part 1: EEG DATA ANALYSIS ------------ """

""" ---- 1.1 READ IN EEG DATA ---- """
# Input: Raw EEG data (.xdf file)
# Output: MNE Raw object for each participant containing 
#         EEG sensor values + timestamps + triggers as annotations
# The files from the output are saved in the WD as .fif files.

read_in_EEG(data_file)


#%%
""" ---- 1.2 PREPROCESS EEG DATA ---- """
# Input: MNE Raw object containing EEG data + triggers (Output from 1.1) 
# Output: 
    # MNE Epochs object containing filtered + epoched EEG data (Format: .fif),
    # Plots: before and after filtering????
    
""" Settings for picking channels """
# channels to pick up signal from auditory cortex: 
# 65, 1, 69 & the four neighboring electrodes 33, 38, 66 & 68 

# channels to pick up signal from right hand area of motor cortex: 
# 71 as center + neighboring electrodes 34, 77, 2, 5, 35 & 78

# visual cortex: 108 54 55 109 61 117 118 63    

eeg_channel_picks = ["EEG_065", "EEG_001", "EEG_069",
                     "EEG_033", "EEG_038", "EEG_066", "EEG_068",
                     "EEG_71", "EEG_034", "EEG_077", 
                     "EEG_002","EEG_005", "EEG_035", "EEG_078", 
                     "EEG_108", "EEG_54", 
                     "EEG_55", "EEG_109", "EEG_61", 
                     "EEG_117", "EEG_118", "EEG_63"]

# choose eog channels (the 3 eeg channels from the face)
#eog_channels = ["EEG_020", "EEG_031", "EEG_032"] 
# first two are for left eye, second two are for right eye:
eog_channels = ["EEG_031", "EEG_021", 
                "EEG_032", "EEG_026"]
        
""" Settings for Filtering: """ 
# set filter type
# = default option; gives improved attenuation using fewer samples than “firwin2”
#eeg_fir_design = 'firwin' 
# set type of window function
#eeg_window_type = 'hamming'    
# zero phase filter
#eeg_phase = "zero"
# highpass filter 
#eeg_bandpass_fmin = 5
# lowpass filter 
#eeg_bandpass_fmax = 35,
# set number of runs to run in parallel
#n_jobs = 1

""" ICA Settings """
# ICA method is fastICA (= default)
# use the first 5 components from the PCA (faster computation, no need to look at all 13 channels)
#ica_n_components = 5
#ica_decim = 5 # only use every 5th sample from the data --> speed up computation
#ica_max_iter = "auto" 
# use a random seed to get the same results every time we run the ICA on the same data
#ica_random_state = 97 

""" Settings for Epoching:"""
#eeg_prestim_cutoff = -1.5 
#eeg_poststim_cutoff = 4 

# I set all those values as default arguments, so can 
# just pass the wd in the function call, but if you want 
# to change arguments, just put them in the brackets as well. 

""" run preproc function """    
EEG_filter_epoching(working_directory = data_file, 
                    eeg_channel_picks = eeg_channel_picks, 
                    eog_channels = eog_channels)

 
#%%   
""" ---- 1.3 STATS FOR EEG DATA ---- """
# Input: MNE Epochs object containing filtered + epoched EEG data (Output from 1.2), Settings for the PSD
# Output: 
    # 1 table containing p- & T-values for each channel & frequency 
    # 1 table containing beta coefficients for each participant, channel & frequency 
    # (both saved in file you set as working directory)


""" Settings for power spectral density (PSD) analysis (using Welch's Method) """ 

# settings for cropping the epoch:
#psd_tmin = 1 
#psd_tmax = 4

# settings for running the PSD:
# sampling rate of eeg    
#psd_sfreq = 500
# freq range we'd like to look at (I need 5-35, so I added +1 at each end)
#psd_fmin = 4
#psd_fmax = 36 

# psd_n_fft = 256*6 #??? 
# 256 is default, but only produces values for each 2nd frequency. 
# If you multiply by 2, you get roughly each frequency, I guess if 
# you go high enough, you get perfekt INTs as freqs.
# If you set n_fft as == len(samples), you get more values aka more freqs.
# --> I hardcoded the n_fft in the PSD function call in 
# the EEG_stats() script as == len(samples)

#psd_n_overlap = 0 
#psd_n_per_seg = None 
#psd_n_jobs = 1
#psd_average = 'mean'
#psd_window = 'hamming'

# I set all those values as default arguments, so can 
# just pass the wd in the function call, but if you want 
# to change arguments, just put them in the brackets as well. 

""" run stats function """  
EEG_stats(working_directory = data_file)

# hint: p-values are plotted in an R-Script called "Master_Plots"

#%%
""" ------------ Part 2: GSS DATA ANALYSIS ------------ """

""" ---- 2.1 READ IN GRIP STRENGTH SENSOR (GSS) DATA ---- """
# Input: Raw GSS data (.xdf file)
# Output: MNE Raw object containing 
#         GSS values + timestamps + triggers as annotations
# The files from the output are saved in the WD as .fif files.

read_in_GSS(working_directory = data_file)


#%%
""" ---- 2.2 PREPROCESS GSS DATA ---- """
# Input: MNE Raw object containing GSS data + triggers (Output from 1.1) 
# Output: 
    # MNE Epochs object containing filtered + epoched GSS data (Format: .fif)
    # Plots: before and after filtering???

""" Settings for Filtering: """
# get "4–12 Hz range since this range is where a majority 
# of the tremor is contained" (Archer et al., 2018)

# set filter design
#gss_fir_design = 'firwin' 
# set type of window function
#gss_window_type = 'hamming'
# highpass filter:
#gss_bandpass_fmin = 4
# lowpass filter
#gss_bandpass_fmax = 12
# set number of runs to run in parallel
#gss_n_jobs = 1
# use zero-phase filter
#gss_phase = "zero"

""" Settings for Epoching: """
# set cutoffs (in seconds) for epoching:
#gss_prestim_cutoff = -1.5 
#gss_poststim_cutoff = 4 

# I set all those values as default arguments, so can 
# just pass the wd in the function call, but if you want 
# to change arguments, just put them in the brackets as well. 

""" run preproc function """
GSS_filter_epoching(working_directory = data_file)

      
#%%
""" ---- STATS FOR GSS DATA ---- """
# Input: MNE Epochs object containing filtered + epoched GSS data (Output from 2.2)
# Output:  
    
# check this:    
gss_psd_tmin = 1
gss_psd_tmax = 4

# sampling rate is 45 Hz for the GSS data
gss_psd_sfreq = 45

# which frequencies should be included?
# get "4–12 Hz range since this range is where a majority 
# of the tremor is contained" (Archer et al., 2018)
gss_psd_fmin = 4
gss_psd_fmax = 12

gss_psd_n_overlap = 0 
gss_psd_n_per_seg = None 

gss_psd_n_jobs = 1 
gss_psd_average = 'mean' 
gss_psd_window = 'hamming'

# gss_psd_n_fft = 256*6 #??? 
# 256 is default, but only produces values for each 2nd frequency. 
# If you multiply by 2, you get roughly each frequency, I guess if 
# you go high enough, you get perfekt INTs as freqs.
# If you set n_fft as == len(samples), you get more values aka more freqs.
# --> I hardcoded the n_fft in the PSD function call in 
# the GSS_stats() script as == len(samples)

GSS_stats(working_directory = data_file)
        
