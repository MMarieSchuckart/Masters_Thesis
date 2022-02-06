
""" Stats script for Merle's Master's Thesis

Stats part for GSS data

Part of Merle's Master Thesis
Version 1: 13.01.2022

"""


#%%

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# create function for running EEG stats script
def GSS_stats(working_directory, 
              gss_psd_sfreq = 80, 
              gss_psd_fmin = 4, 
              gss_psd_fmax = 12, 
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

    # for working with pandas dataframes 
    import pandas as pd
    
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
    
    # 2.2 get list of all files in the directory that end with "_filtered_epo.fif"
    file_list = glob.glob(working_directory + "gss_" + "*_filtered_epo.fif")
        
    # create empty df for the results of the PSDs of all participants
    gss_PSDs_all = pd.DataFrame(columns = ['block', 'sfb', 'sfc', 'feedback', 'epoch', 'power', 'frequency', 'ID'])

   #%%  
    """ 6. loop fif file names in file_list (aka loop participants): """
    for file_idx in range(0, len(file_list)):
        
        """ save participant number"""
        # the participant numbers in file list are in the wrong order
        # so get the number from the filename instead of the loop index number
        participant = file_list[file_idx][-20:-17]
        
        # if participant number has < 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]        
        
        """ read in .fif file with the epochs and .csv file with the epoch conditions"""
        epochs = mne.read_epochs("gss_participant" + str(participant) + "_filtered_epo.fif")
        gss_epochs_conditions = pd.read_csv("gss_participant" + str(participant) + "_epo_conditions.csv", sep = ",")

        # remove weird index column at position 0
        gss_epochs_conditions.drop(columns = gss_epochs_conditions.columns[0], axis = 1, inplace = True)

         
#%%

        """ loop epochs """
        # placeholders for getting power peak and corresponding frequency
        power = []
        freq = []
        
        for pick_epoch in range(0, epochs.__len__()):   
            
            # get current epoch 
            single_epoch = epochs[pick_epoch]
            
            # check if the participant started pressing the sensor before trial onset + 1 sec
            # get data from current epoch, count 0s at the beginning of the epoch
            # --> if the beginning of the epoch looks like this: 
            #     [0,0,0,0,0,0,0,0,0,0,2,3,0,0,0,2,3,4,5,6,3,4,5,9,...]
            #                          ^
            #                          |
            #
            # I assume the participant started using the sensor when the first 
            # "real" values (see amateurish arrow above) were recorded, doesn't matter if 
            # there are 0s afterwards. The reason is I can't know what happened there so 
            # if there are 0s afterwards, maybe the person willingly decided not to use the sensor.
            
            # get data from epoch as list
            epoch_data = list(single_epoch.get_data().flatten())
            
            # if there is at least 1 Zero at the beginning of the recording, 
            # count how many there are and go to next epoch (aka exclude current epoch) 
            # if there are more than 80 (aka 1s * 80Hz)
        
            if epoch_data[0] == 0:
                # counter for zeros
                nr_zeros = 0
                # break while loop condition
                count_on = True
                # start counting at index 0
                idx = 0
                while count_on:
                    # if current value is a 0,
                    if epoch_data[idx] == 0:
                        # add 1 to counter
                        nr_zeros += 1
                        # go to next value
                        idx += 1
                    # if curent value is not a 0...    
                    else: 
                        # stop counting
                        count_on = False
            # if the epochs doesn't start with a 0, we don't have to count            
            else: nr_zeros = 0            

            if nr_zeros >= 80:
                print("Participant didn't react or reacted to slow\nin epoch " + 
                      str(pick_epoch) + ", excluding this epoch now!" )
                # don't compute PSD for this epoch, save None 
                # instead as peak freq and power value:
                power.append(None)
                freq.append(None)
                
            # if everything's fine, compute PSD for current epoch and save results
            else:
            
                # Settings:
                            
                # this is just for testing this part of the script, 
                # I set these as arguments in the function call:
                # gss_psd_sfreq = 80 
                # gss_psd_fmin = 4 
                # gss_psd_fmax = 12 
                # gss_psd_n_overlap = 0 
                # gss_psd_n_per_seg = None 
                # gss_psd_n_jobs = 1
                # gss_psd_average = 'mean' 
                # gss_psd_window = 'hamming'


                # plot psd (no clue which Method this is, the MNE docs don't want me to know this)
                #single_epoch.plot_psd(fmin = 4, fmax = 12, spatial_colors = True)
                 
                
                """compute power spectral density (PSD) analysis using Welch's Method """
                
                # the following function returns... 
                # ...the PSD values for each channel (--> saved in object "psds")... 
                # ...for all frequencies (--> saved in object "freqs")... 
                # ...for the current epoch.
                
                # get data 
                single_epoch = np.array(single_epoch.get_data().flatten())
                
                psds, freqs = mne.time_frequency.psd_array_welch(single_epoch,
                                                                 sfreq = gss_psd_sfreq, 
                                                                 fmin = gss_psd_fmin, 
                                                                 fmax = gss_psd_fmax, 
                                                                 n_fft = len(single_epoch), 
                                                                 n_overlap = gss_psd_n_overlap, 
                                                                 n_per_seg = gss_psd_n_per_seg, 
                                                                 n_jobs = gss_psd_n_jobs, 
                                                                 average = gss_psd_average, 
                                                                 window = gss_psd_window)
                
                # round frequencies
                freqs = np.round(freqs, 1)
    
                # turn psds array into list
                psds = psds.tolist()
    
                # get highest power value and corresponding frequency
                peak_power = max(psds)
                peak_freq = freqs[psds.index(max(psds))]
    
                # save peak frequency & corresponding power value 
                # as temporary 1 row df:
                power.append(peak_power)
                freq.append(peak_freq)
                     
        # END loop epochs
        
        # append freq and power to gss_epochs_conditions
        gss_epochs_conditions["power"] = power
        gss_epochs_conditions["frequency"] = freq
        
        # create list with participant identifier and append to gss_epochs_conditions as well
        gss_epochs_conditions["ID"] = [participant] * len(gss_epochs_conditions)
        
        # append gss_epochs_conditions as new set of rows to df for all participants
        gss_PSDs_all = gss_PSDs_all.append(gss_epochs_conditions)
        
    # END loop participants
    
    
    
   
