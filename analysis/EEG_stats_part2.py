
""" Stats script for Merle's Master's Thesis

Stats for EEG data
Part 2: compute coherences

Part of Merle's Master Thesis
Version 1: 10.02.2022

"""


#%%
#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
#eeg_coh_sf = 80
#eeg_coh_window = 'hann' 
#eeg_coh_detrend = 'constant'
#eeg_coh_axis = - 1
#eeg_coh_tmin = 1
#eeg_coh_tmax = 4


#%%

# create function for running EEG stats script
def EEG_stats_coherences(working_directory, 
                         eeg_coh_sf = 80, 
                         eeg_coh_window = 'hann', 
                         eeg_coh_detrend = 'constant', 
                         eeg_coh_axis = - 1,
                         eeg_coh_tmin = 1, 
                         eeg_coh_tmax = 4):
    
        
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.
    
    # eeg_coh_axis = Axis along which the coherence is computed 
    # for both inputs; default is over the last axis (axis = -1)

    # eeg_coh_tmin and eeg_coh_tmin = onset & offset of part of the epoch we 
    # want to compute coherences for; e.g. epoch from -1.5 - 7, -1.5 - 0 is baseline, 
    # signal for coherence analysis should be 
    # from 1 - 4, use eeg_coh_tmin = 1 and eeg_coh_tmax = 4

#%%    
    """ Settings """

    """ 1. load packages """
    
    
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
    
    # for computing coherences:
    from scipy import signal
    
    # for arrays & maths functions
    import numpy as np

    # for dataframes as in R
    import pandas as pd

    # for plotting
    import matplotlib.pyplot as plt
    
    # for turning nested array into 1D array
    from itertools import chain
    
    # create small function to find element in numpy array 
    # that's nearest to a certain value:
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


#%%

    """ 2. read in data """
    
    # 2.1 set path to working directory
    os.chdir(working_directory)
    
    # 2.2 get list of all files in the directory that start 
    # with "eeg" and end with "epo.fif"
    file_list = glob.glob(working_directory + "eeg*epo.fif")
    
    # 2.3 create df for coherence values
    coherences_all = pd.DataFrame(columns = ["participant", "epoch", 
                                             "feedback", "sfb", "sfc", 
                                             "ROI1", "ROI2", "frequency_band", 
                                             "coherence"])
    
    # 2.4 set ROIs:
    auditory_ROI = ["EEG_001", "EEG_069", "EEG_068","EEG_033"
                    "EEG_038","EEG_066", "EEG_065"]
    
    motor_ROI = ["EEG_034", "EEG_002", "EEG_071", "EEG_077", 
                 "EEG_005", "EEG_035", "EEG_078"]
    
    visual_ROI = ["EEG_108", "EEG_054", "EEG_055", "EEG_061", 
                  "EEG_117", "EEG_118", "EEG_109", "EEG_063"]

    
 
#%%   
    
    """ 3. Loop participants """
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
        participant = file


#%%
        """ loop epochs """
        for pick_epoch in range(0, epochs.__len__()):
                    
            # get single epoch, crop to get time interval from second 1 - second 4 
            single_epoch = epochs[pick_epoch].crop(tmin = eeg_coh_tmin, tmax = eeg_coh_tmax)
            
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
          
            # loop channels, get average signal for each ROI, save averaged signal
            
            # collect 
            data_motor = []
            data_auditory = []
            data_visual = []
            
            for channel in range(0, len(channels)):    
                
                # get data for the current channel
                channel_epoched_data = list(single_epoch._data[0][channel])
                
                # get channel name
                channel_name = single_epoch.ch_names[channel]

                # check which ROI this channel belongs to and append 
                # to list of epoch data
                if channel_name in motor_ROI:
                    data_motor.append(channel_epoched_data)
                
                elif channel_name in auditory_ROI:
                    data_auditory.append(channel_epoched_data)
                    
                elif channel_name in visual_ROI:
                    data_visual.append(channel_epoched_data)

            # get average signal for each ROI
            #(Hint: * is the splat operator, it expands a list into 
            # actual positional arguments in the function call)
            motor_avg_signal = np.array([np.mean(x) for x in zip(*data_motor)])
            auditory_avg_signal = np.array([np.mean(x) for x in zip(*data_auditory)])
            visual_avg_signal = np.array([np.mean(x) for x in zip(*data_visual)])
                
#%%
            """ Compute coherences: Which ROIs are in the different feedback conditions functionally connected?"""
            
            # for all possible combinations of ROIs, compute coherence
            pairs = [("motor", "visual"), ("motor", "auditory"), ("visual", "auditory")]
            
            for ROI1, ROI2 in pairs:
                
                # get data for ROI1
                if ROI1 == "motor":
                    x = motor_avg_signal
                else:
                    x = visual_avg_signal
                
                # get data for ROI2
                if ROI2 == "auditory":
                    y = auditory_avg_signal
                else:
                    y = visual_avg_signal
                                
                # compute coherence, get coherence value 
                # for each frequency:
                f, Cxy = signal.coherence(x = x, 
                                          y = y, 
                                          fs = eeg_coh_sf, 
                                          window = eeg_coh_window,
                                          detrend = eeg_coh_detrend, 
                                          axis = eeg_coh_axis)

                # Plot it!
                #plt.plot(f, Cxy)
                #plt.xlabel('frequency [Hz]')
                #plt.ylabel('Coherence')
                #plt.show()

                """ get average coherence for each freq band """
                # I only want to see the coherences between 4 and 
                # 35 Hz, for the theta, alpha, beta and gamma band (not 
                # single frequencies), so get average coherence
                # for each freq band:
                    
                # A bit hardcoding but who cares:
                # 1. theta: 4 - 7
                # 2. alpha: 8 - 12
                # 3. beta: 13 - 30)   
                # 4. gamma: > 30, I'll just use 31 - 35 so it matches 
                # the range of the PSD in stats part 1
                
                # round frequencies:
                f = np.round(f, 1)
                
                # create list of freq band names
                freq_bands = ["theta", "alpha", "beta", "gamma"]   
                # create list of on- and offsets (include onset, exclude offset)
                freq_band_ranges = [(4, 8), (8, 13), (13, 31), (31, 35)]   
                
                # loop frequency bands, get average coherence in each freq
                # band and add to df:
                for freq_band_idx in range(0, len(freq_bands)):
                    
                    # find index of onset & offset of freq band:
                    
                    # get intended onsets and offsets
                    onset, offset = freq_band_ranges[freq_band_idx]
                
                    # find onset index
                    onset_idx = int(np.where(f == find_nearest(f, onset))[0])
                    # if the found onset is smaller than the indended onset, 
                    # it falls into another freq band, so take the next value instead.
                    if f[onset_idx] < onset:
                        onset_idx += 1
                        
                    # find offset index
                    offset_idx = int(np.where(f == find_nearest(f, offset))[0])
                    # if the found offset is bigger than the indended offset, 
                    # it falls into another freq band, so take the value before instead.
                    if f[offset_idx] >= offset:
                        offset_idx -= 1
                     
                        
                    # get all coherence values between onset & offset 
                    # (values from onset and offset idx included)
                    # compute average coherence
                    coh_val = np.mean(Cxy[onset_idx : offset_idx])

                    # put everything into df
                    coherences_all.loc[len(coherences_all)] = [participant, pick_epoch, 
                                                             feedback, sfb, sfc, 
                                                             ROI1, ROI2, freq_bands[freq_band_idx], 
                                                             round(coh_val, 3)]

    # Let's assume everything works and higher sfcs --> higher tremor amplitudes. 
    # I don't know anything about coherences but I'd assume that the stronger the influence of 
    # the feedback the higher the communication between the ROIs should be.
    # Test this? Or just compute average coherences and leave it at that?


