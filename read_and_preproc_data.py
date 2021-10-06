# -*- coding: utf-8 -*-
"""

Preprocessing for EEG & grip strength sensor (gss) data

Part of Merle's Master Thesis
Version 1: 06.10.2021

"""
#-------------------------------------------------

""" settings """

""" 1.1 load packages """

# Install package to read xdf data
# I spent quite some time figuring this out: 
# You need to pip install pyxdf before being able to import it. 
# You can do this by copying one of the following lines into your terminal:

# Option 1: latest stable version (does not work for me)
# pip install pyxdf

# Option 2 (worked for me): latest developer version
# pip install git+https://github.com/xdf-modules/pyxdf.git
# I found this here: https://github.com/xdf-modules/pyxdf

# (The internet says you should install PyPDF2 
# to be able to import pyxdf, but that doesn't work)

# Now import package for reading in xdf data:
import pyxdf

# Python MNE for eeg data analysis / processing
# put this into your terminal to install python nme
    # pip install mne
# NME should be active by default and should already have the 
# necessary dependencies (numpy, scipy, and matplotlib).    
import mne

# import ICA function
from mne.preprocessing import ICA, find_eog_events

# glob for getting all files in a directory
import glob

# os for setting working directory
import os

# import pandas for using dataframes like I would in R (f*** you python)
import pandas as pd

# For getting means, SDs, sums,...
import numpy as np

# for turning nested list into 1D list
from itertools import chain


# --------------------------------------------------------

""" 1.2 set variables for the preproc"""

""" 1.2.1 pick channels """

# channels to pick up signal from auditory cortex: 
# 65, 1, 69 & the four neighboring electrodes 33, 38, 66 & 68 

# channels to pick up signal from right hand area of motor cortex: 
# 71 as center + neighboring electrodes 34, 77, 2, 5, 35 & 78

# visual cortex: 108 54 55 109 61 117 118 63    

eeg_channel_picks = ["EEG_065", "EEG_001", "EEG_069",
                     "EEG_71", "EEG_034", "EEG_077", 
                     "EEG_002","EEG_005", "EEG_035", 
                     "EEG_078", "EEG_108", "EEG_54", 
                     "EEG_55", "EEG_109", "EEG_61", 
                     "EEG_117", "EEG_118", "EEG_63"]

# choose eog channels (the 3 eeg channels from the face)
eog_channels = ["EEG_020", "EEG_031", "EEG_032"]


""" 1.2.2 set variables for the EEG data preproc """

# set filter type
# = default option; gives improved attenuation using fewer samples than “firwin2”
eeg_fir_design = 'firwin' 
# set type of window function
eeg_window_type = 'hamming'
# zero phase filter
eeg_phase = "zero"
# highpass filter 
eeg_bandpass_fmin = 5
# lowpass filter 
eeg_bandpass_fmax = 35,
# set number of runs to run in parallel
n_jobs = 1


""" 1.2.3 set variables for the gss data preproc """
# set filter design
sensor_fir_design = 'firwin' 
# set type of window function
sensor_window_type = 'hamming'
# highpass filter:
sensor_bandpass_fmin = 4
# lowpass filter
sensor_bandpass_fmax = 12
# use n_jobs from eeg settings (n_jobs = 1)

""" 1.2.4 settings for gss epoching """
# set cutoffs (in seconds) for epoching:
# get "4–12 Hz range since this range is where a majority 
# of the tremor is contained" (Archer et al., 2018)
sensor_prestim_cutoff = -1.5 
sensor_poststim_cutoff = 4 


""" 1.2.5 ICA settings """
# ICA method is fastICA (= default)
# use the first 5 components from the PCA (faster computation, no need to look at all 13 channels)
ica_n_components = 5
ica_decim = 5 # only use every 5th sample from the data --> speed up computation
ica_max_iter = "auto" 
# use a random seed to get the same results every time we run the ICA on the same data
ica_random_state = 97 


# --------------------------------------------------------

""" 2. read in data """

# working directory
os.chdir("/Users/merle/Desktop/Masterarbeit/Master_Testdaten/")

# get list of all xdf files in my directory 
# (the asterix in the path means the name of the 
# file can be anything as long as it has an .xdf ending)
file_list = glob.glob("/Users/merle/Desktop/Masterarbeit/Master_Testdaten/*.xdf")

# set number of subjects as number of xdf files in directory
subj_n = len(file_list)

# count participants 
participant = 1

# Create empty lists to keep track of plots (before and after filtering)
#eeg_figs_before_filtering = []
#eeg_figs_after_filtering = []

# loop xdf file names in file_list aka participants:
for file_name in file_list:
    
    """ 2.1 read in XDF data """
    streams, header = pyxdf.load_xdf(file_name)

    # Raise a hue & cry if data doesn't have 4 streams!
    assert len(streams) == 4  # 1 EEG markers (?), 1 EEG, 1 stim channel, 1 gss channel

    
    """ 2.2 Build NME data object from scratch """
    # stream 0: general info? actiCHampMarkers (whatever that is)
    # stream 1: Actichamp - EEG data
    # stream 2: PsychoPyMarkers - Experiment markers
    # stream 3: Arduino - Grip strength sensor data    

    # each stream contains timestamps (measured in seconds)
    
    
    """ 2.2.1 Create info for Raw Object for EEG data"""
    # Sampling rate: 500 Hz
    sampling_freq = float(streams[1]["info"]["nominal_srate"][0]) # in Hertz
    
    # name and classify channels
    ch_names = [f'EEG_{n:03}' for n in range(1, 129)]
    ch_types = ['eeg'] * 128 

    # combine information 
    info_eeg = mne.create_info(ch_names, 
                               ch_types = ch_types, 
                               sfreq = sampling_freq)
    
    # add name of the curent dataset (I could also add information like age or gender here)
    # (change this if the files names of the exp files are named differently)
    info_eeg['description'] = file_name[len(file_name)-30 : len(file_name)-4 : 1]
   
    # look at the info
    #print(info_eeg)
   


    """ 2.2.2 Get EEG data for Raw object""" 
    # structure should be: 
        # rows: channels
        # columns: sample points
    
    # get EEG data from stream 1:
    # 128 arrays (1 for each electrode), 186013 sampling points
    data_eeg = np.array(streams[1]["time_series"].T) 

    # transform all values in eeg_data from Microvolt to Volt 
    # as NME expects EEG data to be measured in Volt (why tho)
    data_eeg[:] *= 1e-6
    
        
    """ 2.2.3 Create Raw object for EEG data""" 
    # combine info & eeg data
    eeg_Raw = mne.io.RawArray(data_eeg, info_eeg)


    """ 2.2.4 change the 3 EEG channels in the face to EOG channel type """
    # check channel types before converting them to eog:
    #eeg_Raw.get_channel_types("EEG_020")
    # convert channel type of the 3 electrodes in question to eog
    eeg_Raw.set_channel_types(mapping = {"EEG_031" : "eog", "EEG_032" : "eog", "EEG_020" : "eog"})
    # This raises an error message, I don't know why but it works so I don't care rn. 
    # Check channel type again:
    #eeg_Raw.get_channel_types("EEG_020")


    """ 2.2.5 Add Annotations """
       
    # If you look at the first timestamp of stream 1, 2 & 3, you can see that 
    # they don't match. The EEG for example started recording 
    # way earlier than the Arduino:
    #streams[1]["time_stamps"][0] 
    #streams[3]["time_stamps"][0] 
   
    # This means I need to include information on when the 
    # Triggers and GSS values started or else MNE assumes 
    # the streams all started at the same time, which is not the case.
   
    # If you look at when the streams were created, 
    # you see that they are not 0, which means they're 
    # probably all relative to some shared event 
    # (like turning on the computer? Idk.)
    #streams[3]['info']['created_at']
    #streams[2]['info']['created_at']
    #streams[1]['info']['created_at'] 
   
    # Solution: Set onset of EEG stream to None (that's the default anyway)
    # and subtract onset of EEG stream from GSS & Trigger data. 
    # This way the timestamps are relative to the EEG onset.
       
    # get difference between EEG onset and onset of Triggers
    eeg_onset = streams[1]["time_stamps"][0] 
    trigger_timestamps = streams[2]["time_stamps"] - eeg_onset
    gss_timestamps = streams[3]["time_stamps"] - eeg_onset
    

    """ Get Triggers & save as Annotations object """

    # get names of triggers (it's a nested list in the xdf file)    
    trigger_descriptions = streams[2]["time_series"]
    # turn nested list into "normal" one dimensional list
    trigger_descriptions = list(chain.from_iterable(trigger_descriptions)) 
    
    # change trigger descriptions so the trial start descriptions 
    # contain info on the block & feedback condition
    
    # start with block 0:
    block_nr = "block0"

    for i in range(len(trigger_descriptions)):
        trigger_descr = trigger_descriptions[i]
        
        # check if a new block begins:
        # If the current trigger starts with "block", but doesn't correspond to the current block,
        # change name of current block.
        if trigger_descr[0:5] == "block" and trigger_descr!= block_nr:
            # save new block name
            block_nr = trigger_descr   
        # ELSE (aka if the trigger is not a new block name):
        # if trigger name starts with ep (like epoch)...
        elif trigger_descr[0:2] == "ep":
            #... save feedback condition of the current epoch
            curr_epoch = trigger_descr[6:-6]
        # else if the name starts with t (as in trial_start)...    
        elif trigger_descr[0] == "t":
            #... concatenate trial name and epoch name & block number, divided by an underscore
            trigger_descriptions[i] = trigger_descr + "_" + curr_epoch + "_" + block_nr
        
    # save trigger descriptions & their onsets as annotations for our Raw object
    triggers_annot = mne.Annotations(onset = trigger_timestamps, duration = .001, description = trigger_descriptions)
    

    """ Get GSS values & save as Annotations object """
    # (I think I don't even need them in these datasets, but better safe than sorry)
    # get gss values (it's a nested list in the xdf file)    
    gss_values = streams[3]["time_series"]
    
    # turn nested list into "normal" one dimensional list
    gss_values = list(chain.from_iterable(gss_values)) 
    
    # save GSS values & their onsets as annotations
    gss_annot = mne.Annotations(onset = gss_timestamps, duration = 0.001, description = gss_values) 
    

    """ 2.2.6 Automatically detect blinks, name them "bad blink" & save as Annotations object """
    
    # find_eog_events filters the data before detecting oeg events:
    # FIR two-pass forward and reverse, zero-phase, 
    # non-causal bandpass filter from 1 - 10 Hz with von Hann window
        
    # find all eog events
    eog_events = mne.preprocessing.find_eog_events(eeg_Raw)
    
    # set onset as 250 ms before the peak of the blink
    onsets = eog_events[:, 0] / eeg_Raw.info['sfreq'] - 0.25
    
    # set duration of blink epoch (500 ms around peak)
    durations = [0.5] * len(eog_events)
    
    # create description for blinks 
    # (all annotations that start with "bad" 
    # can be automatically rejected later)
    blink_descriptions = ['bad blink'] * len(eog_events)
    
    # create annotation object for the detected blinks
    blink_annot = mne.Annotations(onsets, durations, blink_descriptions, 
                                  orig_time = eeg_Raw.info['meas_date'])
    
    """ Set annotations """
    # there's a bug in MNE, you have to set a filename 
    # to be able to set annotations:
    eeg_Raw._filenames = ['a_random_file_name'] 

    # include gss values as annotations:
    # eeg_Raw.set_annotations(triggers_annot + blink_annot + gss_annot)
    # only use triggers & blinks, not the gss values:
    eeg_Raw.set_annotations(triggers_annot + blink_annot)


    """ 2.2.5 plot raw data """
    # (careful, this takes a lot of time if you include the gss values!)
    # I filtered the data a bit, but this is only for the 
    # representation in the plot, it doesn't affect the data!
    #raw_plot = eeg_Raw.plot(highpass = 1, lowpass = 35, show_scrollbars = True)

    """ save raw data in old_eeg_Raw before preprocessing """
    old_eeg_Raw = eeg_Raw.copy()
    # also save object to disc?
   
    
   
# --------------------------------------------------------
    """ 3. Preprocessing """
    # (I set the variables used here at the beginning of this script)
    
    """ 3.1 pick the right channels """
    # Pick EEG channels
    eeg_Raw.pick_channels(eeg_channel_picks)



    """ 3.2 ICA to get rid of blinks in the EEG data """
    
    # highpass filter the data 
    # (the quality of the ICA fit is negatively affected by 
    # low-freq drifts, so this is important!)
    ica_filt_raw = eeg_Raw.copy()
    ica_filt_raw.load_data().filter(l_freq = 1., h_freq = None)

    # set up ICA (variables are defined at the beginning of the script)
    ica = ICA(n_components = ica_n_components, max_iter = ica_max_iter, random_state = ica_random_state)

    # fit ICA
    ica.fit(ica_filt_raw)

    # check what the ICA captured
    eeg_Raw.load_data()
    # ica.plot_sources(ica_filt_raw, show_scrollbars = False)
    # --> first ICA channel (ICA000) seems to capture the blinks quite well
    
    # exclude first ICA component:
    # (I assume it'll be the 1st ICA channel for all participants)
    ica.exclude = [0] 
    # apply ICA to eeg_Raw 
    ica.apply(eeg_Raw)


    """ 3.3 filter EEG data """
    # (variables are defined at the beginning of the script))
    eeg_Raw.filter(picks = eeg_channel_picks,
                   l_freq = eeg_bandpass_fmin, 
                   h_freq = eeg_bandpass_fmax,   
                   phase = eeg_phase,
                   fir_window = eeg_window_type, 
                   fir_design = eeg_fir_design, 
                   n_jobs = n_jobs)
    













    
    """ 3.4 Epoching """    

    # max_force_xxx -> initial maximum grip force of participant
    # block -> Blocks 0 - 3
    # Block 0 = training 
    # Block 1 & 2 = active condition
    # Block 3 = passive condition (just watching the stimuli)

    # 5 triggers per trial:
    #    1. epoch_vo_start -> start of epoch (like sub-block)
    #            vo = only visual, av = audiovisual, ao = only audio
    #    2. baseline 
    #    3. fix_cross
    #    4. trial_start_sfb_0.3_sfc_0.25
    #        sfb = scaling feedback  
    #        sfc = % of max grip force (e.g. 0.25%)
    #    5. end_trial

    """ get block onsets & crop eeg_Raw to seperate blocks """
    b0_onset = trigger_timestamps[trigger_descriptions.index("block0")]
    b1_onset = trigger_timestamps[trigger_descriptions.index("block1")]
    
    #save data from block 0 (training)
    b_test_Raw = eeg_Raw.copy().crop(tmin = b0_onset, tmax = b1_onset)
    
    # save data from blocks 1 - 3
    b_main_Raw = eeg_Raw.copy().crop(tmin = b1_onset)
   
    
   # --> use tmax as onset of b3

    """ PROBLEM: I need the onset of b3, but I only have training & block 1! """
   
    
   
    
   
    
   
    
   
    """ create epochs, only use data from blocks 1 & 2 """
    
        
    """ create events from annotations """
    # use regular expression to look for strings beginning with t (I only need the trial starts)
    # Also, round strings instead of truncating them so we get unique time values
    trial_events, trial_event_id = mne.events_from_annotations(b_main_Raw, regexp = '(^[t]).*$', use_rounding = True)    
       
    
    """ create metadata for each epoch containing information on sfb, sfc and feedback condition """

    epoch_colnames = [f'epoch_{n:03}' for n in range(1, len(list(trial_event_id.keys()))+1)]
    eeg_epochs_metadata = pd.DataFrame(columns = epoch_colnames)
    
    # row 0: feedback condition (ao = audio only, vo = visual only, or av = audiovisual)
    # row 1: sfb = feedback scaling 
    # row 2: sfc = % of max grip force
    
    row_feedback = [] 
    row_sfb = []
    row_sfc = []
    
    for key in trial_event_id.keys():
        # get feedback conditio
        feedback = key[-2:len(key)]
        
        # get sfb value (= scaling of feedback)
        sfb = key[16:20]
        # if value doesn't have 2 digits after the comma...
        if sfb[-1] == "_":
            # remove underscore
            sfb = sfb[0:-1]

        # same for sfc (= % of max. grip force)
        sfc = key[-7:-3]
        # if there's an underscore before the value...
        if sfc[0] == "_":
            # remove underscore
            sfc = sfc[1:len(sfc)]

        # append to the rows, typecast sfb & sfc values to float
        row_feedback.append(feedback) 
        row_sfb.append(float(sfb))
        row_sfc.append(float(sfc))

    # set as rows in the eeg_epochs_metadata df
    eeg_epochs_metadata.loc[0] = row_feedback
    eeg_epochs_metadata.loc[1] = row_sfb
    eeg_epochs_metadata.loc[2] = row_sfc

    # Aaaand I made a mistake, the metadata must have the same 
    # number of rows as events, but I have the same number of columns as events.
    # Transpose the whole thing because I'm lazy:
    eeg_epochs_metadata = eeg_epochs_metadata.transpose()  
    
    eeg_epochs_metadata.columns = ["feedback", "sfb", "sfc"]
    
    """ get epochs, apply baseline correction on the fly """
    # event = trial start, cut from -1.5 to +7           
    trial_epochs = mne.Epochs(b_main_Raw, trial_events, trial_event_id, 
                              tmin = - 1.5, tmax = 7, baseline = (-1.5, 0), 
                              preload = True, event_repeated = "drop",
                              reject_by_annotation = False, metadata = eeg_epochs_metadata) # metadata = pass
    
    # plot the epochs (only plot the first 2 or else it gets super messy)
    trial_epochs.plot(show_scrollbars = True, n_epochs = 2)
    
    # this is how you can select epochs based on the metadata
    trial_epochs['sfc == 0.25'].plot()
    
    # or select multiple values
    search_values = ['0,25', '.3']

    # more on how to use the metadata: 
    # https://mne.tools/dev/auto_tutorials/epochs/30_epochs_metadata.html

    # save Raw object & epoched data for each participant in the file 
    # I set as the working directory at the beginning of the script
    eeg_Raw.save(fname = "Raw_eeg_participant" + str(participant) + ".fif", fmt = 'single', overwrite = False)
    trial_epochs.save(fname = "eeg_epochs_participant" + str(participant) + ".fif", fmt = 'single', overwrite = False)

    # next participant!
    participant =+ 1


# END LOOP     


# TO DO: 
# add GSS preproc
# add / save plots




    




    """ analyze GSS data """

    # create raw object + trigger annotations for gss data
    
    # ICA for filtering out heartbeats? 
    
    # filter gss data






""" ---------------------- Useful stuff I might need later ------------------------"""

    
# Arduino Sampling Frequency
# to get the sampling frequency of the Arduino, I took 400 timestamp values and 
# their "neighbors" and calculated the difference to get the mean time that passes between the samples. 
# If you divide 1 by this value and round it you get a sampling rate of 45 Hz:   
sampling_freq_gss = np.round(1/np.mean(streams[3]["time_stamps"][300:700] - streams[3]["time_stamps"][299:699]))

# quick check: if you do the same for the EEG timestamps, 
# you get 500 as a result, which corresponds to the sampling rate from the xdf info, so this is correct. 
#np.round(1/np.mean(streams[1]["time_stamps"][300:700] - streams[1]["time_stamps"][299:699]))

