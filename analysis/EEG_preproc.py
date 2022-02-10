"""
Function for preprocessing EEG data (from fif files)

Part of Merle's Master Thesis
Version 1: 13.01.2022

Input: File containing .fif files with EEG data + triggers for each participant
Output: 
        - .fif file containing MNE epochs object with filtered EEG data

"""
#%%  

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

#%% 
# create function to filter + epoch data automatically
def EEG_filter_epoching(working_directory, 
                        eeg_channel_picks, 
                        eog_channels,
                        eeg_fir_design = "firwin", 
                        eeg_window_type = "hamming", 
                        eeg_phase = "zero", 
                        eeg_bandpass_fmin = 4, 
                        eeg_bandpass_fmax = 35, 
                        n_jobs = 1, 
                        eeg_prestim_cutoff = -1.5, 
                        eeg_poststim_cutoff = 4, 
                        ica_n_components = 5, 
                        ica_max_iter = "auto", 
                        ica_random_state = 97,
                        eeg_epochs_tmin = -1.5,
                        eeg_epochs_tmax = 7,
                        eeg_epochs_baseline_start = -1.5,
                        eeg_epochs_baseline_stop = 0):    
    
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.

#%%     
    
    """ 1. load packages """
       
    # Python MNE for eeg data analysis / processing:
    # NME should be active by default and should already have the 
    # necessary dependencies included (numpy, scipy, and matplotlib).    
    import mne
    
    # glob for getting all files in a directory:
    import glob
    
    # os for setting working directory:
    import os
    
    # import ICA function 
    from mne.preprocessing import ICA
    
    # pandas for dataframes
    import pandas as pd

#%%     
    """ 2. set working directory """
    os.chdir(working_directory)
    
    """ 3. get list of all .fif files in my directory """
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list = glob.glob(working_directory + "eeg_participant" + "*_raw.fif")

    """ 4. keep track of files """
    file = 0
    
 #%%      
    """ 4. loop fif file names in file_list (aka loop participants): """
    for file_name in file_list:
        
        """ save participant number"""
        # the participant numbers in file list are in the wrong order
        # so get the number from the filename instead of the loop index number

        participant = file_name[-11:-8]
        
        # if participant number has < 2 or 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]
        
        
        """ 4.1 read in fif file """
        eeg_Raw = mne.io.read_raw_fif(file_name)

#%%  
        """ 4.2 Preprocessing """
        # (variables used here are set in the main script)
    
        """ 4.2.1 pick the right channels """
        # Pick EEG channels
        eeg_Raw.pick_channels(eeg_channel_picks)

#%%  
        
        """ 4.2.2 ICA to get rid of blinks in the EEG data """
        # highpass filter the data 
        # (the quality of the ICA fit is negatively affected by 
        # low-freq drifts, so this is important!)
        ica_filt_raw = eeg_Raw.copy()
        ica_filt_raw.load_data().filter(l_freq = 1., h_freq = None)

        # set up ICA (variables are defined at the beginning of the script)
        ica = ICA(n_components = ica_n_components, 
                  max_iter = ica_max_iter, 
                  random_state = ica_random_state)

        # fit ICA
        ica.fit(ica_filt_raw)

        # check what the ICA captured
        eeg_Raw.load_data()
        # ica.plot_sources(ica_filt_raw, show_scrollbars = True)
        # --> first ICA channel (ICA000) seems to capture the blinks quite well
    
        # exclude first ICA component:
        # (I assume it'll be the 1st ICA channel for all participants)
        ica.exclude = [0] 
        # apply ICA to eeg_Raw 
        ica.apply(eeg_Raw)

 #%%  
        """ 4.2.3 filter EEG data """
        # (variables are defined at the beginning of the script))
        eeg_Raw.filter(picks = eeg_channel_picks,
                       l_freq = eeg_bandpass_fmin, 
                       h_freq = eeg_bandpass_fmax,   
                       phase = eeg_phase,
                       fir_window = eeg_window_type, 
                       fir_design = eeg_fir_design, 
                       n_jobs = n_jobs)

    #%%       
        """ 4.2.4 Epoching """    
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
        
   #%%        
        """ 4.3 Get Triggers & save as Annotations object """
        
        # get trigger timestamps and trigger descriptions
        trigger_descriptions = eeg_Raw.annotations.description.tolist()
        trigger_timestamps = eeg_Raw.annotations.onset.tolist()    
        
   #%%     
        """ 4.4 get block onsets & crop eeg_Raw to seperate blocks """
        b0_onset = trigger_timestamps[trigger_descriptions.index("block0")]
        
        # if the trigger block1 is in the list of trigger descriptions, save onset.
        if "block1" in trigger_descriptions:
            b1_onset = trigger_timestamps[trigger_descriptions.index("block1")]
            
        # if the trigger block3 is in the list of trigger descriptions, save onset.
        if "block3" in trigger_descriptions:
            b3_onset = trigger_timestamps[trigger_descriptions.index("block3")]
            
        #save data from block 0 (training)
        #b_test_Raw = eeg_Raw.copy().crop(tmin = b0_onset, tmax = b1_onset)
    
        """ exclude training block and block 3 """
        # (or just exclude training, if there is no block 3)
        if "block3" in trigger_descriptions:
            b_main_Raw = eeg_Raw.copy().crop(tmin = b1_onset, tmax = b3_onset)
        else:
            b_main_Raw = eeg_Raw.copy().crop(tmin = b1_onset)

 #%%  
        """ 4.5 create epochs, only use data from blocks 1 & 2 """
        
        """ create events from annotations """
        # use regular expression to look for strings beginning with t (I only need the trial starts)
        # Also, round strings instead of truncating them so we get unique time values
        trial_events, trial_event_id = mne.events_from_annotations(b_main_Raw, regexp = '(^[t]).*$', use_rounding = True)    
       

        """ create metadata for each epoch containing information on sfb, sfc and feedback condition """
    
        # create epoch names (--> epoch_001, epoch_002, and so on)
        epoch_colnames = [f'epoch_{n:03}' for n in range(1, len(list(trial_event_id.keys()))+1)]
        # create empty df for metadata with the epoch names as column names
        eeg_epochs_metadata = pd.DataFrame(columns = epoch_colnames)
        
        # row 0: feedback condition (ao = audio only, vo = visual only, or av = audiovisual)
        # row 1: sfb = feedback scaling 
        # row 2: sfc = % of max grip force
        
        row_feedback = [] 
        row_sfb = []
        row_sfc = []
        
        for key in trial_event_id.keys():
            # get feedback condition (ao, vo or av)
            feedback = key[-9:len(key)-7]
            
            # get sfb value (= scaling of feedback; e.g. "0.86")
            sfb = key[16:20]
            # if value doesn't have 2 digits after the comma...
            if sfb[-1] == "_":
                # remove underscore
                sfb = sfb[0:-1]
    
            # same for sfc (= % of max. grip force; e.g. "0.25")
            sfc = key[-14:-10]
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
        # with baseline form -1.5 - 0           
        trial_epochs = mne.Epochs(b_main_Raw, 
                                  trial_events, 
                                  trial_event_id, 
                                  tmin = eeg_epochs_tmin, 
                                  tmax = eeg_epochs_tmax, 
                                  baseline = (eeg_epochs_baseline_start, eeg_epochs_baseline_stop), 
                                  preload = True, 
                                  event_repeated = "drop",
                                  reject_by_annotation = False, 
                                  metadata = eeg_epochs_metadata) # metadata = pass
        
        # plot the epochs (only plot the first 2 or else it gets super messy)
        #trial_epochs.plot(show_scrollbars = True, n_epochs = 2)
        
        # this is how you can select epochs based on the metadata:
        #trial_epochs['sfc == 0.25'].plot()
        # ...or select multiple values:
        #search_values = ['0,25', '.3']
        # more on how to use the metadata: 
        # https://mne.tools/dev/auto_tutorials/epochs/30_epochs_metadata.html

#%%   
        """ 5. save Raw object & epoched data for each participant in the file """
        # I set as the working directory at the beginning of the script
        trial_epochs.save(fname = "eeg_participant" + str(participant) + "_epo.fif", fmt = 'single', overwrite = False)
    
        # save number of processed files
        file += 1

    # END LOOP   

#%%      
    
    """ 8. Create "I'm done!"-message: """
    if file == 0:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, something went wrong: I couldn't run the preproc function on any file. Maybe you should have a look at this.\n\n- - - - - - - - - - - - - - - - - - - - - ")
    
    elif file == 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched 1 file for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the file eeg_participant1_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    else:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched " + str(file) + " files for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the files eeg_participant[number]_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    
# END FUNCTION

