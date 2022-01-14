"""
Function for preprocessing GSS data (from fif files)

Part of Merle's Master Thesis
Version 1: 13.01.2022

Input: File containing .fif files with GSS data + triggers for each participant
Output: .fif file containing MNE epochs object with filtered & epoched GSS data
"""

#%% 
#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# create function to filter + epoch data automatically
def GSS_filter_epoching(working_directory, 
                        gss_bandpass_fmin = 4, 
                        gss_bandpass_fmax = 12, 
                        gss_phase = "zero", 
                        gss_window_type = 'hamming', 
                        gss_fir_design = 'firwin', 
                        gss_n_jobs = 1, 
                        gss_prestim_cutoff = -1.5, 
                        gss_poststim_cutoff = 4):
      
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.
    
    """ 1. load packages """
       
    # Python MNE for eeg data analysis / processing:
    # NME should be active by default and should already have the 
    # necessary dependencies included (numpy, scipy, and matplotlib).    
    import mne
    
    # EMD for Empirical Mode Decomposition
    # please make sure you pip installed emd:
    #pip install emd
    import emd
    
    # glob for getting all files in a directory:
    import glob
    
    # os for setting working directory:
    import os
    
    # pandas for dataframes
    import pandas as pd
    
    # for plotting
    import matplotlib.pyplot as plt
    
    # for converting arrays into lists
    import numpy as np
    
    # import ICA function 
    from mne.preprocessing import ICA
    
    #%% 
    """ 2. set working directory """
    os.chdir(working_directory)
    
    """ 3. get list of all gss .fif files in my directory """
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list = glob.glob(working_directory + "Raw_gss_participant" + "*.fif")
    
    """ 4. Create empty lists to keep track of plots (before and after filtering)"""
    #gss_figs_before_filtering = []
    #gss_figs_after_filtering = []
    
    """ 5. keep track of files """
    file = 0
    
   #%%  
    """ 6. loop fif file names in file_list (aka loop participants): """
    for file_name in file_list:
        
        """ save participant number"""
        # the participant numbers in file list are in the wrong order
        # so get the number from the filename instead of the loop index number
    
        participant = file_name[-7:-4]
        
        # if participant number has < 2 or 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]
        
        
        """ 6.1 read in fif file """
        gss_Raw = mne.io.read_raw_fif(file_name)
        gss_values = gss_Raw[:][0][0]
                
        # plot raw data:
        # turn gss_data into list instead of np.array
        #gss_values = gss_values.tolist()
            
        # get time stamps
        #time_vect = list(gss_Raw[:][1])
            
        #plt.plot(time_vect, gss_values)
        #plt.show()
    
 #%%    
        """ 6.2 Preprocessing """
        # (variables used here are set in the main script)

        
        """ 6.2.1 empirical mode decomposition """
        # (to get rid of motor / heart artifacts)
        # Wich different oscillations are there in the signal 
        # that are non-linear or non-stationary? 
        #--> https://emd.readthedocs.io/en/stable/
    
        # estimate IMFs (=intrinsic mode functions) 
        # for the gss signal...
        imf = emd.sift.sift(gss_values)
    
        # In the imf object, there are 12707 arrays containing 9 values each.
        # So there are basically 9 time-frequency channels here.
        # Have a look:
        #print(imf.shape)
            
        # plot it! 
        emd.plotting.plot_imfs(imf, scale_y = True, cmap = True)    
    

        # I don't need the following bit rn, but could be useful later:
    
        # ...and, from the IMFs, compute the instantaneous 
        # frequency (IF), phase (IP) and amplitude (IA) using the 
        # Normalised Hilbert Transform Method:
        #IP, IF, IA = emd.spectra.frequency_transform(imf, 45, 'hilbert')
    
        # From the instantaneous frequency and amplitude, we can compute the Hilbert-Huang spectrum:
        # Define the frequency range (low_freq, high_freq, nsteps, spacing)
        #freq_range = (0.1, 10, 80, 'log')
        
        # Now compute the Hilbert-Huang spectrum:
        #f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time = False)

        # plot summary information, first the IMFs...
        #emd.plotting.plot_imfs(imf, scale_y = True, cmap = True)
        
        # ...and now the Hilbert-Huang transform of this decomposition:
        #fig = plt.figure(figsize=(10, 6))
        #emd.plotting.plot_hilberthuang(hht, 
        #                               time_vect, 
        #                               f,
        #                               time_lims = (2, 4), 
        #                               freq_lims = (0.1, 15),
        #                               fig = fig, 
        #                               log_y = True)
 
    
#%% 
        """ 6.2.2 Turn IMFs into MNE RAW object again"""        
        #create Raw objects for the 9 IMF "channels"


        """ get data for IMF Raw object """
        
        # get IMF values:
        # create list with data from each channel, put them all in one "main" list
        
        imf_channels = list()
        channel_names = list()
        channel_types = list()
        
        for n in range(0, len(imf[0])) :
            # use list comprehension to get nth element of each array in imf
            # n = number of imf "channel"
            curr_channel = [i[n] for i in imf]
            imf_channels.append(curr_channel)
            
            # collect info data for Raw object
            channel_names.append("IMF_" + str(n+1))
            channel_types.append("eeg")
            
        # get timestamps:
        time_vect = list(gss_Raw[:][1])
            
        # get sampling frequency of the GSS data:
        # to get the sampling frequency of the Arduino, I took 400 timestamp values and 
        # their "neighbors" and calculated the difference to get the mean time that passes between the samples. 
        # If you divide 1 by this value and round it you get a sampling rate of 45 Hz for the GSS data:          
        sampling_freq_gss = np.round(1/np.mean(np.array(time_vect[300:700]) - np.array(time_vect[299:699])))
            
        """ create info object for IMF Raw object """ 
        info_imf = mne.create_info(ch_names = channel_names, ch_types = channel_types, sfreq = sampling_freq_gss)

        # look at the info
        #print(info_imf)
           
        
        # transform all values in eeg_data from Microvolt to Volt 
        # as NME expects EEG data to be measured in Volt (why tho)
        #data_gss[:] *= 1e-6
    
        """ 4.2.3 Create Raw object for IMF data""" 
        #combine info & eeg data
        IMF_Raw = mne.io.RawArray(imf_channels, info_imf)
        
  #%%     
        """ 6.2.3 ICA """
        # Now run ICA on the IMFs!

        # Normally I'd highpass filter the data as the quality of the 
        # ICA fit is negatively affected by low-freq drifts, 
        # but I think there are no slow drifts in the IMFs, 
        # because their mean is always 0 anyway or something like that?
        # Just look at the IMF plot.
        
        # set up ICA (variables are defined at the beginning of the script)
        # how many components do I need here? 5?

        # ICA Settings:
        # ICA method is fastICA (= default)
        # use the first 5 components from the PCA (faster computation, no need to look at all 13 channels)
        ica_n_components = 5
        ica_decim = 5 # only use every 5th sample from the data --> speed up computation
        ica_max_iter = "auto" 
        # use a random seed to get the same results every time we run the ICA on the same data
        ica_random_state = 42 
        
        ica = ICA(n_components = ica_n_components, max_iter = ica_max_iter, random_state = ica_random_state)

        # fit ICA
        ica.fit(IMF_Raw)

        # Plot to check what the ICA captured:
        # I don't see anything. Just slighly wavy lines. 
        # Is it supposed to look like this???
        #ica.plot_sources(IMF_Raw, show_scrollbars = True)
    
        # exclude first ICA component 
        # (because I don't see anything in the ICA data
        # and I assume the 1st channel always captures the motor artifacts best?
        # Also I assume it'll be the 1st ICA channel for all participants?
        ica.exclude = [0] 
        
        # apply ICA to IMF_Raw
        ica.apply(IMF_Raw)
    
        
        """ IMPORTANT QUESTION (seriously, I don't have a f***ing clue):
        How do I transform my ICAed 9 IMF-channels back to 1 channel? 
        Or do I just continue using the IMF data?
        Also: Does my ICA make sense???
        """


 #%% 
        """ IDEA: the EMD just decomposes a complex signal into it's components, right? 
        Can I just add the channels to get the complex signal back, 
        just without the component I excluded in the ICA? """

        # F**k it, I'll just try it.
        
        # get each data channel from IMF_Raw, 
        # put everything into nested list:
        imf_channels = list()
        for channel in range(0,len(IMF_Raw._data)):
            curr_channel = IMF_Raw._data[channel]
            imf_channels.append(curr_channel)
        
        # Now get sum of the nested lists (but elementwise)
        # --> like this:
        #              [[1, 2, 1],
        #            +  [3, 3, 1]]
        # Output:       [4, 5, 2]    
        gss_ICAed = [sum(i) for i in zip(*imf_channels)]

        # plot old signal in red
        #plt.plot(gss_values, color = "red", alpha = 0.2)
        # plot ICAed signal in blue
        #plt.plot(gss_ICAed, color = "steelblue", alpha = 1)
        #plt.show()
        # Okayyy the signals look pretty much alike
        # so my idea seems to work!
        

 #%% 
        """ put ICAed signal back into gss_Raw """ 
        
        # create copy of gss_Raw
        gss_ICAed_Raw = gss_Raw.copy()
        
        # load data so I can use indexing 
        gss_ICAed_Raw.load_data()
    
        # change data in data channel
        gss_ICAed_Raw._data[0] = gss_ICAed
 
        # Now continue using the ICAed data!
 
 #%% 
 
        """ 6.2.3 filter GSS data """
        # (variables are defined at the beginning of the script))
                
        gss_ICAed_Raw.filter(l_freq = gss_bandpass_fmin, 
                             h_freq = gss_bandpass_fmax,   
                             phase = gss_phase,
                             fir_window = gss_window_type, 
                             fir_design = gss_fir_design, 
                             n_jobs = gss_n_jobs)
        
        # plot unfiltered ICAed signal in blue
        #plt.plot(gss_ICAed, color = "red", alpha = 0.2)
        # plot filtered ICAed signal in green
        #plt.plot(gss_ICAed, color = "steelblue", alpha = 1)
        #plt.show()
        
        # This looks exactly the same. I wonder if the EMD "smoothes" 
        # the signal a bit by excluding all temporary weird oscillations.
        
  #%%        
        """ 6.2.4 Epoching """    
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
        """ Get Triggers & save as Annotations object """
        
        # get trigger timestamps and trigger descriptions
        trigger_descriptions = gss_ICAed_Raw.annotations.description.tolist()
        trigger_timestamps = gss_ICAed_Raw.annotations.onset.tolist()   
        
        
        # Have a look at the descriptions: 
        # print(trigger_descriptions)
        # The sensor seems to start mid-experiment, this is a bit weird. 
        # So apparently there is no b0 trigger.
    
        """ get block onsets & crop eeg_Raw to seperate blocks """
        if "block0" in trigger_descriptions:
            b0_onset = trigger_timestamps[trigger_descriptions.index("block0")]
        # if there is no block0 in the triggers, use the first 
        # timestamp as the onset of b0    
        else: 
            b0_onset = trigger_timestamps[0]
            
        # if the trigger block1 is in the list of trigger descriptions, save onset.
        if "block1" in trigger_descriptions:
            b1_onset = trigger_timestamps[trigger_descriptions.index("block1")]
            
        # if the trigger block3 is in the list of trigger descriptions, save onset.
        if "block3" in trigger_descriptions:
            b3_onset = trigger_timestamps[trigger_descriptions.index("block3")]
            
        #save data from block 0 (training)
        #b_test_Raw = eeg_Raw.copy().crop(tmin = b0_onset, tmax = b1_onset)
    
        # exclude training block and block 3 (or just exclude training, if there is no block 3)
        if "block3" in trigger_descriptions:
            b_main_Raw = gss_ICAed_Raw.copy().crop(tmin = b1_onset, tmax = b3_onset)
        else:
            b_main_Raw = gss_ICAed_Raw.copy().crop(tmin = b1_onset)

 #%% 
        """ create epochs, only use data from blocks 1 & 2 """
        
        """ create events from annotations """
        # use regular expression to look for strings beginning with t (I only need the trial starts)
        # Also, round strings instead of truncating them so we get unique time values
        trial_events, trial_event_id = mne.events_from_annotations(b_main_Raw, regexp = '(^[t]).*$', use_rounding = True)    
       

        """ create metadata for each epoch containing information on sfb, sfc and feedback condition """
    
        # create epoch names (--> epoch_001, epoch_002, and so on)
        epoch_colnames = [f'epoch_{n:03}' for n in range(1, len(list(trial_event_id.keys()))+1)]
        # create empty df for metadata with the epoch names as column names
        gss_epochs_metadata = pd.DataFrame(columns = epoch_colnames)
        
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
        gss_epochs_metadata.loc[0] = row_feedback
        gss_epochs_metadata.loc[1] = row_sfb
        gss_epochs_metadata.loc[2] = row_sfc
    
        # Aaaand I made a mistake, the metadata must have the same 
        # number of rows as events, but I have the same number of columns as events.
        # Transpose the whole thing because I'm lazy:
        gss_epochs_metadata = gss_epochs_metadata.transpose()  
        
        gss_epochs_metadata.columns = ["feedback", "sfb", "sfc"]

#%%
        """ get epochs, apply baseline correction on the fly """
        # event = trial start, cut from -1.5 to +7           
        trial_epochs = mne.Epochs(b_main_Raw, trial_events, trial_event_id, 
                                  tmin = - 1.5, tmax = 7, baseline = (-1.5, 0), 
                                  preload = True, event_repeated = "drop",
                                  reject_by_annotation = False, metadata = gss_epochs_metadata) # metadata = pass
        
        # plot the epochs (only plot the first 2 or else it gets super messy)
        #trial_epochs.plot(show_scrollbars = True, n_epochs = 2)
        
        # this is how you can select epochs based on the metadata:
        #trial_epochs['sfc == 0.25'].plot()
        # ...or select multiple values:
        #search_values = ['0,25', '.3']
        # more on how to use the metadata: 
        # https://mne.tools/dev/auto_tutorials/epochs/30_epochs_metadata.html

    #%%         
        """ 7. save Raw object & epoched data for each participant in the file 
        I set as the working directory at the beginning of the script """
        trial_epochs.save(fname = "gss_participant" + str(participant) + "_epo.fif", fmt = 'single', overwrite = False)

    
        # save number of processed files
        file += 1

    # END LOOP   
    
    
    #%% 
    
    """ 8. Create "I'm done!"-message: """
    if file == 0:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, something went wrong: I couldn't run the preproc function on any file. Maybe you should have a look at this.\n\n- - - - - - - - - - - - - - - - - - - - - ")
    
    elif file == 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched 1 file for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the file gss_participant1_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    else:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched " + str(file) + " files for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the files gss_participant[number]_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    
# END FUNCTION

