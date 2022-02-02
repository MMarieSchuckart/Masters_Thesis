"""
Function for preprocessing GSS data (from fif files)

Part of Merle's Master Thesis
Version 1: 13.01.2022

Input: File containing .fif files with GSS data + triggers for each participant
Output: .fif file containing MNE epochs object with filtered & epoched GSS data
"""

#%% 
working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
gss_bandpass_fmin = 4 
gss_bandpass_fmax = 12 
gss_phase = "zero" 
gss_window_type = 'hamming' 
gss_fir_design = 'firwin' 
gss_n_jobs = 1 


# create function to filter + epoch data automatically
def GSS_filter_epoching(working_directory, 
                        gss_bandpass_fmin = 4, 
                        gss_bandpass_fmax = 12, 
                        gss_phase = "zero", 
                        gss_window_type = 'hamming', 
                        gss_fir_design = 'firwin', 
                        gss_n_jobs = 1):
      
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
    
    # Please make sure you pip installed the hampel package:
    #pip install hampel
    from hampel import hampel
    
    #%% 
    """ 2. set working directory """
    os.chdir(working_directory)
    
    """ 3. get list of all gss .fif files in my directory """
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list = glob.glob(working_directory + "gss_participant" + "*raw.fif")
    
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
    
        participant = file_name[-11:-8]
        
        # if participant number has < 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]        
        
        """ 6.1 read in fif file """
        gss_Raw = mne.io.read_raw_fif(file_name)

    
 #%%    
        """ 6.2 Preprocessing """
        # (variables used here are set in the main script)
        
        # If you have a look at the gss data, you'll see that the Arduino 
        # didn't record the whole time but only during the trials. 
                
        # plot raw data:
        # get data & turn gss_data into list instead of np.array
        #gss_values = gss_Raw[:][0][0]
        #gss_values = gss_values.tolist()
            
        # get time stamps
        #time_vect = list(gss_Raw[:][1])
            
        #plt.plot(time_vect, gss_values)
        #plt.show()

        # Solution: Epoching before cleaning the data.
        
#%%        
        """ Epoching """
    
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
        
        
        """ Get Triggers & save as Annotations object """
        
        # get trigger timestamps and trigger descriptions
        trigger_descriptions = gss_Raw.annotations.description.tolist()
        trigger_timestamps = gss_Raw.annotations.onset.tolist()   
        
        # Have a look at the descriptions: 
        # print(trigger_descriptions)
    
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
        else:
            # this shouldn't happen, but in the test dataset, 
            # I only have a few trials from block 0, 
            # so I'll set the onset as onset of the trigger timestamps 
            # and offset as offset of the trigger timestamps. 
            b1_onset = trigger_timestamps[0]
            
        # if the trigger block3 is in the list of trigger descriptions, save onset.
        if "block3" in trigger_descriptions:
            b3_onset = trigger_timestamps[trigger_descriptions.index("block3")]
        else: 
            # if there's no block 3 in the data, set offset 
            # as end of the dataset (aka last timestamp)
            b3_onset = trigger_timestamps[len(trigger_timestamps)-1]
            
        #save data from block 0 (training)
        #b_test_Raw = eeg_Raw.copy().crop(tmin = b0_onset, tmax = b1_onset)
    
        # exclude training block and block 3 (or just exclude training, if there is no block 3)
        b_main_Raw = gss_Raw.copy().crop(tmin = b1_onset, tmax = b3_onset)


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
        """ get epochs """
        # event = trial start, cut from 0 to 6 (in seconds, measured from trial start)         
        gss_epochs = mne.Epochs(b_main_Raw, 
                                trial_events, 
                                trial_event_id, 
                                tmin = 0, 
                                tmax = 6, 
                                baseline = None,
                                preload = True, 
                                event_repeated = "drop",
                                reject_by_annotation = False, 
                                metadata = gss_epochs_metadata) # metadata = pass
        
        # plot the epochs (only plot the first 3 or else it gets super messy)
        # Have a look at the first epoch: There's a break where the Arduino stopped recording.
        # The other ones look okay tho.
        #trial_epochs.plot(show_scrollbars = True, n_epochs = 3, scalings = 'auto')
        
        
  #%%        
        """ Hampel Filter """
              
        # There's another problem besides the Arduino only recording during trials: 
        # In the trials, there are single sample points missing.
        
        # Solution: Hampel Filter
        #https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d


        """ check if the filter works: """
            
        # # get a random epoch, insert a 0 at a random index to create a "break"
        # simu_epoch = pd.Series(gss_epochs.get_data(item = 5)[0][0])
        # simu_epoch[34] = 0
        
        # # set sampling rate
        # sample_rate = 80
        
        # # create time vector
        # times = np.linspace(0, len(simu_epoch)/sample_rate, sample_rate, endpoint = False) 
        
        # # use Hampel filter on data
        # simu_epoch_after = hampel(simu_epoch, window_size = 3, n = 2, imputation = True)
        
        # # plot signal before and after having used the Hampel filter
        # plt.figure(figsize=(8, 4))   
        # plt.plot(times[:sample_rate], simu_epoch[:sample_rate], color='red', alpha = 1, linestyle = 'dotted')
        # plt.plot(times[:sample_rate], simu_epoch_after[:sample_rate], color='teal', alpha = 0.5)
        # plt.xlabel('Zeit (in s)')
        # plt.ylabel('Amplitude (in V)')
        # plt.legend(['Signal vor dem Filtern', 'Signal nach dem Filtern'], loc = 'lower right')
        # plt.show()
        # plt.savefig('/Users/merle/Desktop/Masterarbeit/Plots/Hampel_example.pdf')
        # # hooray! :-D


        """ loop epochs & apply Hampel filter """
        
        # get data from Raw object containing all epochs
        epochs_data = gss_epochs.get_data()
        epochs_data_hamfilt = list()

        for epoch_idx in range(0, len(epochs_data)):
            
            # get epoch data
            epoch = epochs_data[epoch_idx,:,:][0]
        
            # convert to pandas.Series object
            epoch = pd.Series(epoch)
        
            # apply Hampel filter (detect outliers & )
            epoch_hampeled = hampel(epoch, window_size = 3, n = 2, imputation = True)

            # put data back in df
            #epochs_data[epoch_idx,:,:] = epoch_hampeled

            epochs_data_hamfilt.append(list(epoch_hampeled))


        # Plot the epochs:
        # # set sampling rate
        # sample_rate = 80

        # for epoch_idx in range(0, len(epochs_data)):
            
        #     # get unfiltered and hampel-filtered data 
        #     unfiltered = epochs_data[epoch_idx,:,:][0]
        #     filtered = epochs_data_hamfilt[epoch_idx]
            
        #     # create time vector
        #     times = np.linspace(0, len(filtered)/sample_rate, sample_rate, endpoint = False) 
        
        
        #     # plot signal before and after having used the Hampel filter
        #     plt.figure(figsize=(8, 4))   
        #     plt.plot(times[:sample_rate], unfiltered[:sample_rate], color='indianred', alpha = 1)
        #     plt.plot(times[:sample_rate], filtered[:sample_rate], color='teal', alpha = 1, linestyle = 'dotted')
        #     plt.xlabel('Zeit (in s)')
        #     plt.ylabel('Amplitude (in V)')
        #     plt.title('epoch ' + str(epoch_idx))
        #     plt.legend(['Signal vor dem Filtern', 'Signal nach dem Filtern'], loc = 'lower right')
        #     plt.show()
        

 #%%      
        """ 6.2.1 EMD (empirical mode decomposition) """
        # (to get rid of motor / heart artifacts if there are any in the signal)
        # Which different oscillations are there in the signal 
        # that are non-linear or non-stationary? 
        #--> https://emd.readthedocs.io/en/stable/
    
        # loop hampel filtered epochs
        for epoch_idx in range(0, len(epochs_data_hamfilt)):
                        
            # get epoch data:
            epoch = np.array(epochs_data_hamfilt[epoch_idx])

            # estimate IMFs (=intrinsic mode functions)
            imf = emd.sift.sift(epoch)
            
            # exclude last component because it's only the residual trend
            imf_excl = imf[:, 0:len(imf[0,:])-1]           

            # the first IMF contains high freq components (in EEG mostly artifacts)
            # but I assume in the gss data, those are the IMFs with the 
            # highest amount of signals so keep those before I exclude anything important!
            
            # project back to 1 signal (without the component we excluded)
            
            # save first IMF 
            preprocd_gss_signal = imf_excl[:, 0] 
            # loop & add the rest of the IMFs
            for imf_nr in range(1, len(imf_excl[0])):
                preprocd_gss_signal = preprocd_gss_signal + imf_excl[:, imf_nr]           
                        
            # plot it!
            # plot old signal (aka signal before exclusion) in red + dotted line
            #plt.plot(epoch, color = "red", alpha = 1, linestyle="dotted")
            # plot new signal (without last IMF) in blue
            #plt.plot( preprocd_gss_signal, color = "steelblue", alpha = 1)
            #plt.show()

            # put new preprocessed signal back into Raw object so we 
            # don't loose the information on the conditions in the trial            
                                    
            gss_epochs._data[epoch_idx,0,:] = preprocd_gss_signal

 
 #%% 
 
        """ 6.2.3 filter GSS data """
        # (variables are defined at the beginning of the script))
                
        #gss_bandpass_fmin = 4 
        #gss_bandpass_fmax = 12 
        #gss_phase = "zero" 
        #gss_window_type = 'hamming' 
        #gss_fir_design = 'firwin' 
        #gss_n_jobs = 1 
        
        gss_epochs_filtered = gss_epochs.copy()
        gss_epochs_filtered.filter(l_freq = gss_bandpass_fmin, 
                                   h_freq = gss_bandpass_fmax,   
                                   phase = gss_phase,
                                   fir_window = gss_window_type, 
                                   fir_design = gss_fir_design, 
                                   n_jobs = gss_n_jobs)
        
        # plot unfiltered signal in blue
        #gss_epochs.plot(picks = "all", scalings = "auto", n_epochs = 1, show_scrollbars=True)
        
        # loop epochs, plot data before and after filtering
        #for epoch_nr in range(0, len(gss_epochs)):   
        #    # plot unfiltered signal in blue
        #    before = gss_epochs._data[epoch_nr,0,:]
        #    after = gss_epochs_filtered._data[epoch_nr,0,:]
        #    
        #    plt.plot(before, color = "red", alpha = 0.5)
        #    plt.plot(after, color = "blue", alpha = 0.5)
        #    plt.show()


    #%%         
        """ 7. save Raw object & epoched data for each participant in the file 
        I set as the working directory at the beginning of the script """
        gss_epochs_filtered.save(fname = "gss_participant" + str(participant) + "_epo.fif", fmt = 'single', overwrite = False)

    
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

