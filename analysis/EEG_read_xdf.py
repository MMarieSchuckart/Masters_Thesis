"""
Function for reading in EEG xdf files

Part of Merle's Master Thesis
Version 1: 12.8.2021

Input: File containing .xdf files with EEG data for each participant
Output: .fif file containing EEG data + triggers for each participant

"""
        
#%% 
#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
        
#%% 
# create function to read in data automatically
def read_in_EEG(working_directory):


    """ 1. load packages """
    
    # pyxdf for reading in xdf data:
    import pyxdf
    
    # Python MNE for eeg data analysis / processing:
    # NME should be active by default and should already have the 
    # necessary dependencies included (numpy, scipy, and matplotlib).    
    import mne
    
    # glob for getting all files in a directory:
    import glob
    
    # numpy for numpy arrays
    import numpy as np
    
    # os for setting working directory:
    import os
    
    # chain for turning nested list into 1D list:
    from itertools import chain
    
    # for finding EOG events / plotting EOG epochs    
    from mne.preprocessing import find_eog_events
    from mne.preprocessing import create_eog_epochs
    
    # write small function to find out if a value is a float
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
        
#%% 

    """ 2. set working directory """
    os.chdir(working_directory)
    
    """ 3. get list of all xdf files in my directory """
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list = glob.glob(working_directory + "*.xdf")

    """ 4. Create empty lists to keep track of plots (before and after filtering)"""
    #eeg_figs_before_filtering = []
    #eeg_figs_after_filtering = []
    
    """ 5. keep track of participants """
    participant = 0

#%%
    """ 6. loop xdf file names in file_list (aka loop participants): """
    for file_name in file_list:
        
        """ save participant number"""
        participant += 1
        
        """ 6.1 read in XDF data """
        streams, header = pyxdf.load_xdf(file_name)
    
        # Raise a hue & cry if data doesn't have 4 streams!
        assert len(streams) == 4  # 1 EEG markers (?), 1 EEG, 1 stim channel, 1 gss channel
    
#%%   
        """ 6.2 Build NME Raw object from scratch """
        # stream 0: general info? actiCHampMarkers (whatever that is)
        # stream 1: Actichamp - EEG data
        # stream 2: PsychoPyMarkers - Experiment markers
        # stream 3: Arduino - Grip strength sensor data    
    
        # each stream contains timestamps (measured in seconds)
        
  #%%       
        
        """ Find out which stream contains which kind of data """
        # The streams might be in a different order in each file, so find out 
        # where the GSS data are and which stream contains the triggers
            
        for stream_idx in range(0, 4):
        
            # if there are 22 data arrays, it's not the triggers 
            # and probably not the GSS data, so ignore that stream
            if len(streams[stream_idx]["time_series"][0]) == 128:
                eeg_idx = stream_idx
                print(str(stream_idx) + " = EEG")
            
            # If it doesn't have 128 channels, it can only be the triggers, 
            # the GSS data or the empty other channel.
            # Check if the first element in the stream contains text and is not empty:
            elif isfloat(streams[stream_idx]["time_series"][0][0]) == False and streams[stream_idx]["time_series"][0][0] != '':
                trig_idx = stream_idx
                print(str(stream_idx) + " = Triggers")
        
            # If the stream doesn't have 128 channels and the first element 
            # doesn't contain text, it has to be either the GSS data or the other stream.
            # If the first element is empty, it has to be the empty stream.
            elif streams[stream_idx]["time_series"][0][0] == '':
                print(str(stream_idx) + " = empty")
        
            # the only stream left now is the GSS stream:
            else:
                gss_idx = stream_idx
                print(str(stream_idx) + " = GSS")        
    
    
        print("---------------------------------------")
        
                  
  #%% 

        
        """ 6.2.1 Create info for Raw Object for EEG data"""
        # Sampling rate: 500 Hz
        sampling_freq = float(streams[eeg_idx]["info"]["nominal_srate"][0]) # in Hertz
        
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
      
                
  #%%     
    
        """ 6.2.2 Get EEG data for Raw object""" 
        # structure should be: 
            # rows: channels
            # columns: sample points
        
        # get EEG data from stream 1:
        # 128 arrays (1 for each electrode), 186013 sampling points
        data_eeg = np.array(streams[eeg_idx]["time_series"].T) 
    
        # transform all values in eeg_data from Microvolt to Volt 
        # as NME expects EEG data to be measured in Volt (why tho)
        data_eeg[:] *= 1e-6
        
            
        """ 6.2.3 Create Raw object for EEG data""" 
        # combine info & eeg data
        eeg_Raw = mne.io.RawArray(data_eeg, info_eeg)
    
    
        """ 6.2.4 change the 3 EEG channels in the face to EOG channel type """
        # check channel types before converting them to eog:
        #eeg_Raw.get_channel_types("EEG_020")
        # convert channel type of the 3 electrodes in question to eog
        eeg_Raw.set_channel_types(mapping = {"EEG_031" : "eog", "EEG_032" : "eog", "EEG_020" : "eog"})
        # This raises an error message, I don't know why but it works so I don't care rn. 
        # Check channel type again:
        #eeg_Raw.get_channel_types("EEG_020")
        
        
      #%% 
    
        """ 6.3 Add Annotations """
           
        # If you look at the first timestamp of stream 1, 2 & 3, you can see that 
        # they don't match. The EEG for example started recording 
        # way earlier than the Arduino:
        #streams[eeg_idx]["time_stamps"][0] 
        #streams[gss_idx]["time_stamps"][0] 
       
        # This means I need to include information on when the 
        # Triggers started or else MNE assumes the streams all 
        # started at the same time, which is not the case.
       
        # If you look at when the streams were created, 
        # you see that they are not 0, which means they're 
        # probably all relative to some shared event 
        # (like turning on the computer? Idk.)
        #streams[gss_idx]['info']['created_at']
        #streams[trig_idx]['info']['created_at']
        #streams[eeg_idx]['info']['created_at'] 
       
        # Solution: Set onset of EEG stream to None (that's the default anyway)
        # and subtract onset of EEG stream from Trigger data. 
        # This way the timestamps are relative to the EEG onset.
           
        # get difference between EEG onset and onset of Triggers
        eeg_onset = streams[eeg_idx]["time_stamps"][0] 
        trigger_timestamps = streams[trig_idx]["time_stamps"] - eeg_onset

#%%       
        """ 6.3.1 Get Triggers & save as Annotations object """
    
        # get names of triggers (it's a nested list in the xdf file)    
        trigger_descriptions = streams[trig_idx]["time_series"]
        # turn nested list into "normal" one dimensional list
        trigger_descriptions = list(chain.from_iterable(trigger_descriptions)) 
        
        """ 6.3.2 change trigger descriptions so the trial start descriptions 
        contain info on the block & feedback condition """
        
        # start with block 0:
        block_nr = "block0"
    
        # loop trigger descriptions
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
            
        """ 6.3.3 save trigger descriptions & their onsets as annotations for our Raw object"""
        triggers_annot = mne.Annotations(onset = trigger_timestamps, duration = .001, description = trigger_descriptions)
        
        
#%%       
        """ 6.4 Automatically detect blinks, name them "bad blink" & save as Annotations object """
        
        # find_eog_events filters the data before detecting oeg events:
        # FIR two-pass forward and reverse, zero-phase, 
        # non-causal bandpass filter from 1 - 10 Hz with von Hann window
            
        """ 6.4.1 find all eog events """
        eog_events = find_eog_events(eeg_Raw)
        
        # set onset as 250 ms before the peak of the blink
        onsets = eog_events[:, 0] / eeg_Raw.info['sfreq'] - 0.25
        
        # set duration of blink epoch (500 ms around peak)
        durations = [0.5] * len(eog_events)
        
        """ 6.4.2 create description for blinks """
        # (all annotations that start with "bad" 
        # can be automatically rejected later)
        blink_descriptions = ['bad blink'] * len(eog_events)
        
        """" 6.4.3 create annotation object for the detected blinks """
        blink_annot = mne.Annotations(onsets, durations, blink_descriptions, 
                                      orig_time = eeg_Raw.info['meas_date'])
        
        """ 6.5 Set annotations """
        # there's a bug in MNE, you have to set a filename 
        # to be able to set annotations:
        eeg_Raw._filenames = ['a_random_file_name'] 
    
        # set triggers & blinks as annotations in the EEG Raw object
        eeg_Raw.set_annotations(triggers_annot + blink_annot)
    
#%%     
        """ 6.7 plot raw data """
        # Plot raw data with EOG events
        # I filtered the data a bit, but this is only for the 
        # representation in the plot, it doesn't affect the data!
        #raw_plot = eeg_Raw.plot(n_channels = 6, highpass = 5, duration = 8, lowpass = 35, show_scrollbars = True)

        # Plot EOG events as butterfly plot        
        # find EOG epochs,create new eog object
        #eog_evoked = create_eog_epochs(eeg_Raw).average()
        # baseline correct data
        #eog_evoked.apply_baseline(baseline = (None, -0.2))
        # plot EOG events as butterfly plot (all events in one plot)
        #eog_evoked.plot_joint()

#%% 
        """ 6.8 save Raw object containing EEG data in the file 
        we already stored the xdf files in """
        eeg_Raw.save(fname = "eeg_participant" + str(participant) + "_raw.fif", fmt = 'single', overwrite = False)
    

    ### END LOOP PARTICIPANTS 
    
  #%% 
    
    """ 7. Create "I'm done!"-message: """
    if participant == 0:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, something went wrong: I couldn't run the read-in function on any file. Maybe you should have a look at this.\n\n- - - - - - - - - - - - - - - - - - - - - ")
    
    elif participant == 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in 1 xdf-file for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the fif-file eeg_participant1_raw.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    else:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in " + str(participant) + " xdf-files for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the fif-files eeg_participant[number]_raw.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
### END FUNCTION
        
#%% 

""" 8. Bonus: test if function works """
# set working directory
# working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# run function
# read_in_EEG(working_directory)

