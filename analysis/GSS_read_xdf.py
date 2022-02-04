"""
Function for reading in GSS data from xdf files

Part of Merle's Master Thesis
Version 1: 4.2.2021

Input: File containing .xdf files with GSS data for each participant
Output: .csv file containing GSS data + triggers for each participant 

"""
#%%
#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# create function to read in data automatically
def read_in_GSS(working_directory):
    

    """ 1. load packages """
        
    # pyxdf for reading in xdf data:
    import pyxdf
    
    # glob for getting all files in a directory
    import glob
    
    # os for setting working directory
    import os
    
    # For numpy arrays
    import numpy as np
    
    # for dataframes as in R
    import pandas as pd
    
    # for turning nested list into 1D list
    from itertools import chain
    
    # for plotting
    from matplotlib import pyplot as plt
    
    # write small function to find out if a value is a float
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    
#%%
 
    """ 2. read in data """
    
    # set working directory (--> variable must be given if function is called!)
    os.chdir(working_directory)
    
    # get list of all xdf files in my directory 
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list = glob.glob(working_directory + "*.xdf")
        
    """ 3. keep track of participants """
    participant = 0
    
  #%%      
    """ 4. loop xdf file names in file_list (aka loop participants): """
    for file_name in file_list:
        
        # save number of current participant
        participant +=  1
        
        """ 4.1 read in XDF data """
        streams, header = pyxdf.load_xdf(file_name)
    
        # Raise a hue & cry if data doesn't have 4 streams!
        assert len(streams) == 4  # 1 EEG markers (?), 1 EEG, 1 stim trigger channel, 1 gss channel
    
        
        """ 4.2 Build NME data object from scratch """
        # stream 0: general info? actiCHampMarkers (whatever that is)
        # stream 1: Actichamp - EEG data
        # stream 2: PsychoPyMarkers - Experiment triggers
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
                #eeg_idx = stream_idx
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

        # The Arduino stops recording between the trials. This is a problem when working with MNE because
        # I can't define timestamps for each sample when creating a Raw object, I 
        # can only define a sampling rate and MNE reconstructs evenly spaced time stamps for the values. 
        # Obviously if the timing of the measured values isn't always the same because there are breaks in the
        # recordings, this doesn't work out, so we can't create a Raw object for the data. 
        # We have to start with Epochs.
    
        
        # For creating Epochs, I need epoched data (duh.). 
        # So what do I do about my continuous signal with weird breaks in it???
        
        """ Find weird breaks in signal, create data for Epochs object """
        
        # compute difference between timestamps
        # IDEA: If there's a large difference between time stamps, the recording stopped in that interval.
        timestamps_first_to_penultimate = streams[gss_idx]["time_stamps"][:-1]
        timestamps_second_to_last = streams[gss_idx]["time_stamps"][1:]
        # get difference between timestamps
        diff_timestamps = timestamps_second_to_last - timestamps_first_to_penultimate
    
        # length of diff_timestamps is length of all timestamps - 1
        # so if I find a large difference somewhere, the indices of the timestamps with the break between them is 
        # for the 1st timestamp: idx(diff) and for the 2nd one idx(diff)+1 
        
        # find breaks,
        # get data from last break to current break and save in nested array
        # set onset of first epoch as onset of the recording
        epoch_counter = 0
        idx_epoch_start = 0
        # create placeholders for nested lists for epoch data
        epoch_timestamps_all = []
        epoch_data_all = []
        # loop difference array
        for idx in range(0, len(diff_timestamps)):
            
            # find big differences > 0.5s!
            if diff_timestamps[idx] > 0.5:
                print("found break in signal, creating next epoch!")
    
                # get data for epoch
                epoch_timestamps = streams[gss_idx]["time_stamps"][idx_epoch_start:idx]
                epoch_data = streams[gss_idx]["time_series"][idx_epoch_start:idx]
                # turn weird nested list into 1D list
                epoch_data = list(chain.from_iterable(epoch_data)) 
                
                # transform all values in gss_data from Microvolt to Volt 
                # as NME expects time series data of type "eeg" to be measured in Volt (why tho)
                # and I assume that the GSS measures in mV?
                #for i in range(0, len( epoch_data)-1):
                #    epoch_data[i] *= 1e-6
    
                # put into nested array
                epoch_timestamps_all.append(epoch_timestamps)
                epoch_data_all.append(epoch_data)
                
                # find next epoch! 
                idx_epoch_start = idx + 1
                epoch_counter += 1
        
#%% 
        """ Get Triggers """
    
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
            
        # get names of triggers (it's a nested list in the xdf file)    
        trigger_descriptions = streams[trig_idx]["time_series"]
        # turn nested list into "normal" one dimensional list
        trigger_descriptions = list(chain.from_iterable(trigger_descriptions)) 
        
        """ 4.3.1.1 change trigger descriptions so the trial start descriptions 
        contain info on the block & feedback condition """
        
        # start with block 0:
        block_nr = "block0"
    
        for i in range(len(trigger_descriptions)):
            # get trigger name
            trigger_descr = trigger_descriptions[i]
            
            # check if a new block begins:
           
            # If the current trigger starts with "block", but doesn't correspond to the current block,
            # change name of current block.
            if trigger_descr[0:5] == "block" and trigger_descr!= block_nr:
                # save new block name
                block_nr = trigger_descr   
            
            # else if the trigger is not a new block name &
            # if trigger name starts with ep (like epoch)...
            elif trigger_descr[0:2] == "ep":
                #... save feedback condition of the current epoch
                curr_epoch = trigger_descr[6:-6]
            
            # else if the trigger is not a new block name &
            # if the name starts with t (as in trial_start)...    
            elif trigger_descr[0] == "t":
                
                #... concatenate trial name and epoch name & block number, divided by an underscore
                trigger_descriptions[i] = trigger_descr + "_" + curr_epoch + "_" + block_nr
            
     
#%%
        """ cut epochs to segments of 6s after trial start """
        
        # get all timestamps of trial starts
        # find indices
        trial_start_idx = [i for i, s in enumerate(trigger_descriptions) if 'trial_start' in s]
        # get timestamps
        trigger_timestamps = streams[trig_idx]["time_stamps"]
        # get timestamps from the same indices as the "trial_start*" descriptions
        trial_timestamps = trigger_timestamps[trial_start_idx]
        # get list of all trial descriptions starting with "trial_start"
        trial_descriptions = np.array(trigger_descriptions)[trial_start_idx]
        
        
        # Now cut epochs:
        
        # we want 6 seconds aka 6 * 80 sample points, starting at trial start
        tmax = 6 * 80 
        
        # count how many epochs are excluded or not found
        epochs_excluded = 0
        epochs_not_found = 0
        
        # empty list for collecting info and data for the epochs
        epoch_data = []
        
        # loop timestamps
        for trigger_idx in range(0, len(trial_timestamps)):

            # get current trigger we're trying to find a matching epoch for!
            curr_trig_timestamp = trial_timestamps[trigger_idx]
            
            # initialize check if matching epoch for current trigger was found
            found_epoch = False
            
            # loop epochs            
            for epoch_idx in range(0, len(epoch_data_all)):
                
                # get first timestamp from current epoch
                curr_epoch_onset = epoch_timestamps_all[epoch_idx][0]
                
                # if there are less than 200 ms between the trial start trigger and 
                # the epoch onset...
                if curr_epoch_onset - curr_trig_timestamp < 0.2 and curr_epoch_onset - curr_trig_timestamp > - 0.2:
                    print("Found epoch for trigger " + str(trigger_idx) + "! Cutting Epoch Nr " + str(epoch_idx) + " now.")
                    found_epoch = True
                    
                    # cut epoch so it has a duration of about 6s from trial onset 
                    # (or rather from the first sample)
                    
                    # index of epoch onset is 0
                    # get index of intended epoch offset (index + tmax, 
                    # in our case 6s aka 6 * 80 samples)
                    
                    # if there's more than 1 sec (aka 1 * 80 samples) of 
                    # the epoch missing, exclude epoch:
                    if len(epoch_data_all[epoch_idx]) <= tmax - 80:
                        # ignore epoch and count as excluded:
                        epochs_excluded += 1
                        
                    # if the epoch is long enough to use it...
                    else: 
                    
                        # get sfb condition from trigger description for current epoch
                        sfb = trial_descriptions[trigger_idx][16:20]
                        # shorten sfb if last digit is _
                        if sfb[3] == "_":
                            sfb = sfb[0:3]
                        
                        # get sfc condition from trigger description for current epoch
                        sfc = trial_descriptions[trigger_idx][-14:-10]
                        # shorten sfc if first digit is _
                        if sfc[0] == "_":
                            sfc = sfc[1:]
                            
                        # typecast sfb and sfc to floats
                        sfb = float(sfb)
                        sfc = float(sfc)
                        
                        # get feedback condition from trigger description for current epoch
                        feedback = trial_descriptions[trigger_idx][-9:-7]
                        
                        # get block nr from trigger description for current epoch
                        block = trial_descriptions[trigger_idx][-6:]

                        # if the epoch is max. as long as intended, 
                        # you don't have to cut it. 
                        if len(epoch_data_all[epoch_idx]) <= tmax:
                        
                            # get data for current epoch
                            curr_epoch_data = epoch_data_all[epoch_idx]
                            # get timestamps for current epoch
                            curr_epoch_timestamps = epoch_data_all[epoch_idx]
                            
                        # if it's longer than tmax, you have to cut it!
                        else: 
                           
                            # get data for current epoch, but cut off some samples 
                            # at the end so it has the right length:
                            curr_epoch_data = epoch_data_all[epoch_idx][ :tmax]
                            # get timestamps for current epoch
                            curr_epoch_timestamps = epoch_data_all[epoch_idx][ :tmax]

                        # save information on conditions as well as timestamps & data in list
                        epoch_data.append([block, sfb, sfc, feedback, curr_epoch_timestamps, curr_epoch_data])

                    # don't look at the other epochs for this 
                    # trigger as we already found a match!
                    break
                
            if found_epoch == False:
                print("\nEpoch for trigger " + str(trigger_idx) + " could not be found! Sorry queen!")
                # add 1 to counter for epochs that were not found:
                epochs_not_found += 1
                
                if len(epoch_data_all) < len(trial_timestamps):
                    print("There are less epochs than trial onsets.\n")


        # turn list with epoch data into dataframe:
        gss_epochs = pd.DataFrame(epoch_data, columns=["block", "sfb", "sfc", "feedback", "time_stamps", "time_series"])

        # runtime for this is a bit longer than necessary because we're always 
        # looping all epochs (also the ones we already found) but f*** it.
    

#%% 
        """ Exclude training block and block 3 """
        
        # exclude block 0 (= training):
        gss_epochs.drop(gss_epochs[gss_epochs.block == "block0"].index, inplace=True)
        # exclude block 3:
        gss_epochs.drop(gss_epochs[gss_epochs.block == "block3"].index, inplace=True)


#%% 
        """ 4.5 save df with epoched data for each participant in the working directory """ 
        gss_epochs.to_csv(path_or_buf = working_directory + "gss_participant" + str(participant) + "_raw_epo.csv")
            
        
    ### END LOOP PARTICIPANTS 
    
#%%    
    # Create "I'm done!"-message:
    if participant == 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in " + str(participant) + " xdf-file for you.\nHave a look at the file you set as a \nworking directory in the function call!\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    elif participant > 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in " + str(participant) + " xdf-files for you.\nHave a look at the file you set as a \nworking directory in the function call!\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
### END FUNCTION    

#%%

""" 5. Bonus: test if function works """
# set working directory
# working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# run function
# read_in_GSS(working_directory)

