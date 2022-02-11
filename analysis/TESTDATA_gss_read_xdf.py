
""" Reading in GSS test data for testing the preproc & stats scripts

Part of Merle's Master Thesis
Version 1: 11.02.2022


This script is for reading in some test datasets from Julius  
where he measured GSS data & used the eyetracker (pilot study? Idk).

These test datasets have a slightly different shape 
than the other datasets so you can use this script to read them in
and then you can proceed as you normally would and use 
the functions for preprocessing & analyzing GSS data.

"""

   
""" 1. Settings """

""" load packages """
    
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

# Python MNE for time series data analysis / processing:
# NME should be active by default and should already have the 
# necessary dependencies included    
import mne

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
working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/trmr_fdbck_ms/"

# set working directory (--> variable must be given if function is called!)
os.chdir(working_directory)

# get list of all xdf files in my directory 
# (the asterix in the path means the name of the 
# file can be anything as long as it has an .xdf ending)
file_list = glob.glob(working_directory + "*.xdf")
    
# keep track of participants
participant = 0

#%%      

""" 2.1 loop xdf file names in file_list (aka loop participants): """
for file_name in file_list:
    
    # save number of current participant
    participant +=  1
    
    """ 2.2 read in XDF data """
    streams, header = pyxdf.load_xdf(file_name)

    """ 2.3 Build NME data object from scratch """
        
  #%%     
    """ 2.4 Find out which stream contains which kind of data """
    # The streams might be in a different order in each file, so find out 
    # where the GSS data are and which stream contains the triggers
    
    # loop streams:
    for stream_idx in range(0, 3):
        
        # if there are 22 data arrays, it's the eye tracker stream 
        # --> ignore that stream
        if len(streams[stream_idx]["time_series"][0]) == 22:
            print(str(stream_idx) + " = Eye Tracker")
            
        # If it doesn't have 22 channels, it can only be 
        # the triggers or the GSS data.
        # Check if the first element in the stream contains 
        # text (aka can't be typecasted to float)
        elif isfloat(streams[stream_idx]["time_series"][0][0]) == False:
            trig_idx = stream_idx
            print(str(stream_idx) + " = Triggers")
        
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
    
    """ 2.5 Find weird breaks in signal, create data for Epochs object """
    
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

            """ found epoch, get data """
            epoch_timestamps = streams[gss_idx]["time_stamps"][idx_epoch_start:idx]
            epoch_data = streams[gss_idx]["time_series"][idx_epoch_start:idx]
            # turn weird nested list into 1D list
            epoch_data = list(chain.from_iterable(epoch_data)) 
            
            # put into nested arrays
            epoch_data_all.append(epoch_data)
            # hint: you need to turn the timestamps array into a 1D list or else 
            # it's not saved correctly (idk why this is only the case with the timestamps)
            epoch_timestamps_all.append(epoch_timestamps.tolist())
            
            # find next epoch! 
            idx_epoch_start = idx + 1
            epoch_counter += 1
    
     
                    
#%%
    """ 2.6 Exclude epochs that are too short """
    # shouldn't happen but you never know
    
    # Let's say we exclude all epochs where there's more 
    # than 1 sec aka 80 samples missing.
    
    # count how many epochs had to be excluded
    epochs_excluded = 0
    # keep track of which epochs to exclude
    excl_epoch = []
    
    # get length of each epoch array
    epoch_sizes = []
    for epoch_idx in range(0, len(epoch_data_all)):
        epoch_sizes.append(len(epoch_data_all[epoch_idx]))

    # get length of longest epoch (aka max. number of samples)
    max_size = max(epoch_sizes)
    #--> keep in mind we did this, we'll use max_size again later!
    
    # loop epochs 
    for epoch_idx in range(0, len(epoch_data_all)-1):

        # if current epoch is too short...
        if len(epoch_data_all[epoch_idx]) < max_size - 80:
                    
            # save index of epoch you want to exclude:
            excl_epoch.append(epoch_idx)
            
            # add 1 excluded epoch to counter
            epochs_excluded += 1
    print(str(epochs_excluded) +" epoch(s) had to be excluded because\nthere was more than 1 sec missing.")
    
    
    # Exclude epochs now:

    # small hack: indices are sorted in ascending order, 
    # which means we'd mess the indices up if we remove rows. 
    # And thats why I reverse the order. BAM!
    # (Not that impressive but that's the hack actually.)
    for idx in sorted(excl_epoch, reverse = True):       
        del epoch_data_all[idx]
        del epoch_timestamps_all[idx]
        del epoch_sizes[idx]
    
    
#%% 
    """ 2.7 Small overview so the next parts make sense: """ 

    # What I'll do now is:

    # 1. get trigger descriptions, change descriptions so 
    #    they contain info on sfc, sfb, feedback & block
        
    # 2. get only descriptions + timestamps of trial start triggers
    
    # 3. check if there's a matching trigger for each epoch
    
    # 4. exclude epochs I can't find a trigger for
    
    # 4. for all trigger-epoch-matches, save info on sfc, sfb, feedback & block in df
            
    
    # Okay so there are going to be a lot of loops now. :-)
    
#%% 
    """ 2.8 Get Triggers """
    
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
    
    """ 2.9 Change trigger descriptions so the trial start descriptions 
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
            #... concatenate trial name and epoch name & block number, 
            # divided by an underscore
            trigger_descriptions[i] = trigger_descr + "_" + curr_epoch + "_" + block_nr
            
            
#%%
    """ 2.10 Find matching epoch for each trigger """
    
    # get all timestamps of trial starts
    # find indices
    trial_start_idx = [i for i, s in enumerate(trigger_descriptions) if 'trial_start' in s]
    # get timestamps
    trigger_timestamps = streams[trig_idx]["time_stamps"]
    # get timestamps from the same indices as the "trial_start*" descriptions
    trial_timestamps = trigger_timestamps[trial_start_idx]
    # get list of all trial descriptions starting with "trial_start"
    trial_descriptions = np.array(trigger_descriptions)[trial_start_idx]
    
    
    # empty list for collecting information on sfb, sfc and 
    # feedback for the epochs
    gss_epochs_conditions = []
    
    # we haven't found a trigger for any of the epochs yet, so put all of their indices in 
    # exclude_these_epochs for a start 
    # --> IDEA: Delete epoch index from this list if trigger was found for an epoch
    #           This way, there will only be epochs left we couldn't find a trigger for.
    exclude_these_epochs = list(range(0, len(epoch_data_all)))
    
    """ 2.10.1 loop triggers """
    for trigger_idx in range(0, len(trial_timestamps)):

        # get current trigger we're trying to find a matching epoch for!
        curr_trig_timestamp = trial_timestamps[trigger_idx]

        # we haven't found an epoch for this trigger yet, so...            
        found_epoch = False

        """ 2.10.2 loop epochs  """           
        for epoch_idx in range(0, len(epoch_data_all)):
            
            # get first timestamp from current epoch
            curr_epoch_onset = epoch_timestamps_all[epoch_idx][0]
            
            """ 2.10.3 Save information on epoch if epoch matches trigger """
            # if there are less than 200 ms between the trial start trigger and 
            # the epoch onset, they proooobably belong together
            if curr_epoch_onset - curr_trig_timestamp < 0.2 and curr_epoch_onset - curr_trig_timestamp > - 0.2:
                
                print("Found epoch for trigger " + str(trigger_idx) + "!")
                found_epoch = True
                # remove epoch from list of epochs no trigger could be found for
                exclude_these_epochs.remove(epoch_idx)
                
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

                # epoch idx 
                epoch = epoch_idx
                # save information on conditions as well as timestamps & data in list
                gss_epochs_conditions.append([block, sfb, sfc, feedback, epoch])
                                        
                # don't look at the other epochs for this 
                # trigger as we already found a match!
                break

        """ 2.10.4 If there is no epoch for the current trigger, print warning """ 
        if found_epoch == False:
            print("\nEpoch for trigger " + str(trigger_idx) + " could not be found! Sorry girl!")
            
            if len(epoch_data_all) < len(trial_timestamps):
                print("There are less epochs than trial onset triggers.")
            if epochs_excluded == 1:
                print("Could be because 1 epoch had to\nbe excluded because it was too short.")
            elif epochs_excluded > 1:
                print("Could be because " + str(epochs_excluded) + " epochs had to\nbe excluded because they were too short.")

    # runtime for this is a bit longer than necessary because we're always 
    # looping all epochs (also the ones we already found) but f*** it.


    # turn list with epoch data into dataframe:
    gss_epochs_conditions = pd.DataFrame(gss_epochs_conditions, columns=["block", "sfb", "sfc", "feedback", "epoch"])

#%%   

    """ 2.11 Exclude epochs no trigger was found for """
    # small hack: indices are sorted in ascending order, 
    # which means we'd mess the indices up if we remove rows. 
    # And thats why I reverse the order. BAM!
    # (Not that impressive but that's the hack actually.)
    if len(exclude_these_epochs) > 0 :
        for idx in sorted(exclude_these_epochs, reverse = True):              
            del epoch_data_all[idx]
            del epoch_timestamps_all[idx]
            del epoch_sizes[idx]

#%% 
    """ 2.12 Exclude training block and block 3 """
    
    # get indices of all epochs from block 0 (training) and block 3
    idx_b0 = gss_epochs_conditions[gss_epochs_conditions.block == "block0"].index.tolist()
    idx_b3 = gss_epochs_conditions[gss_epochs_conditions.block == "block3"].index.tolist()
    idx_exclude_blocks = idx_b0 + idx_b3
    
    # same "hack" as before: reverse order of indices so this works
    for idx in sorted(idx_exclude_blocks, reverse = True):              
        del epoch_data_all[idx]
        del epoch_timestamps_all[idx]
        del epoch_sizes[idx]
        gss_epochs_conditions.drop(idx, inplace=True)
        
    # assign new indices in gss_epochs_conditions so the indices in the 
    # epoch column still matches the epochs in the MNE epochs object:
    gss_epochs_conditions["epoch"] = list(range(0, len(epoch_data_all)))

#%% 
    """ 2.13 Make sure epochs all have the correct length """
    # Right now, there might be some epochs that are a few 
    # sampling points shorter than others, but for being able to 
    # use MNE Epochs, I need data arrays of the same length so that's a problem.
    
    # IDEA: Set max number of sample points, add more timestamps and 
    # 0s as data if there are not enough.
    
    """ Loop epochs, add 0s at the beginning of the data arrays and more 
        timestamps at the beginnig of the timestamp arrays so that each epoch 
        has exactly the same size (=> max_size). """
        
    # Doesn't matter if there are a few 0s at the beginning I guess,
    # most people start pressing the force sensor a few ms after trial onset anyway.
    
    # loop epochs
    for epoch_idx in range(0, len(epoch_data_all)):
        # check if epoch has the correct length. 
        # If yes, go on to next epoch, if not, add 0s & additional 
        # timestamps at the beginning of the epoch's arrays
        if epoch_sizes[epoch_idx] < max_size:
            
            # get data & timestamps for current epoch
            curr_epoch_data = epoch_data_all[epoch_idx]
            curr_epoch_timestamps = epoch_timestamps_all[epoch_idx]            
        
            # get difference between max_size and length of current epoch
            # (to determine how many additional timestamps & 0s we need)
            diff_size = max_size - len(curr_epoch_data)

            # get median difference between timestamps in array
            timestamps_diff = np.median(np.array(curr_epoch_timestamps[1:]) - np.array(curr_epoch_timestamps[:-1]))

            # do the following as often as needed to append "missing" 0s and timestamps
            for i in range(0, diff_size):
                curr_epoch_data = [0.0] + curr_epoch_data
                curr_epoch_timestamps = [curr_epoch_timestamps[0] - timestamps_diff] + curr_epoch_timestamps

            # mutate original dfs: put arrays back into epoch_data_all and epoch_timestamps_all
            epoch_data_all[epoch_idx] = curr_epoch_data
            epoch_timestamps_all[epoch_idx] = curr_epoch_timestamps

            print("\n\nadded " + str(diff_size) + " missing sample point(s) to epoch " + str(epoch_idx))

    
#%%
    """ 2.14 Create info for MNE Epochs object """

    info_epochs = mne.create_info(ch_names = ['GSS'], 
                               ch_types = ['eeg'], 
                               sfreq = 80)
    
    # look at the info
    # print(info_epochs)
    
         
#%%         
    """ 2.15 Create MNE Epochs object """

    # first, change data shape a little
    
    # create a 2D array
    data_gss_epochs = np.array(epoch_data_all)

    # change the shape of the 2D array to 3D array 
    # by adding 1 dimension
    data_gss_epochs = np.reshape(data_gss_epochs, (data_gss_epochs.shape[0], 1, data_gss_epochs.shape[1])) 

    # check size, should be 3D numpy array with shape 
    # (n_epochs, n_channels = 1, n_samples)
    # print(data_gss_epochs.shape) 
   
    # Now create epochs object:
    gss_epochs = mne.EpochsArray(data_gss_epochs, info_epochs)
    
    # plot it!
    #gss_epochs.plot(n_epochs = 1, scalings = "auto")
    
    # Btw: In MNE, it's not possible to turn on axes descriptions
    # (aka Volt and sec values at the axes ticks) for plotting epochs. 
    # So I'm afraid you can't see when exactly something is 
    # happening or how high the amplitudes are. I felt like a complete 
    # moron asking this in the forum because I thought I just overlooked 
    # a function argument but I didn't. 
    # Could build a plot by hand now but naah.
    
    # Anyway. As you can see, there are sometimes missing data 
    # points in the signal. We'll fix that later in the preproc script. :-)

#%% 
    """ 3. save df with epoched data for each participant in the working directory """ 
    # save epochs
    gss_epochs.save(fname = "gss_participant" + str(participant) + "_epo.fif", fmt = 'single', overwrite = False)
    # save df with info on sfc, sfb, feedback, block & epoch idx
    gss_epochs_conditions.to_csv(path_or_buf = working_directory + "gss_participant" + str(participant) + "_epo_conditions.csv")
       
### END LOOP PARTICIPANTS 

#%%    
""" 4. Create "I'm done!"-message: """
if participant == 1:
    print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in " + str(participant) + " xdf-file for you.\nHave a look at the file you set as a \nworking directory in the function call!\n\n- - - - - - - - - - - - - - - - - - - - - ")
    
elif participant > 1:
    print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I read in " + str(participant) + " xdf-files for you.\nHave a look at the file you set as a \nworking directory in the function call!\n\n- - - - - - - - - - - - - - - - - - - - - ")
    

#%%  

""" 5. Test preproc and stats functions: """

from GSS_preproc import GSS_filter_epoching
from GSS_stats import GSS_stats

GSS_filter_epoching(working_directory)
# (running this takes some time!)

GSS_stats(working_directory)

