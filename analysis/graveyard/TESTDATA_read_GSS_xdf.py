
""" Reading in GSS test data for testing the preproc & stats scripts

Part of Merle's Master Thesis
Version 1: 02.02.2022

"""


""" 1. load packages """
    
# pyxdf for reading in xdf data:
import pyxdf

# Python MNE for eeg data analysis / processing
# NME should be active by default and should already have the 
# necessary dependencies (numpy, scipy, and matplotlib).    
import mne

# glob for getting all files in a directory
import glob

# os for setting working directory
import os

# For numpy arrays
import numpy as np

# for turning nested list into 1D list
from itertools import chain


# write small function to find out if a value is a float
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False



#%%

 
""" 2. read in data """
# set working directory
os.chdir("/Users/merle/Desktop/Masterarbeit/Master_Testdaten/trmr_fdbck_ms/")

# get list of all xdf files in my directory 
# (the asterix in the path means the name of the 
# file can be anything as long as it has an .xdf ending)
file_list = glob.glob("*.xdf")
    
""" 3. keep track of participants """
participant = 0

  #%%      
""" 4. loop xdf file names in file_list (aka loop participants): """
for file_name in file_list:
    
    # save number of current participant
    participant +=  1
    
    """ 4.1 read in XDF data """
    streams, header = pyxdf.load_xdf(file_name)

    
    """ 4.2 Build NME data object from scratch """
    # stream 0: Arduino - Grip strength sensor data 
    # stream 1: who knows? Just timestamps?!
    # stream 2: PsychoPyMarkers - Experiment triggers

    # each stream contains timestamps (measured in seconds)
    
    
    """ 4.2.1 Create info for Raw Object for GSS data"""
    
    # set sampling rate of Arduino measuring the GSS data:
    sampling_freq_gss = 80
    
    # combine information, name and classify channels while doing so
    info_gss = mne.create_info(ch_names = ["GSS"], 
                               ch_types = ["eeg"], 
                               sfreq = sampling_freq_gss)
    
    # add name of the curent dataset (I could also add information like age or gender here)
    # (change this if the files names of the exp files are named differently)
    info_gss['description'] = file_name[0 : 4]
    
    # look at the info
    #print(info_gss)
    
  #%%     
    """ Find out which stream contains which kind of data """
    # the streams are in a different order in each file, so find out 
    # where the GSS data are and which stream contains the triggers
        
    for stream_idx in range(0, 3):

        # if there are 22 data arrays, it's not the triggers 
        # and probably not the GSS data, so ignore that stream
        if len(streams[stream_idx]["time_series"][0]) != 22:
            # get first time series value from current stream
            curr_stream_val = streams[stream_idx]["time_series"][0][0]
    
            # check if first value is a number or text
            # if it's a number, it's the GGS stream
            if isfloat(curr_stream_val):
                gss_idx = stream_idx
                print(str(stream_idx) + " = GSS")
            # if it's text, it's the trigger stream
            else: 
                trig_idx = stream_idx
                print(str(stream_idx) + " = Triggers")
        # I assume the other stream contains EEG data from 22 channels? Idk. 
        else: print(str(stream_idx) + " = EEG")
    
    print("---------------------------------------")
    
  #%%  
    """ 4.2.2 Get GSS data for Raw object""" 
    # structure should be: 
        # rows: channels
        # columns: sample points
    
    # get GSS data from stream 3:
    # 128 arrays (1 for each electrode), 186013 sampling points
    data_gss = np.array(streams[gss_idx]["time_series"].T) 

    # transform all values in gss_data from Microvolt to Volt 
    # as NME expects time series data of type "eeg" to be measured in Volt (why tho)
    data_gss[:] *= 1e-6
    
 #%%           
    """ 4.2.3 Create Raw object for GSS data""" 
    # combine info & gss data
    gss_Raw = mne.io.RawArray(data_gss, info_gss)



    """ 4.3 Add Annotations """

    """ 4.3.1 Get Triggers & save as Annotations object """

    # get names of triggers (it's a nested list in the xdf file)    
    trigger_descriptions = streams[trig_idx]["time_series"]
    # turn nested list into "normal" one dimensional list
    trigger_descriptions = list(chain.from_iterable(trigger_descriptions)) 
    
    """ 4.3.1.1 change trigger descriptions so the trial start descriptions 
    contain info on the block & feedback condition """
    
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

    # get trigger timestamps as well
    trigger_timestamps = streams[trig_idx]["time_stamps"]

    
    """ 4.3.1.2 save trigger descriptions & their onsets as annotations for our Raw object """
    triggers_annot = mne.Annotations(onset = trigger_timestamps, duration = .001, description = trigger_descriptions)
    
#%% 
    """ 4.3.2 Set annotations """
    # there's a bug in MNE, you have to set a filename 
    # to be able to set annotations:
    gss_Raw._filenames = ['a_random_file_name'] 

    # include gss values as annotations:
    # eeg_Raw.set_annotations(triggers_annot + blink_annot + gss_annot)
    # only use triggers & blinks, not the gss values:
    gss_Raw.set_annotations(triggers_annot)

#%%    
    """ 4.4 plot raw data """
    # I filtered the data a bit, but this is only for the 
    # representation in the plot, it doesn't affect the data!
    # gss_Raw.plot(highpass = 2, lowpass = 15, show_scrollbars = True, scalings = "auto")

 #%%   
    """ 4.5 save Raw object for each participant in the working directory """ 
    gss_Raw.save(fname = "gss_participant" + str(participant) + "_raw.fif", fmt = 'single', overwrite = False)


### END LOOP PARTICIPANTS 

