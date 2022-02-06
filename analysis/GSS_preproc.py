"""
Function for preprocessing GSS data (from fif files)

Part of Merle's Master Thesis
Version 1: 13.01.2022

Input: File containing .fif files with GSS data + triggers for each participant
Output: .fif file containing MNE epochs object with filtered & epoched GSS data
"""

#%% 

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
#gss_bandpass_fmin = 4 
#gss_bandpass_fmax = 12 
#gss_phase = "zero" 
#gss_window_type = 'hamming' 
#gss_fir_design = 'firwin' 
#gss_n_jobs = 1 


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
    
    # for turning multidimensional list into 1D list
    from itertools import chain
    
    # pandas for dataframes
    import pandas as pd
    
    # for plotting
    import matplotlib.pyplot as plt
    
    # for converting arrays into lists
    import numpy as np
    
    # Please make sure you pip installed the hampel package:
    #pip install hampel
    from hampel import hampel
    
    
    #%% 
    
    """ 2. set working directory """
    os.chdir(working_directory)
    
    """ 3. get list of all gss .fif files in my directory """
    # (the asterix in the path means the name of the 
    # file can be anything as long as it has an .xdf ending)
    file_list_epochs = glob.glob(working_directory + "gss_participant" + "*_epo.fif")
        
    """ 4. Create empty lists to keep track of plots (before and after filtering)"""
    #gss_figs_before_filtering = []
    #gss_figs_after_filtering = []
    
    """ 5. keep track of files """
    file = 0
    
   #%%  
   
    """ 6. loop fif file names in file_list (aka loop participants): """
    for file_idx in range(0, len(file_list_epochs)):
        
        """ save participant number"""
        # the participant numbers in file list are in the wrong order
        # so get the number from the filename instead of the loop index number
        participant = file_list_epochs[file_idx][-11:-8]
        
        # if participant number has < 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]        
        
        """ 6.1 read in .fif file with the epochs """
        gss_epochs = mne.read_epochs(file_list_epochs[file_idx])
      
 #%%    
        """ 6.2 Preprocessing """
      
        """ Hampel Filter """
              
        # There's another problem besides the Arduino only recording during trials: 
        # In the trials, there are single sample points missing. 
        # Thankfully, the timestamps are still recorded, the Arduino just didn't record any data 
        # for that timestamp so the value at that point is a 0.
        
        # Solution: Apply Hampel Filter
        #https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d


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
            
        #      # get unfiltered and hampel-filtered data 
        #      unfiltered = epochs_data[epoch_idx,:,:][0]
        #      filtered = epochs_data_hamfilt[epoch_idx]
            
        #      # create time vector
        #      times = np.linspace(0, len(filtered)/sample_rate, sample_rate, endpoint = False) 
        
        
        #      # plot signal before and after having used the Hampel filter
        #      plt.figure(figsize=(8, 4))   
        #      plt.plot(times[:sample_rate], unfiltered[:sample_rate], color='indianred', alpha = 1)
        #      plt.plot(times[:sample_rate], filtered[:sample_rate], color='teal', alpha = 1, linestyle = 'dotted')
        #      plt.xlabel('Zeit (in s)')
        #      plt.ylabel('Amplitude (in V)')
        #      plt.title('epoch ' + str(epoch_idx))
        #      plt.legend(['Signal vor dem Filtern', 'Signal nach dem Filtern'], loc = 'lower right')
        #      plt.show()
        
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
        
        # Took this part out because I think it's pretty useless to filter the signal 
        # if I only want to compute PSDs anyway?
        
        
        # (variables are defined at the beginning of the script))
             
        #gss_bandpass_fmin = 4 
        #gss_bandpass_fmax = 12 
        #gss_phase = "zero" 
        #gss_window_type = 'hamming' 
        #gss_fir_design = 'firwin' 
        #gss_n_jobs = 1 
        
        #gss_epochs_filtered = gss_epochs.copy()
        #gss_epochs_filtered.filter(l_freq = gss_bandpass_fmin, 
        #                           h_freq = gss_bandpass_fmax,   
        #                           phase = gss_phase,
        #                           fir_window = gss_window_type, 
        #                           fir_design = gss_fir_design, 
        #                           n_jobs = gss_n_jobs)
        
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
        gss_epochs.save(fname = "gss_participant" + str(participant) + "_filtered_epo.fif", fmt = 'single', overwrite = False)
        #gss_epochs_filtered.save(fname = "gss_participant" + str(participant) + "_epo.fif", fmt = 'single', overwrite = False)

    
        # save number of processed files
        file += 1

    # END LOOP   
    
    
    #%% 
    
    """ 8. Create "I'm done!"-message: """
    if file == 0:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, something went wrong: I couldn't run the preproc function on any file. Maybe you should have a look at this.\n\n- - - - - - - - - - - - - - - - - - - - - ")
    
    elif file == 1:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched 1 file for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the file gss_participant1_filtered_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    else:
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I filtered & epoched " + str(file) + " files for you.\n\nHave a look at the file you set as a \nworking directory in the function call!\n\nI called the files gss_participant[number]_filtered_epo.fif\n\n- - - - - - - - - - - - - - - - - - - - - ")
        
    
# END FUNCTION
