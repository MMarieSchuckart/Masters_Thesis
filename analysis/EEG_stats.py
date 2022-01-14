
""" Stats script for Merle's Master's Thesis

Stats part for EEG data

Part of Merle's Master Thesis
Version 1: 14.01.2022

"""


#%%

""" 1.2 Settings for power spectral density (PSD) analysis using Welch's Method """ 

#psd_tmin = 1 
#psd_tmax = 4
#psd_sfreq = 500
#psd_fmin = 4
#psd_fmax = 36 
#psd_n_overlap = 0 
#psd_n_per_seg = None 
#psd_n_jobs = 1
#psd_average = 'mean'
#psd_window = 'hamming'

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

#%%

# create function for running EEG stats script
def EEG_stats(working_directory, 
              psd_tmin = 1, 
              psd_tmax = 4, 
              psd_sfreq = 500, 
              psd_fmin = 4, 
              psd_fmax = 36, 
              psd_n_overlap = 0, 
              psd_n_per_seg = None, 
              psd_n_jobs = 1, 
              psd_average = 'mean', 
              psd_window = 'hamming'):
    
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.

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

    # for rounding frequency values
    import numpy as np

    # for dataframes as in R
    import pandas as pd

    # for computing the multiple linear regression & getting beta values
    import statsmodels.formula.api as smf
    
    # import function for 1-sample t-test
    from scipy.stats import ttest_1samp

    # function for false detection rate (FDR) correction
    from statsmodels.stats.multitest import fdrcorrection as fdr

    # for plotting
    import matplotlib.pyplot as plt


#%%

    """ 2. read in data """
    
    # 2.1 set path to working directory
    os.chdir(working_directory)
    
    # 2.2 get list of all files in the directory that end with "epo.fif"
    file_list = glob.glob(working_directory + "*epo.fif")
    
    # 2.3 create df for power values
    power_vals_all = pd.DataFrame(columns = ["participant", "epoch", "feedback", "sfb", "sfc", "channel", "frequency", "power_value"])
    tmp_df = pd.DataFrame(columns = ["participant", "epoch", "feedback", "sfb", "sfc", "channel", "frequency", "power_value"])
    
 
#%%   
    
    """ 3. Stats Loop Nr 1: 
        compute PSD for each participant, extract Power at each frequency & each channel """
    
    
    """ 3.1  Loop participants"""
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
        part_nr = file
        
        # 3.4.2 get power spectrum for each electrode & each epoch, then extract power at each freq
        
        """ loop epochs """
        for pick_epoch in range(0, epochs.__len__()):
                    
            # get single epoch, crop to get time interval from second 1 - second 4 
            single_epoch = epochs[pick_epoch].crop(tmin = psd_tmin, tmax = psd_tmax)
            
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
          
            # set title of epoch
            #title = "participant " + file + ", epoch " + str(pick_epoch) + " of " + str(len(epochs.picks))
            
            # plot psd (no clue which Method this is, the MNE docs don't want me to know this)
            #single_epoch.plot_psd(fmin = 5, fmax = 35)
            
            """ Loop channels """
            for channel in range(0, len(channels)):    
                # get data for the current channel
                channel_epoched_data = single_epoch._data[0][channel]
    
                # save channel name in df
                channel_nr = single_epoch.ch_names[channel]
                
                """compute power spectral density (PSD) analysis using Welch's Method """
                # the following function returns... 
                # ...the PSD values for each channel (--> saved in object "psds")... 
                # ...for all frequencies (--> saved in object "freqs")... 
                # ...for the current epoch
                psds, freqs = mne.time_frequency.psd_array_welch(channel_epoched_data,
                                                                 sfreq = psd_sfreq, 
                                                                 fmin = psd_fmin, 
                                                                 fmax = psd_fmax, 
                                                                 n_fft = len(channel_epoched_data), 
                                                                 n_overlap = psd_n_overlap, 
                                                                 n_per_seg = psd_n_per_seg, 
                                                                 n_jobs = psd_n_jobs, 
                                                                 average = psd_average, 
                                                                 window = psd_window)
                # round frequencies
                freqs = np.round(freqs, 1)
    
                """ loop frequencies """
                for freq_val in range(0,len(freqs)-1):
    
                    # if freq value is an integer...
                    if freqs[freq_val].is_integer():
                        # get corresponding value from psds array and add to df
                        freq = freqs[freq_val] 
                        psd_val = psds[freq_val]                  
     
                        # save nr of participant, nr of epoch, metadata,
                        # nr of channel, frequency & corresponding power value 
                        # as temporary 1 row df:
                        tmp_df.loc[0] = [part_nr, epoch_nr, feedback, sfb, sfc, channel_nr, freq, psd_val]
                        
                        # append as new row to dataframe containing the values for all participants:
                        power_vals_all = power_vals_all.append(tmp_df)
                    
                # END loop frequencies    
            # END loop channels
        # END loop epochs
        print(" -- finished computing betas for participant " + str(file) + " -- ")
    # END loop participants


#%%

    """ 4. aggregate dataframe """
        
    # get the mean amplitude at each channel for each freq

    # include sfb
    #power_vals_agg = power_vals_all.groupby(["participant", "sfb", "sfc", "feedback", "channel", "frequency"], as_index = False)["power_value"].mean()
    
    # don't include sfb
    #power_vals_agg = power_vals_all.groupby(["participant", "channel", "frequency", "sfc", "feedback"], as_index = False)["power_value"].mean()
    

#%%
        
    """ 5. Stats Loop Nr 2: Loop participants, channels & frequencies
    --> compute beta coefficients """
        
    # recode data
    # for some reason I get multiple betas for feedback if 
    # I pass it as a categorical variable (aka a text string).
    # So recode this:
    # 0 = vo
    # 1 = ao
    # 2 = av
    
    power_vals_all["feedback"] = power_vals_all["feedback"].replace(["vo", "ao", "va"], [0, 1, 2])

    # create empty dfs for beta coefficients
    tmp_df = pd.DataFrame(columns = ["participant", "channel", "freq", "beta_feedback", "beta_sfc"])
    beta_coeffs_res = pd.DataFrame(columns = ["participant", "channel", "freq", "beta_feedback", "beta_sfc"])
    
    # loop participants
    for participant in np.unique(power_vals_all["participant"]):

        # only get data of the current participant
        df_participant = power_vals_all[power_vals_all["participant"] == participant]
        
        # loop channels:
        for channel in np.unique(df_participant["channel"]): 
            # get subset of df with data for the current channel
            df_channel = df_participant[df_participant["channel"] == channel]
        
            # loop frequencies:
            for freq in np.unique(df_channel["frequency"]): 
                # get subset of df with data for the current frequency
                df_freq = df_channel[df_channel["frequency"] == freq]
        
        
        
                """ TO DO: test conditions: sphericity & normality of distribution
                # --> if not given, rank transform data   """
                    
                """ Regression Model: power ~ feedback + sfc """

                # fit model (ols = ordinary least squares; an OLS regression is the same as a linear regression)
                lm = smf.ols(formula='power_value ~ feedback + sfc', data = df_freq).fit()
                
                # get beta coefficients (model parameters) for feedback and sfc:
                # 1st parameter = intercept, 2nd param. = feedback, 3rd param. =  sfc
                intercept, beta_feedback, beta_sfc = lm.params
                                        
                # beta coefficients indicate if there's a relationship between 
                # dependent variable (power value) and independent variables (feedback & sfc) 
                # If yes: beta ≠ 0, if no: beta = 0
                
                # append beta coefficients for current participant, channel & frequency to df
                tmp_df.loc[0] = [participant, channel, freq, beta_feedback, beta_sfc]
                beta_coeffs_res = beta_coeffs_res.append(tmp_df) 
 
#%%                                     
               
    """ 6. Stats Loop Nr 3: Loop channels & frequencies in beta_coeffs_res (--> aggregate over all participants)
    --> compute 1-sample t-test """

    # get list of frequencies and list of channels
    freqs = set(list(beta_coeffs_res["freq"])) 
    channels = set(list(beta_coeffs_res["channel"])) 
     
    
    # prepare empty dfs to save the results from the t-tests:
    tmp_df = pd.DataFrame(columns = ["channel", "freq", 
                                     "p_feedback", "T_feedback",
                                     "p_sfc", "T_sfc"])
    
    t_test_results = pd.DataFrame(columns = ["channel", "freq", 
                                             "p_feedback", "T_feedback",
                                             "p_sfc", "T_sfc"])
    
    # loop channels:
    for channel in channels:
        
        # loop frequencies
        for freq in freqs:

             # get subset of df where channel == current channel & freq == current frequency
             tmp_betas = beta_coeffs_res[(beta_coeffs_res["channel"] == channel) & (beta_coeffs_res["freq"] == freq)]



             """ TO DO: test conditions for running t-tests """




             # run t-tests:
             # for feedback vo
             p_feedback = ttest_1samp(tmp_betas["beta_feedback"], 0, axis=0, alternative='two-sided').pvalue
             T_feedback = ttest_1samp(tmp_betas["beta_feedback"], 0, axis=0, alternative='two-sided').statistic

             # for sfc
             p_sfc = ttest_1samp(tmp_betas["beta_sfc"], 0, axis=0, alternative='two-sided').pvalue
             T_sfc = ttest_1samp(tmp_betas["beta_sfc"], 0, axis=0, alternative='two-sided').statistic
             

             # save results in df
             # append beta coefficients for current participant, channel & frequency to df
             tmp_df.loc[0] = [channel, freq, 
                              p_feedback, T_feedback,
                              p_sfc, T_sfc]
            
             t_test_results = t_test_results.append(tmp_df) 
             

             # False Detection Rate (FDR) Correction to correct for multiple comparisons
             # concerning the method argument: 
             # indep = Benjamini/Hochberg for independent or positively correlated tests (default)

             t_test_results["p_feedback"] = fdr(t_test_results["p_feedback"], alpha = 0.05, method='indep', is_sorted=False)[1]
             t_test_results["p_sfc"] = fdr(t_test_results["p_sfc"], alpha = 0.05, method='indep', is_sorted=False)[1]

             #save file as csv so I can do mad plotting stuff in R 
             t_test_results.to_csv(working_directory + "t_test_results.csv", index = False, float_format='%.16g')
             beta_coeffs_res.to_csv(working_directory + "beta_coeffs_results.csv", index = False, float_format='%.16g')

# END FUNCTION

#%%

""" TO DO:"""
# Which PSD Method (Welch or Multitaper)? 
# nonparametric PSD method: Periodogram as direct transformation of signal (--> Welch method)?
       
