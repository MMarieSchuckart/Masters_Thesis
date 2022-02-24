
""" Stats script for Merle's Master's Thesis

Stats for EEG data
Part 1: compute betas and t-tests 

Part of Merle's Master Thesis
Version 1: 14.01.2022

"""


#%%

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

#%%

# create function for running EEG stats script
def EEG_stats_ttests(working_directory, 
                     psd_tmin = 1, 
                     psd_tmax = 6, 
                     psd_sfreq = 500, 
                     psd_fmin = 4, 
                     psd_fmax = 35, 
                     psd_n_fft = 500,
                     psd_n_overlap = 100, 
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
    
    # Durbin Watson Test to check assumption of 
    # multicollinearity of linear regression
    from statsmodels.stats.stattools import durbin_watson
    
    # import function for 1-sample t-test
    #from scipy.stats import ttest_1samp
    from pingouin import ttest

    # function for false detection rate (FDR) correction
    from statsmodels.stats.multitest import fdrcorrection as fdr

    # for plotting
    import matplotlib.pyplot as plt
    
    # for plotting assumptions tests
    import seaborn as sns
    # settings for seaborn plots
    sns.set_style('darkgrid')
    sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
    
    # for testing multicollinearity of data
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    
    # for testing homoscedasticity of data
    import statsmodels.stats.api as sms   

    # rank transform data
    from scipy.stats import rankdata

    # Shapiro-Wilk test function
    from scipy.stats import shapiro    

    # for getting all possible pairs of elements from 2 lists
    from itertools import product    

#%%

    """ 2. read in data """
    
    # 2.1 set path to working directory
    os.chdir(working_directory)
    
    # 2.2 get list of all files in the directory that start with "eeg" and end with "epo.fif"
    file_list = glob.glob(working_directory + "eeg*epo.fif")
 
    # check if there are data from enough subjects to run the tests
    # (you need at least 3 values in each group for most tests to work and 
    # more if you want meaningful results)
    if len(file_list) < 4:
        # print warning if there are not enough participants to run tests
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nWARNING! \n\nYou have data of less than 4 participants. \nYou can't run some statistical tests with this! \n\nRunning this function was stopped. \n\n- - - - - - - - - - - - - - - - - - - - - ")
        # stop the further execution of this function
        return
    
    elif len(file_list) >= 4 and len(file_list) < 10:
        # print warning if there are not enough participants to run tests
        print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nWARNING! \n\nYou have data of less than 10 participants! \nYou can run statistical tests with this \nbut be careful with the interpretation of the results! \n\n- - - - - - - - - - - - - - - - - - - - - ")

    # If everything's fine, go on!
    
#%%   
    
    """ 3. Stats Loop Nr 1: 
        compute PSD for each participant, extract Power at each frequency & each channel """
    
    # create df for power values
    power_vals_all = pd.DataFrame(columns = ["participant", "epoch", "feedback", "sfb", "sfc", "channel", "frequency", "power_value"])
    tmp_df = pd.DataFrame(columns = ["participant", "epoch", "feedback", "sfb", "sfc", "channel", "frequency", "power_value"])
        
    
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
        
        """ 3.2 Loop epochs """
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
            
            """ 3.3 Loop channels """
            for channel in range(0, len(channels)):    
                # get data for the current channel
                channel_epoched_data = single_epoch._data[0][channel]
    
                # save channel name in df
                channel_nr = single_epoch.ch_names[channel]
                
                """ compute power spectral density (PSD) analysis using Welch's Method """
                # the following function returns... 
                # ...the PSD values for each channel (--> saved in object "psds")... 
                # ...for all frequencies (--> saved in object "freqs")... 
                # ...for the current epoch
                psds, freqs = mne.time_frequency.psd_array_welch(channel_epoched_data,
                                                                 sfreq = psd_sfreq, 
                                                                 fmin = psd_fmin, 
                                                                 fmax = psd_fmax, 
                                                                 n_fft = 500, 
                                                                 n_overlap = psd_n_overlap, 
                                                                 n_per_seg = psd_n_per_seg, 
                                                                 n_jobs = psd_n_jobs, 
                                                                 average = psd_average, 
                                                                 window = psd_window)
    
                """ 3.4 Loop frequencies """
                for freq_idx in range(0, len(freqs)):

                    # get frequency and corresponding value from psds 
                    # array and add both to df
                    freq = freqs[freq_idx] 
                    psd_val = psds[freq_idx]                 
     
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

    """ 4. Create new columns: ROI & Frequency Band """
    
    # currently, the index is 0 in every row, 
    # so make a neat new index column instead:
    power_vals_all.reset_index(drop = True, inplace = True)


    # Now create 2 new grouping variables: ROI and frequency band
    ROI = []
    freq_band = []

    # loop rows in power_vals_all    
    for idx in range(0, len(power_vals_all)):
        
        # 1. Frequency Band
        # get current frequency 
        curr_freq = power_vals_all["frequency"][idx]
        # check which frequency band the frequency falls 
        # in & save name of correct frequency band:
        if curr_freq < 8:
            freq_band.append("theta")
        elif curr_freq >= 8 and curr_freq <= 12:
            freq_band.append("alpha")
        elif curr_freq > 12 and curr_freq <= 30:
            freq_band.append("beta")
        elif curr_freq > 30:
            freq_band.append("gamma")
            
        # 2. ROI
        # get current channel
        curr_channel = power_vals_all["channel"][idx]
        # assign ROI name based on location
        if curr_channel in ["EEG_001", "EEG_069", "EEG_068",
                            "EEG_033", "EEG_038","EEG_066", 
                            "EEG_065"]:
            ROI.append("auditory")
        elif curr_channel in ["EEG_034", "EEG_002", "EEG_071",
                              "EEG_077", "EEG_005", "EEG_035",
                              "EEG_078"]: 
            ROI.append("motor")
        elif curr_channel in ["EEG_108", "EEG_054", "EEG_055",
                              "EEG_061", "EEG_117", "EEG_118",
                              "EEG_109", "EEG_063"]: 
            ROI.append("visual")
        else: 
            print(str(curr_channel) + "not found!")
    
    # append new columns to df:
    power_vals_all['ROI'] = ROI
    power_vals_all['freq_band'] = freq_band
        

#%%
        
    """ 5. Stats: Compute linear regression & t-Tests """
    
    """ Test Assumptions of t-Tests / ANOVAs """
        
    # Assumption of linear regressions:
    # 1. Linearity of the data: There is a linear relationship between  
    #                           the predictors (x) and the outcome (y)
    # 2. Multicollinearity: Predictors (x) are independent and observed with negligible error
    # 3. Mean value of the Residual Errors = 0
    # 4. Constant Variance of Residual Errors
    # 5. Residual Errors are independent from each other and predictors (x)
    # --> if violated: Idk. Do it like everyone else, ignore it and report it in the paper I guess?!
    
    # Assumptions of t-Tests:
    # 1. continuous or ordinal scale of scale of measurement --> given in our case
    # 2. random sample = the data is collected from a representative, randomly selected portion of the 
    #                    total population (more or less given, self-selection is real)
    # 3. data are normally distributed (--> check this!)
    # 4. reasonably large sample size
    # --> if violated: rank transform data before running t-Tests!

#%%

    """ 5.1 Stats Loop Nr 2: Loop participants, ROIs & frequency bands
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
    tmp_df_betas = pd.DataFrame(columns = ["participant", "ROI", "freq_band", "beta_feedback", "beta_sfc"])
    beta_coeffs_res = pd.DataFrame(columns = ["participant", "ROI", "freq_band", "beta_feedback", "beta_sfc"])

    """ 5.1.1 Loop participants """
    for participant in np.unique(power_vals_all["participant"]):

        # only get data of the current participant
        df_participant = power_vals_all[power_vals_all["participant"] == participant]
        
        """ 5.1.2 loop ROIs: """
        for roi in np.unique(df_participant["ROI"]): 
            # get subset of df with data for the current channel
            df_roi = df_participant[df_participant["ROI"] == roi]
        
            """ 5.1.3 loop frequency bands: """
            for freq_band in np.unique(df_roi["freq_band"]): 
                # get subset of df with data for the current frequency
                df_freq_band = df_roi[df_roi["freq_band"] == freq_band]
                    
                """ 5.1.4 Regression Model: power ~ feedback + sfc """
                # Hint for Future-Merle: if you use OLS(), you need to add a constant/intercept, 
                # if you use ols() like I did here, it's added automatically by default.
                
                # fit model (ols = ordinary least squares; an OLS 
                # regression is the same as a linear regression)
                lm = smf.ols(formula='power_value ~ feedback + sfc', data = df_freq_band).fit()

                
                """ 5.1.5 check assumptions of linear regression """
                # Assumptions of OLS regressions:
                # 1. Linearity of data
                # 2. all groups have a normal distribution
                # 3. Mean of Residuals = 0
                # 4. little or no multicollinearity of the residuals
                # 5. little to no autocorrelation of the residuals
                # 6. Homoscedasticity of the residuals
                
                # --> if assumptions are violated: Idk. Do it like everyone else, 
                # ignore it and report it in the paper I guess?!
                
                # So print "assumption violated" if one of the assumptions is violated.
                # Don't know how to automate this for the visual checks though.
                
                # rank transform data if assumptions are violated
                # we haven't detected any violation yet, so...
                rank_transform = False                       
                
                """ 5.1.5.1 Assumption 1: Linearity of data """
                # You can only check this by visual inspection
                # uncomment this section if you want to plot everything:
                    
                #fitted_vals = lm.predict()
                #resids = lm.resid

                #fig, ax = plt.subplots(1,2)
    
                #sns.regplot(x = fitted_vals, 
                #            y = df_freq_band["power_value"], 
                #            lowess = True, 
                #            ax = ax[0], 
                #            line_kws = {'color': 'red'})
                #ax[0].set_title('Observed vs. Predicted Values', 
                #                fontsize = 16)
                #ax[0].set(xlabel = 'Predicted', 
                #          ylabel = 'Observed')

                #sns.regplot(x = fitted_vals, 
                #            y = resids, 
                #            lowess = True, 
                #            ax = ax[1], 
                #            line_kws = {'color': 'red'})
                #ax[1].set_title('Residuals vs. Predicted Values', 
                #                fontsize = 16)
                #ax[1].set(xlabel = 'Predicted', 
                #          ylabel = 'Residuals')
                # Ah yes, this looks super weird. Nice.
                

                """ 5.1.5.2 Assumption 2: the groups follow a normal distribution """
                # We need to test each group separately.
                # If test is significant, distribution is not Gaussian - rank transform data 
                # before computing tests
                # If it's not significant, it couldn't be shown that it's not Gaussian (≠ it's Gaussian).

                # get unique values in sfc & feedback
                sfc_values = list(set(df_freq_band["sfc"]))
                feedback_values = list(set(df_freq_band["feedback"]))
                # get all possible combinations of feedback x sfc
                pairs = list(product(feedback_values, sfc_values))
                
                # empty list for results
                p_values = []

                # loop pairs, get data for group, 
                # test distribution and print warning if one 
                # of the groups is not normally distributed
                for feedback_val, sfc_val in pairs:
                    
                    # get data
                    tmp_df = df_freq_band[(df_freq_band.sfc == sfc_val) & 
                                          (df_freq_band.feedback == feedback_val)]
                    # run Shapiro-Wilk test if there are enough values to do so
                    if len(tmp_df) > 2:
                        stat, p = shapiro(tmp_df["power_value"])
                    else: 
                        p = 0    
                    # collect data
                    p_values.append(p)
                    
                # if one of the results is significant or there were 
                # not enough values in one of the dfs, print warning:
                if any(p <= 0.05 for p in p_values):
                    print("\n\nWARNING: \nParticipant " + participant + ", ROI: " + 
                          roi + ", Freq. Band: " + freq_band + 
                          "\nAssumtion of normally distributed data \nin all groups is violated!\n\n")
                    rank_transform = True
                
                """ 5.1.5.3 Assumption 3: Mean of Residuals = 0 """
                # Get mean of residuals. This can be an insanely small number, 
                # so I thought I round to 3 digits after the point and then I compare 
                # it to 0. I hope that's not cheating.
                
                # if ≠ 0, print warning
                if round(lm.resid.mean(), 3) != 0:
                    print("\n\nWARNING: \nParticipant " + participant + ", ROI: " + 
                          roi + ", Freq. Band: " + freq_band + 
                          "\nMean of Residuals is " + str(round(lm.resid.mean(), 3)) +
                          ", so assumtion that mean of residuals = 0 is violated!\n\n")
                    rank_transform = True

                """ 5.1.5.4 Assumption 4: little to no multicollinearity of residuals """
                # The features should be linearly independent aka we should not be 
                # able to use a linear model to accurately predict 
                # one feature using another one.
                # --> detect multicollinearity using the 
                # variance inflation factor (VIF)
                
                # create small df for predictors
                X = df_freq_band[["feedback", "sfc"]]
                # add constant 
                X_constant = sm.add_constant(X)
                # compute VIFs
                vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
                #vif_df = pd.DataFrame({'vif': vif[1:]}, index = X.columns).T
                
                # If no features are correlated, then all values for VIF will be 1.
                # So print a warning if this is not the case.
                if any(vif[1:]) != 1: 
                    print("\n\nWARNING: \nParticipant " + participant + ", ROI: " + 
                          roi + ", Freq. Band: " + freq_band + 
                          "\nAssumtion of no multicollinearity in the data is violated!\nYour predictors are correlated, try 'merging' them to 1 variable!\n\n")
                    rank_transform = True
                    
                """ 5.1.5.5 Assumption 5: little to no autocorrelation of residuals """
                # --> Durbin Watson Test
                # If test statistic of Durbin Watson Test is outside a range 
                # of 1.5 - 2.5, you have autocorrelation in your data 
                dw_test_stat = round(durbin_watson(lm.resid), 3)
                if dw_test_stat < 1.5 or dw_test_stat > 2.5:
                    print("\n\nWARNING: \n\nParticipant " + participant + ", ROI: " + 
                          roi + ", Freq. Band: " + freq_band + 
                          "\nDurbin Watson Test statistic is " + str(dw_test_stat) +
                          ", assumtion of multicollinearity is violated!\n\nThere seems to be another independent variable that's systematically affecting your data!\n\n")
                    

                """ 5.1.5.6 Assumption 6: Homoscedasticity (equal variance) of residuals """
                # --> Goldfeld-Quandt Test
                # If significant, data don't have equal variances (= Heteroscedasticity)
                # which is not great.
                gq_test = pd.DataFrame(sms.het_goldfeldquandt(lm.resid, lm.model.exog)[:-1],
                                       columns=['value'],
                                       index=['F statistic', 'p-value'])
                
                if round(gq_test["value"]["p-value"], 3) <= 0.05:
                    print("\n\nWARNING: \nParticipant " + participant + ", ROI: " + 
                          roi + ", Freq. Band: " + freq_band + 
                          "\nHeteroscedasticity detected! Assumtion of homoscedasticity is violated!\n\n")
                    # For the record: I wanted to print this gif as an addition 
                    # to the hetero-warning-message: 
                    # https://giphy.com/gifs/queereye-queer-eye-6-queereye6-XCo9O0xRumy2F4iqLa
                    rank_transform = True

                """ 5.1.6 get beta coefficients (model parameters) for feedback and sfc """

                # rank transform data if necessary:
                if rank_transform:
                    df_freq_band["power_value"] = rankdata(df_freq_band["power_value"], method='average').tolist()

                    # fit model again with rank transformed data:
                    lm = smf.ols(formula='power_value ~ feedback + sfc', data = df_freq_band).fit()
                    
                # 1st parameter = intercept, 2nd param. = feedback, 3rd param. =  sfc
                intercept, beta_feedback, beta_sfc = lm.params
                                        
                # beta coefficients indicate if there's a relationship between 
                # dependent variable (power value) and independent variables (feedback & sfc) 
                # If yes: beta ≠ 0, if no: beta = 0
                
                # append beta coefficients for current participant, channel & frequency to df
                tmp_df_betas.loc[0] = [participant, roi, freq_band, beta_feedback, beta_sfc]
                beta_coeffs_res = beta_coeffs_res.append(tmp_df_betas) 
       

            
#%%                                     
               
    """ 6. Stats Loop Nr 3: Loop channels & frequencies in beta_coeffs_res (--> aggregate over all participants)
    --> compute 1-sample t-test """


    """ 6.1 get list of frequencies and list of channels """
    freq_bands = set(list(beta_coeffs_res["freq_band"])) 
    ROIs = set(list(beta_coeffs_res["ROI"])) 
     
    
    # prepare empty dfs to save the results from the t-tests:
    tmp_df = pd.DataFrame(columns = ["ROI", "freq_band", 
                                     "p_feedback", "T_feedback", "df_feedback", "cohens_d_feedback", "BF_feedback",
                                     "p_sfc", "T_sfc", "df_sfc", "cohens_d_sfc", "BF_sfc"])
    
    t_test_results = pd.DataFrame(columns = ["ROI", "freq_band", 
                                             "p_feedback", "T_feedback", "df_feedback", "cohens_d_feedback", "BF_feedback",
                                             "p_sfc", "T_sfc", "df_sfc", "cohens_d_sfc", "BF_sfc"])
    
    tmp_shapiro_df = pd.DataFrame(columns = ["ROI", "freq_band", 
                                             "p_feedback", "stat_feedback",
                                             "p_sfc", "stat_sfc"])
    
    shapiro_results = pd.DataFrame(columns = ["ROI", "freq_band", 
                                              "p_feedback", "stat_feedback",
                                              "p_sfc", "stat_sfc"])
    
    
    """ 6.2  loop channels: """
    for roi in ROIs:
        
        # loop frequencies
        for freq_band in freq_bands:

             # get subset of df where ROI == current ROI & freq_band == current freq_band
             tmp_betas = beta_coeffs_res[(beta_coeffs_res["ROI"] == roi) & 
                                         (beta_coeffs_res["freq_band"] == freq_band)]


             """ 6.3 test conditions for 1-sample t-tests """
             # Assumptions:
             # scale of measurement: continuously or ordinaly scaled
             # simple random sample
             # normal distribution --> check this
             # reasonably large sample size
             
             """ Test Normality of Distribution"""
             # --> Shapiro-Wilk Test
             stat_feedback, p_feedback = shapiro(tmp_betas["beta_feedback"])
             stat_sfc, p_sfc = shapiro(tmp_betas["beta_feedback"])

             # save results!
             tmp_shapiro_df.loc[0] = [roi, freq_band, 
                                      p_feedback, stat_feedback,
                                      p_sfc, stat_sfc]
            
             shapiro_results = shapiro_results.append(tmp_shapiro_df) 

             # if the data are not normally distributed, rank transform them before computing t-tests.
             if p_feedback <= 0.05:
                 tmp_betas["beta_feedback"] = rankdata(tmp_betas["beta_feedback"], method='average').tolist()

             if p_sfc <= 0.05:
                 tmp_betas["beta_sfc"] = rankdata(tmp_betas["beta_sfc"], method='average').tolist()


             """ 6.4 run 1-sample t-tests: """
             
             # for feedback
             feedback_ttest_res = ttest(x = tmp_betas["beta_feedback"],
                                        y = 0, 
                                        alternative='two-sided')
             # get results
             p_feedback = feedback_ttest_res["p-val"][0]
             T_feedback = feedback_ttest_res["T"][0]
             df_feedback = feedback_ttest_res["dof"][0]
             cohens_d_feedback = feedback_ttest_res["cohen-d"][0]
             BF_feedback = float(feedback_ttest_res["BF10"][0])
             
             
             
             # same for sfc:
             sfc_ttest_res = ttest(x = tmp_betas["beta_sfc"],
                                        y = 0, 
                                        alternative='two-sided')
             # get results
             p_sfc = sfc_ttest_res["p-val"][0]
             T_sfc = sfc_ttest_res["T"][0]
             df_sfc = sfc_ttest_res["dof"][0]
             cohens_d_sfc = sfc_ttest_res["cohen-d"][0]
             BF_sfc = float(sfc_ttest_res["BF10"][0])
             
             # save results in df
             # append results for current channel & frequency to df
             tmp_df.loc[0] = [roi, freq_band, 
                              p_feedback, T_feedback, df_feedback, cohens_d_feedback, BF_feedback,
                              p_sfc, T_sfc, df_sfc, cohens_d_sfc, BF_sfc]
             # append to big df
             t_test_results = t_test_results.append(tmp_df) 
             

    """ 6.5 False Detection Rate (FDR) Correction to correct for multiple comparisons """
    # concerning the method argument: 
    # indep = Benjamini/Hochberg for independent or positively correlated tests (default)

    t_test_results["p_feedback"] = fdr(t_test_results["p_feedback"], 
                                       alpha = 0.05, 
                                       method = 'indep', 
                                       is_sorted = False)[1]
    t_test_results["p_sfc"] = fdr(t_test_results["p_sfc"], 
                                  alpha = 0.05, 
                                  method = 'indep', 
                                  is_sorted = False)[1]

#%%
    """ 7. save results files as csv """
    t_test_results.to_csv(working_directory + "t_test_results.csv", 
                          index = False, 
                          float_format = '%.16g')
    beta_coeffs_res.to_csv(working_directory + "beta_coeffs_results.csv", 
                           index = False, 
                           float_format = '%.16g')

#%%

    """ 8. Create "I'm done!"-message: """
    print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I saved the results for the\nEEG data (part 1: betas & t-test results) as \n't_test_results.csv' and 'beta_coeffs_results.csv' \nin the working directory.\n\n- - - - - - - - - - - - - - - - - - - - - ")


# END FUNCTION
