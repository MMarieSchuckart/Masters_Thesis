

""" Stats script for Merle's Master's Thesis

Stats part for GSS data

Part of Merle's Master Thesis
Version 1: 13.01.2022

"""


#%%

#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"

# create function for running EEG stats script
def GSS_stats(working_directory, 
              gss_psd_sfreq = 80, 
              gss_psd_fmin = 4, 
              gss_psd_fmax = 12, 
              gss_psd_n_overlap = 0, 
              gss_psd_n_per_seg = None, 
              gss_psd_n_jobs = 1, 
              gss_psd_average = 'mean', 
              gss_psd_window = 'hamming'):
    
    # I set default arguments, but they can be overwritten 
    # if you set different arguments in the function call.

    """ settings """

    """ 1.1 load packages """

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

    # for working with pandas dataframes 
    import pandas as pd
    
    # for rounding frequency values
    import numpy as np

    # for dataframes as in R
    import pandas as pd
    
    # Shapiro-Wilk test function
    from scipy.stats import shapiro
    
    # Levene test function
    from scipy.stats import levene

    # Mauchly's test
    from pingouin import sphericity
    
    # rank transform data
    from scipy.stats import rankdata
    
    # get one way repeated measures ANOVA
    from pingouin import rm_anova
    
    # get one way repeated measures ANOVA
    from pingouin import ttest
    
    # for plotting
    import matplotlib.pyplot as plt


#%%
    """ 2. read in data """
    
    # 2.1 set path to working directory
    os.chdir(working_directory)
    
    # 2.2 get list of all files in the directory that end with "_filtered_epo.fif"
    file_list = glob.glob(working_directory + "gss_" + "*_filtered_epo.fif")
        
    # create empty df for the results of the PSDs of all participants
    gss_PSDs_all = pd.DataFrame(columns = ['block', 'sfb', 'sfc', 'feedback', 'epoch', 'power', 'frequency', 'ID'])

   #%%  
    """ 6. loop fif file names in file_list (aka loop participants): """
    for file_idx in range(0, len(file_list)):
        
        """ save participant number"""
        # the participant numbers in file list are in the wrong order
        # so get the number from the filename instead of the loop index number
        participant = file_list[file_idx][-20:-17]
        
        # if participant number has < 3 digits, shorten the number
        if participant[0] == "n":
            participant = participant[-1]
        elif participant[0] == "t":
            participant = participant[-2:]        
        
        """ read in .fif file with the epochs and .csv file with the epoch conditions"""
        epochs = mne.read_epochs("gss_participant" + str(participant) + "_filtered_epo.fif")
        gss_epochs_conditions = pd.read_csv("gss_participant" + str(participant) + "_epo_conditions.csv", sep = ",")

        # remove weird index column at position 0
        gss_epochs_conditions.drop(columns = gss_epochs_conditions.columns[0], axis = 1, inplace = True)

         
#%%

        """ loop epochs """
        # placeholders for getting power peak and corresponding frequency
        power = []
        freq = []
        
        for pick_epoch in range(0, epochs.__len__()):   
            
            # get current epoch 
            single_epoch = epochs[pick_epoch]
            
            # check if the participant started pressing the sensor before trial onset + 1 sec
            # get data from current epoch, count 0s at the beginning of the epoch
            # --> if the beginning of the epoch looks like this: 
            #     [0,0,0,0,0,0,0,0,0,0,2,3,0,0,0,2,3,4,5,6,3,4,5,9,...]
            #                          ^
            #                          |
            #
            # I assume the participant started using the sensor when the first 
            # "real" values (see amateurish arrow above) were recorded, doesn't matter if 
            # there are 0s afterwards. The reason is I can't know what happened there so 
            # if there are 0s afterwards, maybe the person willingly decided not to use the sensor.
            
            # get data from epoch as list
            epoch_data = list(single_epoch.get_data().flatten())
            
            # if there is at least 1 Zero at the beginning of the recording, 
            # count how many there are and go to next epoch (aka exclude current epoch) 
            # if there are more than 80 (aka 1s * 80Hz)
        
            if epoch_data[0] == 0:
                # counter for zeros
                nr_zeros = 0
                # break while loop condition
                count_on = True
                # start counting at index 0
                idx = 0
                while count_on:
                    # if current value is a 0,
                    if epoch_data[idx] == 0:
                        # add 1 to counter
                        nr_zeros += 1
                        # go to next value
                        idx += 1
                    # if curent value is not a 0...    
                    else: 
                        # stop counting
                        count_on = False
            # if the epochs doesn't start with a 0, we don't have to count            
            else: nr_zeros = 0            

            if nr_zeros >= 80:
                print("Participant didn't react or reacted to slow\nin epoch " + 
                      str(pick_epoch) + ", excluding this epoch now!" )
                # don't compute PSD for this epoch, save None 
                # instead as peak freq and power value:
                power.append(None)
                freq.append(None)
                
            # if everything's fine, compute PSD for current epoch and save results
            else:
            
                # Settings:
                            
                # this is just for testing this part of the script, 
                # I set these as arguments in the function call:
                #gss_psd_sfreq = 80 
                #gss_psd_fmin = 4 
                #gss_psd_fmax = 12 
                #gss_psd_n_overlap = 0 
                #gss_psd_n_per_seg = None 
                #gss_psd_n_jobs = 1
                #gss_psd_average = 'mean' 
                #gss_psd_window = 'hamming'


                # plot psd (no clue which Method this is, the MNE docs don't want me to know this)
                #single_epoch.plot_psd(fmin = 4, fmax = 12, spatial_colors = True)
                 
                
                """compute power spectral density (PSD) analysis using Welch's Method """
                
                # the following function returns... 
                # ...the PSD values for each channel (--> saved in object "psds")... 
                # ...for all frequencies (--> saved in object "freqs")... 
                # ...for the current epoch.
                
                # get data 
                single_epoch = np.array(single_epoch.get_data().flatten())
                
                psds, freqs = mne.time_frequency.psd_array_welch(single_epoch,
                                                                 sfreq = gss_psd_sfreq, 
                                                                 fmin = gss_psd_fmin, 
                                                                 fmax = gss_psd_fmax, 
                                                                 n_fft = len(single_epoch), 
                                                                 n_overlap = gss_psd_n_overlap, 
                                                                 n_per_seg = gss_psd_n_per_seg, 
                                                                 n_jobs = gss_psd_n_jobs, 
                                                                 average = gss_psd_average, 
                                                                 window = gss_psd_window)
                
                # round frequencies
                freqs = np.round(freqs, 1)
    
                # turn psds array into list
                psds = psds.tolist()
    
                # get highest power value and corresponding frequency
                peak_power = max(psds)
                peak_freq = freqs[psds.index(max(psds))]
    
                # save peak frequency & corresponding power value 
                # as temporary 1 row df:
                power.append(peak_power)
                freq.append(peak_freq)
                     
        # END loop epochs
        
        # append freq and power to gss_epochs_conditions
        gss_epochs_conditions["power"] = power
        gss_epochs_conditions["frequency"] = freq
        
        # create list with participant identifier and append to gss_epochs_conditions as well
        gss_epochs_conditions["ID"] = [participant] * len(gss_epochs_conditions)
        
        # append gss_epochs_conditions as new set of rows to df for all participants
        gss_PSDs_all = gss_PSDs_all.append(gss_epochs_conditions)
        
    # END loop participants
          
#%%
   
    """ Compute pairwise comparisons """
    
    # 1st Hypothesis: Higher scaling of Feedback (sfc) should lead to higher tremor amplitudes, regardless of modality:
    #                 Power in sfc: 20% < 25% < 30%
    
    # 2nd Hypothesis: Multisensory Feedback should evoke a stronger reaction aka higher tremor amplitude
    #                 --> Power should be higher in multisensory feedback condition than in 
    #                     auditory and visual condition, but no difference auditory & visual feedback. 
    
         
#%%
    """ TEST 1ST HYPOTHESIS  --> SFC """
    
    """ Aggregate dataframe """
    # I need 1 value for each participant in each sfc condition
    # --> aggregate dataframe
    
    sfc_aggregated = gss_PSDs_all.groupby(['ID', 'sfc']).mean()
    # getting the mean for sfb doesn't make sense so delete that column
    sfc_aggregated = sfc_aggregated.drop(['sfb'], axis=1)
    # use sfc as column instead of index (you have to do this twice)
    sfc_aggregated.reset_index(level = 0, inplace = True)
    sfc_aggregated.reset_index(level = 0, inplace = True)
      
#%%
    """ Test Assumptions of t-Tests / ANOVAs to see if we can run parametrical tests """
    
    # Assumptions of t-Tests:
    # 1. continuous or ordinal scale of scale of measurement --> given in our case
    # 2. random sample = the data is collected from a representative, randomly selected portion of the 
    #                    total population (more or less given, self-selection is real)
    # 3. data in each group are normally distributed (--> check this!)
    # 4. reasonably large sample size (ahahahahahahaha yes sure)
    # 5. homogeneity of variance = Homogeneous (= equal) variance exists when 
    #                              the standard deviations of samples are approximately equal.
    
    # Additional assumption of repeated measures ANOVAs:
    # Sphericity: the variances of differences in responses between any 
    #             two levels of the independent variable (within-subjects factor) 
    #             should be equal. 
    #             This means for 3 groups, you compute the differences between 
    #             group 1 & 2 as well as between group 1 & 3 and group 2 & 3 
    #             and then you check if homogeneity of variance of these differences is given. 
    #             This assumption is therefore also known as 
    #             homogeneity-of-variance-of-differences assumption. :-)
        
         
#%%
    # Test assumption 3 - Normality of Distribution: 
    """ run Shapiro-Wilk Test """
    # We need to test each group separately.
    # If test is significant, distribution is not Gaussian.
    # If it's not significant, it couldn't be shown that it's not Gaussian (≠ it's Gaussian).
    
    # I want to collect my results in a df, so get info on the test, the data we tested and the results:
    gss_results_df = pd.DataFrame(columns = ["test name", "data", "p-values", "test statistics", 
                                             "df", "effect size", "bayes factor", "power"])
    # get unique values in sfc
    sfc_values = list(set(sfc_aggregated["sfc"]))
    # save test name (once for each test we run)
    test_name = ["Shapiro-Wilk Test for Normality of Distribution"]  *  len(sfc_values)
    df = [None]  *  len(sfc_values)
    # empty lists for results
    p_values = []
    Test_statistics = []
    data = []
    # loop sfc values, test distribution and save test results
    for sfc_val in sfc_values :
        # run shapiro wilk test
        stat, p = shapiro(sfc_aggregated[sfc_aggregated["sfc"] == sfc_val]["power"])
        # save results
        p_values.append(p)
        Test_statistics.append(stat)
        data.append("power in trials with sfc = " + str(sfc_val))
    
    # put everything into df
    gss_results_df["test name"] = pd.Series(test_name)
    gss_results_df["data"] = pd.Series(data)
    gss_results_df["p-values"] = pd.Series(p_values)
    gss_results_df["test statistics"] = pd.Series(Test_statistics)
    gss_results_df["df"] = pd.Series(df)
             
#%%
    """ If Normality of Distribution is given..."""
    # If none of the p-values are significant, this could mean that all 
    # distributions are Gaussian, so assumption #3 would be given.

    # --> If all p-values are > 0.05, test homogeneity of variance 
    # to find out if we can use parametrical tests
    
    # before having checked the homogeneity of variance, we assume 
    # it's not given:
    run_parametrical_tests = False
    # We also assume we don't have to use a Greenhouse-Geisser 
    # correction for the ANOVA:
    GG_correction = False
    
    # get data (not a flexible approach if you add more sfc 
    # levels but I don't care rn)    
    sfc_2 = sfc_aggregated[sfc_aggregated["sfc"] == 0.2]["power"]
    sfc_25 = sfc_aggregated[sfc_aggregated["sfc"] == 0.25]["power"]
    sfc_3 = sfc_aggregated[sfc_aggregated["sfc"] == 0.3]["power"]
    
    # if all p-values of the Shapiro-Wilk tests were significant...
    if all(p > 0.05 for p in p_values):
        
        # Test assumption 5 - Normality of Distribution: 
        """ run Levene Test """
        # run Levene test, get p and test statistic        
        stat, p = levene(sfc_2, sfc_25, sfc_3)
        
        # add to results df
        gss_results_df.loc[len(gss_results_df)] = ["Levene Test for Homogeneity of Variance", 
                                                   "all sfc groups", 
                                                   p, stat, None,
                                                   None, None, None]
        
        # If the Levene test was not significant, this means the variances of the groups 
        # were more or less equal. If this is the case, 
        # go on with testing the last assumtion (aka the ANOVA assumption): Sphericity  
        if p > 0.05:
             run_parametrical_tests = True

             # Test ANOVA assumption: Sphericity: 
             """ run Mauchly’s Test """
             spher, stat, chi2, df, p = sphericity(sfc_aggregated, dv = 'power', subject = 'ID', within = "sfc")
             
             # append results to df
             gss_results_df.loc[len(gss_results_df)] = ["Mauchly's Test for Sphericity", 
                                                        "all sfc groups", 
                                                        p, stat, df,
                                                        None, None, None]
             # if sphericity is not given, apply Greenhouse Geisser Correction
             if p <= 0.05: 
                 GG_correction = True

        # else if the Levene test was signifcant, the variance differs between the groups, 
        # so assumption of homogeneity of variance is not given and we can't 
        # run parametrical tests without rank-transforming the data first. 
        # --> Keep run_parametrical_tests = False as we set it before running the Levene test.
        
            
#%% 
    """ If assumptions are violated, rank-transform data """
    
    # If not all of the p-values from the Shapiro-Wilk tests 
    # are > 0.05, at least one of the groups is not 
    # normally distributed, which means parametrical 
    # tests can't be used without rank-transforming the data first. 
    # Same applies if homogeneity of variance is not given.
    # If sphericity is not given, either, apply 
    # GG correction to p-value after computing the ANOVA.
    
    # If assumptions are violated, rank transform data before using ANOVA & t-tests.
    if (any(p <= 0.05 for p in p_values) or run_parametrical_tests == False):    
        # if you have ties (aka >= 2x the same value), assign average rank
        rank_sfc_2 = rankdata(sfc_2, method='average').tolist()
        rank_sfc_25 = rankdata(sfc_25, method='average').tolist()
        rank_sfc_3 = rankdata(sfc_3, method='average').tolist()

        # mutate original pandas df:
        # sort by sfc
        sfc_aggregated = sfc_aggregated.sort_values(by=['sfc'])
        # put rank transformed data into sfc column
        sfc_aggregated["power"] = rank_sfc_2 + rank_sfc_25 + rank_sfc_3

           
#%% 
    """ Repeated Measures ANOVA (for 3 dependent groups) """
    sfc_anova_res = rm_anova(data =  sfc_aggregated, 
                             dv = "power", 
                             within = "sfc", 
                             subject = "ID",
                             correction = True,
                             effsize = "np2")         
    
    # save results
    stat = float(sfc_anova_res["F"])
    df1 = float(sfc_anova_res["ddof1"])
    df2 = float(sfc_anova_res["ddof2"])
    
    # Hint: The effect size partial eta-squared is the same as 
    # eta-squared in the 1-way rep measures ANOVA
    eff_size = str(float(sfc_anova_res["np2"])) + " (partial-eta squared)"
    # If the Mauchly Test was significant, apply GG correction
    if GG_correction:
        p = float(sfc_anova_res["p-GG-corr"])
    else: 
        p = float(sfc_anova_res["p-unc"])
    
    # append results to df
    gss_results_df.loc[len(gss_results_df)] = ["one-way repeated measures ANOVA", 
                                               "all sfc groups", 
                                               p, stat, [df1, df2],
                                               eff_size, None, None]
              
#%%            
    # if ANOVA is significant, run t-tests

    # I assume more feedback (aka higher sfc) --> higher tremor amplitudes
    # --> this is what I'll test in the t-test: 
    #     0.25 > 0.2, 0.3 > 0.2 and 0.3 > 0.25
    pairs = [(0.25, 0.2), (0.3, 0.2), (0.3, 0.25)]

    for pair in pairs:   
        # run one-sided t-test, assume x > y 
       res = ttest(x = np.array(sfc_aggregated[sfc_aggregated["sfc"] == pair[0]]["power"]), 
                   y = np.array(sfc_aggregated[sfc_aggregated["sfc"] == pair[1]]["power"]), 
                   paired = True, 
                   alternative = "greater")
       
       # use Bonferroni correction on p-values:
       p = float(res["p-val"]) * len(pairs)
       
       stat = float(res["T"])
       df = float(res["dof"])
       eff_size = str(float(res["cohen-d"])) + " (Cohen's d)"
       bayes_factor = float(res["BF10"])
       power = float(res["power"])
       
       # append results to df
       gss_results_df.loc[len(gss_results_df)] = ["one-sided t-test", 
                                                  "compare power in sfc: " + str(pair[0]) + " > " + str(pair[1]), 
                                                  p, stat, df,
                                                  eff_size, bayes_factor, power]
       
    
#%%    
    
    """ TEST 2ND HYPOTHESIS --> FEEDBACK """
    
    """ Aggregate dataframe """
    # I need 1 value for each participant in each feedback condition
    # --> aggregate dataframe
    
    feedback_aggregated = gss_PSDs_all.groupby(['ID', 'feedback']).mean()
    # getting the mean for sfb & sfc doesn't make sense so delete those columns
    feedback_aggregated = feedback_aggregated.drop(['sfb'], axis=1)
    feedback_aggregated = feedback_aggregated.drop(['sfc'], axis=1)
    # use feedback as column instead of index (you have to do this twice)
    feedback_aggregated.reset_index(level = 0, inplace = True)
    feedback_aggregated.reset_index(level = 0, inplace = True)

     
#%%   
    
    # Test assumption 3 - Normality of Distribution: 
    """ run Shapiro-Wilk Test """
    # We need to test each group separately.
    # If test is significant, distribution is not Gaussian.
    # If it's not significant, it couldn't be shown that it's not Gaussion (≠ it's Gaussian).
    
    # get unique values in sfc
    feedback_values = list(set(feedback_aggregated["feedback"]))
    # collect p-values for later use
    p_values = []
    # loop feedback values, test distribution and save test results
    for feedback_val in feedback_values :
        # run shapiro wilk test
        stat, p = shapiro(feedback_aggregated[feedback_aggregated["feedback"] == feedback_val]["power"])
        # save results
        test_name = "Shapiro-Wilk Test for Normality of Distribution"
        data = "power in trials with feedback = " + str(feedback_val)
    
        p_values.append(p)
    
        # put everything into df
        # add to results df
        gss_results_df.loc[len(gss_results_df)] = [test_name, 
                                                   data, 
                                                   p, stat, None,
                                                   None, None, None]
    
             
#%%
    """ If Normality of Distribution is given..."""
    # If none of the p-values for the Shapiro-Wilk tests for 
    # the feedback conditions are significant, this could mean 
    # that all distributions are Gaussian, so assumption #3 would be given.

    # --> If all p-values are > 0.05, test homogeneity of variance 
    # to find out if we can use parametrical tests
    
    # before having checked the homogeneity of variance, we assume 
    # it's not given:
    run_parametrical_tests = False
    # We also assume we don't have to use a Greenhouse-Geisser 
    # correction for the ANOVA:
    GG_correction = False
    
    # get data (not a flexible approach if you add more feedback 
    # levels but I don't care rn)    
    feedback_ao = feedback_aggregated[feedback_aggregated["feedback"] == "ao"]["power"]
    feedback_va = feedback_aggregated[feedback_aggregated["feedback"] == "va"]["power"]
    feedback_vo = feedback_aggregated[feedback_aggregated["feedback"] == "vo"]["power"]
    
    # if all p-values of the Shapiro-Wilk tests were significant...
    if all(p > 0.05 for p in p_values):
        
        # Test assumption 5 - Normality of Distribution: 
        """ run Levene Test """
        # run Levene test, get p and test statistic        
        stat, p = levene(feedback_ao, feedback_va, feedback_vo)
        
        # add to results df
        gss_results_df.loc[len(gss_results_df)] = ["Levene Test for Homogeneity of Variance", 
                                                   "all feedback groups", 
                                                   p, stat, None,
                                                   None, None, None]
        
        # If the Levene test was not significant, this means the variances of the groups 
        # were more or less equal. If this is the case, 
        # go on with testing the last assumtion (aka the ANOVA assumption): Sphericity  
        if p > 0.05:
             run_parametrical_tests = True

             # Test ANOVA assumption: Sphericity: 
             """ run Mauchly’s Test """
             spher, stat, chi2, df, p = sphericity(feedback_aggregated,
                                                   dv = 'power', 
                                                   subject = 'ID', 
                                                   within = "feedback")
             
             # append results to df
             gss_results_df.loc[len(gss_results_df)] = ["Mauchly's Test for Sphericity", 
                                                        "all feedback groups", 
                                                        p, stat, df,
                                                        None, None, None]
             # if sphericity is not given, apply Greenhouse Geisser Correction
             if p <= 0.05: 
                 GG_correction = True

        # else if the Levene test was signifcant, the variance differs between the groups, 
        # so assumption of homogeneity of variance is not given and we can't 
        # run parametrical tests without rank-transforming the data first. 
        # --> Keep run_parametrical_tests = False as we set it before running the Levene test.
        
            
#%% 
    """ If assumptions are violated, rank-transform data """
    
    # If not all of the p-values from the Shapiro-Wilk tests 
    # are > 0.05, at least one of the groups is not 
    # normally distributed, which means parametrical 
    # tests can't be used without rank-transforming the data first. 
    # Same applies if homogeneity of variance is not given.
    # If sphericity is not given, either, apply 
    # GG correction to p-value after computing the ANOVA.
    
    # If assumptions are violated, rank transform data before using ANOVA & t-tests.
    if (any(p <= 0.05 for p in p_values) or run_parametrical_tests == False):    
        # if you have ties (aka >= 2x the same value), assign average rank
        rank_feedback_ao = rankdata(feedback_ao, method='average').tolist()
        rank_feedback_va = rankdata(feedback_va, method='average').tolist()
        rank_feedback_vo = rankdata(feedback_vo, method='average').tolist()

        # mutate original pandas df:
        # sort by feedback
        feedback_aggregated = feedback_aggregated.sort_values(by=['feedback'])
        # put rank transformed data into sfc column
        feedback_aggregated["power"] = rank_feedback_ao + rank_feedback_va + rank_feedback_vo

           
#%% 
    """ Repeated Measures ANOVA (for 3 dependent groups) """
    feedback_anova_res = rm_anova(data =  feedback_aggregated, 
                             dv = "power", 
                             within = "feedback", 
                             subject = "ID",
                             correction = True,
                             effsize = "np2")         
    
    # save results
    stat = float(feedback_anova_res["F"])
    df1 = float(feedback_anova_res["ddof1"])
    df2 = float(feedback_anova_res["ddof2"])
    
    # Hint: The effect size partial eta-squared is the same as 
    # eta-squared in the 1-way rep measures ANOVA
    eff_size = str(float(feedback_anova_res["np2"])) + " (partial-eta squared)"
    # If the Mauchly Test was significant, apply GG correction
    if GG_correction:
        p = float(feedback_anova_res["p-GG-corr"])
    else: 
        p = float(feedback_anova_res["p-unc"])
    
    # append results to df
    gss_results_df.loc[len(gss_results_df)] = ["one-way repeated measures ANOVA", 
                                               "all sfc groups", 
                                               p, stat, [df1, df2],
                                               eff_size, None, None]
              
#%%            
    # if ANOVA is significant, run t-tests

    # 1. I assume ao should evoke at least the same tremor intensity as vo, 
    #    so I have to "test" the H0 I guess 
    # 2. I also assume multisensory feedback should lead to higher 
    #    tremor amplitudes than unisensory feedback (ao or vo)
    
    # --> this is what I'll test: 
    #     ao ≠ vo (--> 2-sided, alpha = 20%, hope for non-significant result).     
    #     0.25 > 0.2, 0.3 > 0.2 and 0.3 > 0.25
    pairs = [("ao", "vo"), ("va", "ao"), ("va", "vo")]
    alternative = ["two-sided", "greater", "greater"]
    test_name = ["two-sided t-test (alpha 20%)", "one-sided t-test", "one-sided t-test"]
    data = ["compare power in feedback: ao ≠ vo", 
            "compare power in feedback: va > ao",
            "compare power in feedback: va > vo"]
         
    # loop our planned tests:
    for idx in range(0, len(pairs)): 
        
        # get data for current pair we'd like to compare
        pair = pairs[idx]
        # run one-sided t-test, assume x > y 
        res = ttest(x = np.array(feedback_aggregated[feedback_aggregated["feedback"] == pair[0]]["power"]), 
                    y = np.array(feedback_aggregated[feedback_aggregated["feedback"] == pair[1]]["power"]), 
                    paired = True, 
                    alternative = alternative[idx])
       
        # use Bonferroni correction on p-value:
        p = float(res["p-val"]) * len(pairs)
        
        stat = float(res["T"])
        df = float(res["dof"])
        eff_size = str(float(res["cohen-d"])) + " (Cohen's d)"
        bayes_factor = float(res["BF10"])
        power = float(res["power"])
       
        # append results to df
        gss_results_df.loc[len(gss_results_df)] = [test_name[idx], 
                                                   data[idx], 
                                                   p, stat, df,
                                                   eff_size, bayes_factor, power]
    
          
#%%    

    """ save dataframe with results as .csv """
    gss_results_df.to_csv(path_or_buf = working_directory + "gss_test_results.csv")

    """ Create "I'm done!"-message: """
    print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I saved the results for the\ngss data as 'gss_test_results.csv' in\nthe working directory. I also returned it\nas a new variable called 'gss_results_df'!\n\n- - - - - - - - - - - - - - - - - - - - - ")

    """ return gss_results_df as function output """
    return(gss_results_df)

# END OF FUNCTION     
