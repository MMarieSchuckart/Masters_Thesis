
""" Stats script for Merle's Master's Thesis

Stats part for GSS data

Part of Merle's Master Thesis
Version 1: 13.01.2022

"""

#%%

# wd --> "normal" data:
#working_directory = "/Users/merle/Desktop/Masterarbeit/Master_Testdaten/"
    
#%%

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

    """ 1. Settings """

    """ load packages """

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
    
    # for getting catesian products from 2 lists
    from itertools import product
    
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
        
    # 2.3 check if there are data from enough subjects to run the tests
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

#%%
    # If everything is fine, go on:
    # create empty df for the results of the PSDs of all participants
    gss_PSDs_all = pd.DataFrame(columns = ['block', 'sfb', 'sfc', 'feedback', 'epoch', 'power', 'frequency', 'ID'])

    """ 3. loop particiants: """
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
        """ 3. Get PSDs for each participant at each epoch """
        # Hint: We don't loop channels here because we only have one ;-)
        
        """ 3.1 Loop epochs """
        # placeholders for getting power peak and corresponding frequency
        power = []
        freq = []
        
        for pick_epoch in range(0, epochs.__len__()):   
            
            """ 3.1.1 get current epoch """ 
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
            
            """ 3.1.2 get data from epoch, exclude epoch if there's at least 1 sec missing """
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
                
           
            else:
                """ 3.1.3 if everything's fine, compute PSD for current epoch and save results """
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
                 
                
                """ 3.1.3.1 compute power spectral density (PSD) analysis using Welch's Method """
                
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
                
                #  round frequencies
                freqs = np.round(freqs, 1)
    
                # turn psds array into list
                psds = psds.tolist()
    
                """ 3.1.3.2 get highest power value and corresponding frequency """
                peak_power = max(psds)
                peak_freq = freqs[psds.index(max(psds))]
    
                # save peak frequency & corresponding power value 
                # as temporary 1 row df:
                power.append(peak_power)
                freq.append(peak_freq)
                     
        # END loop epochs
        
        """ 3.1.4 put frequency and power values into df gss_PSDs_all """
        # append freq and power to gss_epochs_conditions
        gss_epochs_conditions["power"] = power
        gss_epochs_conditions["frequency"] = freq
        
        # create list with participant identifier and append to gss_epochs_conditions as well
        gss_epochs_conditions["ID"] = [participant] * len(gss_epochs_conditions)
        
        # append gss_epochs_conditions as new set of rows to df for all participants
        gss_PSDs_all = gss_PSDs_all.append(gss_epochs_conditions)
        
    # END loop participants
          
#%%
    """ 4. Compute pairwise comparisons """

    """ 4.1 Hypotheses """
    
    # 1st Hypothesis: Effect of SFC
    # Higher scaling of Feedback (sfc) should lead to higher tremor amplitudes.
    # Power in sfc: 20% < 25% < 30%
    
    # Hypothesis 1.1:
    # This should especially be the case in the only visual feedback condition ov (--> Archer replication), 
    # otherwise this means  we couldn't show that more visual feedback
    # leads to higher tremor amplitudes, so the replication failed.
    # H1: Power(vo, sfc = 20%) < Power(vo, sfc = 25%) < Power(vo, sfc = 30%)
    
    # Hypothesis 1.2:
    # If the effect of sfc could be shown for the visual feedback, 
    # this should also be the case for the unisensory auditory condition ao... 
    # H1: Power(ao, sfc = 20%) < Power(ao, sfc = 25%) < Power(ao, sfc = 30%)
    
    # Hypothesis 1.3:
    # ...and for the multisensory audiovisual condition va 
    # H1: Power(va, sfc = 20%) < Power(va, sfc = 25%) < Power(va, sfc = 30%)
    
    # -----------------------------------------------------------------------
    
    # 2nd Hypothesis: Effect of Feedback condition
    # Multisensory Feedback should evoke a stronger reaction aka higher tremor amplitudes (regardless of sfc)
    # Power in va > Power in vo and ao
    
    # Hypothesis 2.1: 
    # Power should be higher in multisensory feedback condition va than in unisensory auditory condition ao
    # H1: Power(va) > Power(ao)
    
    # Hypothesis 2.2: 
    # Power should be higher in multisensory feedback condition va than in unisensory visual condition vo
    # H1: Power(va) > Power(vo)
    
    # Hypothesis 2.3: 
    # No difference in Power between the visual and the auditory feedback condition (vo and ao)
    # H0: Power(vo) = Power(ao)
    # "test" H0 with alpha = 20%, hope for non-significant result, get Bayes factor

    
# %%

    """ 4.2 Assumptions of t-Tests / ANOVAs """

    # (we need to check the green ones before using ANOVAs / t-Tests)

    # Assumptions of t-Tests:
    #   1. continuous or ordinal scale of scale of measurement --> given in our case
    #   2. random sample = the data is collected from a representative, randomly selected portion of the 
    #                    total population (more or less given, self-selection is real)
    """ 3. Normality of distribution --> Shapiro-Wilk Test """
    #      data in each group are normally distributed (--> check this!)
    #   4. reasonably large sample size (ahahahahahahaha yes sure)
    """ 5. Homogeneity of variance --> Levene Test """ 
    #      = Homogeneous (= equal) variance exists when the standard 
    #      deviations of samples are approximately equal. (--> check this!)
    
    
    # Additional assumption of repeated measures ANOVAs:    
    """ Sphericity --> Mauchly's Test """
    # the variances of differences in responses between any two levels of the independent 
    # variable (within-subjects factor) should be equal. 
    # This means for 3 groups, you compute the differences between group 1 & 2 as well 
    # as between group 1 & 3 and group 2 & 3 and then you check if homogeneity of 
    # variance of these differences is given. This assumption is therefore also known 
    # as the homogeneity-of-variance-of-differences assumption. :-)
    # (--> check this!)

         
#%%
    """ 4.2.1 Test Assumptions of t-Tests / ANOVAs to see if we can run parametrical tests """
    
    # I'll do all assumptions tests for all groups I'll test 
    # later on now (I don't do it individually for hypotheses 1 and 2)  
  
    """ 4.2.1.1 Aggregate dataframe """
    # h1 = data for hypotheses 1.1 - 1.3, h2 = data for hypotheses 2.1 - 2.3
    
    # For testing hypotheses 1.1 - 1.3, I need 1 value for each participant 
    # in each sfc & feedback condition --> aggregate dataframe
    df_aggregated_h1 = gss_PSDs_all.groupby(["ID", "sfc", "feedback"]).mean()
    # use ID, sfc & feedback as columns instead of indices 
    # (= you have to reset the index 3x, once for each variable)
    for i in range(0,3):
        df_aggregated_h1.reset_index(level = 0, inplace = True)
        
    # For testing hypotheses 2.1 - 2.3, I need 1 value for each participant 
    # in each feedback condition --> aggregate dataframe
    df_aggregated_h2 = gss_PSDs_all.groupby(["ID", "feedback"]).mean()
    # use ID & feedback as columns instead of indices 
    # (= you have to reset the index 2x, once for each variable)
    for i in range(0,2):
        df_aggregated_h2.reset_index(level = 0, inplace = True)
              
#%%
    """ 4.2.1.2 Test assumption 3: Normality of Distribution --> Shapiro-Wilk Test """
    # We need to test each group separately.
    # If test is significant, distribution is not Gaussian.
    # If it's not significant, it couldn't be shown that it's not Gaussian (≠ it's Gaussian).
    
    # I want to collect my results in a df, so get info on the test, the data we tested and the results:
    gss_results_df = pd.DataFrame(columns = ["test name", "data", "p-values", "test statistics", 
                                             "df", "effect size", "bayes factor", "power"])
    # get unique values in sfc and feedback
    sfc_values = list(set(df_aggregated_h1["sfc"]))
    feedback_values = list(set(df_aggregated_h1["feedback"]))
    
    # get all possible combinations (cartesian product) of elements from the 2 lists
    pairs = list(product(feedback_values, sfc_values))
    
    # assume that all tests will be non-significant and we don't 
    # have to rank-transform our data later (change this if one of the tests is significant)
    # and we also don't have to apply a Greenhouse-Geisser Correction.
    rank_transform_h1 = False
    rank_transform_h2 = False
    GG_correction_h1 = False
    GG_correction_h2 = False
    
    # loop pairs values, test distribution and save test results
    for feedback, sfc in pairs:
        
        # get data 
        curr_data = df_aggregated_h1[(df_aggregated_h1["sfc"] == sfc) & 
                                     (df_aggregated_h1["feedback"] == feedback)]["power"]
        
        # save name of dataset
        data = "power in trials with feedback = " + feedback + " & sfc = " + str(sfc) 
        
        # if there are not enough values (aka less than 3), you can't run the Shapiro-Wilk test 
        # save None for p and stat instead:
        if len(curr_data) < 3:
            stat = None
            p = None
            rank_transform_h1 = True
        # if everything's fine and we have enough data:    
        else:
            # run Shapiro-Wilk test
            stat, p = shapiro(curr_data)
            
            # if test was significant and data are not normally distributed,
            # rank transform data before running the ANOVA
            if p <= 0.05:
                rank_transform_h1 = True
        
        # append resuts to df
        gss_results_df.loc[len(gss_results_df)] = ["Shapiro-Wilk Test for Normality of Distribution", 
                                                   data, 
                                                   p, stat, None,
                                                   None, None, None]
        
        
        # Now do it again but only group by feedback
        # each feedback condition occurs 3x in the pairs, so make sure 
        # you only compute this once:
        if feedback in feedback_values:
            
            # delete value from feedback_values so you don't run the following part 
            # for the current feedback condition again and again:
            feedback_values.remove(feedback)
            
            # get data 
            curr_data = df_aggregated_h2[df_aggregated_h2["feedback"] == feedback]["power"]
        
            # save name of dataset
            data = "power in trials with feedback = " + feedback
        
            # if there are not enough values (aka less than 3), you can't run the Shapiro-Wilk test 
            # save None for p and stat instead:
            if len(curr_data) < 3:
                stat = None
                p = None
                rank_transform_h2 = True
                # if everything's fine and we have enough data:    
            else:
                # run Shapiro-Wilk test
                stat, p = shapiro(curr_data)
        
                # if test was significant and data are not normally distributed,
                # rank transform data before running the ANOVA
                if p <= 0.05:
                    rank_transform_h2 = True
        
            # append resuts to df
            gss_results_df.loc[len(gss_results_df)] = ["Shapiro-Wilk Test for Normality of Distribution", 
                               data, 
                               p, stat, None,
                               None, None, None]
        
    # sort the df a bit
    gss_results_df = gss_results_df.sort_values("data")
         
#%%
    """ 4.2.1.3 Test assumption 5: Homogeneity of Variance --> Levene test"""

    # If the Shapiro-Wilk Test for a certain group was significant, 
    # this could mean that the distribution of the data in this group is Gaussian. 
    # If this is the case for all groups we want to test in a certain test, assumption #3 is given.

    # If none of the p-values of the Shapiro-Wilk tests were 
    # significant & all datasets had the necessary length to run tests,
    # test homogeneity of variance
    
    # check if this is the case for the data for h1 or h2
    if rank_transform_h1 == False:
        
        # collect datasets in a list
        tmp_list_dfs = []
    
        # loop pairs again
        for feedback, sfc in pairs:
            
            # get data 
            curr_data = df_aggregated_h1[(df_aggregated_h1["sfc"] == sfc) & 
                                         (df_aggregated_h1["feedback"] == feedback)]["power"]
            
            # append to list of dfs
            tmp_list_dfs.append(curr_data)
        
        """ Test assumption 5 - Homogeneity of Variance: """
        """ Levene Test """
        
        # run Leve test; hint: the asterix can be used to "unpack" objects, in this case I pass a 
        # list of arguments and unpack them so they're recognized as single arguments
        stat, p = levene(*tmp_list_dfs)
            
        # add to results df
        gss_results_df.loc[len(gss_results_df)] = ["Levene Test for Homogeneity of Variance", 
                                                   "all combinations of sfc x feedback", 
                                                   p, stat, None,
                                                   None, None, None]
        # if test was significant, rank transform data later on
        if p <= 0.05:
            rank_transform_h1 = True    
            
#%%      
        # If the Levene test was not significant, this means the 
        # variances of the groups were more or less equal. 
        # If this is the case, go on with testing the last 
        # assumtion (aka the ANOVA assumption): Sphericity  --> Mauchly's test 
        else: 
            # Spericity can't be computed for 2-way repeated-measures designs 
            # if neither of the independent variables have 
            # exactly 2 levels (we have 3 in both). More complex designs are 
            # neither supported by pingouin
            # So set GG correction to true in order not to get false positives
            GG_correction_h1 = True   
                 
#%%      
    # Do it all again for the h1 data (feedback)
    if rank_transform_h2 == False:
        # get feedback values again
        feedback_values = list(set(df_aggregated_h2["feedback"]))
        
        # collect datasets in a list
        tmp_list_dfs = []
    
        # loop feedback values
        for feedback in feedback_values:
            
            # get data 
            curr_data = df_aggregated_h2[df_aggregated_h2["feedback"] == feedback]["power"]
            
            # append to list of dfs
            tmp_list_dfs.append(curr_data)
            
            
        """ Test assumption 5 - Homogeneity of Variance: """
        """ Levene Test """
        # run Levene test, get p and test statistic   
    
        stat, p = levene(*tmp_list_dfs)
    
        # add to results df
        gss_results_df.loc[len(gss_results_df)] = ["Levene Test for Homogeneity of Variance", 
                                                   "all feedback levels", 
                                                   p, stat, None,
                                                   None, None, None]
        # if test was significant, rank transform data later on
        if p <= 0.05:
            rank_transform_h2 = True

#%%      
        # If the Levene test was not significant, this means the 
        # variances of the groups were more or less equal. 
        # If this is the case, go on with testing the last 
        # assumtion (aka the ANOVA assumption): Sphericity  

        else: 
            # Test ANOVA assumption: Sphericity: 
            """ 4.2.3 If the Levene test was not significant, run Mauchly’s Test """
            spher, stat, chi2, df, p = sphericity(df_aggregated_h2,
                                                  dv = "power", 
                                                  subject = "ID", 
                                                  within = "feedback")
      
            # append results to df
            gss_results_df.loc[len(gss_results_df)] = ["Mauchly's Test for Sphericity", 
                                                       "all feedback groups", 
                                                       p, stat, df,
                                                       None, None, None]
            # if sphericity is not given, apply Greenhouse Geisser Correction
            if p <= 0.05: 
                GG_correction_h2 = True
                
            
#%% 
    """ 4.2.2 If assumptions are violated, rank-transform data """
    
    # If not all of the p-values from the Shapiro-Wilk tests 
    # are > 0.05, at least one of the groups is not 
    # normally distributed, which means parametrical 
    # tests can't be used without rank-transforming the data first. 
    # Same applies if homogeneity of variance is not given.

    # I used this for testing the ANOVAs:
    #df_aggregated["power"] = [26.5, 5, 14, 35, 10.5, 23.5, 5, 18.5, 25, 4.8,
    #                          16, 35, 15, 25, 3, 19, 28, 29, 15, 31, 12, 22.5, 
    #                          4, 19, 29, 7, 15,32, 12.5, 23.5, 1, 17.5]
    
    # If assumptions are violated, rank transform data before using ANOVA & t-tests.
    # if you have ties (aka >= 2x the same value), assign average rank
    if rank_transform_h1:    
        df_aggregated_h1["power"] = rankdata(df_aggregated_h1["power"], method='average').tolist()
    
    if rank_transform_h2:    
        df_aggregated_h2["power"] = rankdata(df_aggregated_h2["power"], method='average').tolist()

#%% 
    """ 4.3 Run repeated measures ANOVAs """
    
    """ 4.3.1 ANOVA for Hypotheses 1.1 - 1.3: """
    # 2-way repeated measures ANOVA aka ANOVA for dependent groups with 2 independent variables
    
    anova_res_h1 = rm_anova(data =  df_aggregated_h1, 
                            dv = "power", 
                            within = ["sfc", "feedback"], 
                            subject = "ID",
                            correction = True,
                            effsize = "n2")         
    
    # save results:
    # get results for interaction effect sfc x feedback
    
    # if you don't have enough data / the data in all groups 
    # has range = 0 because you copy & pasted one test df like me
    # the ANOVA won't give you an F value. So it that's the case, 
    # save None instead
    try:
        stat = round(float(anova_res_h1["F"][2]), 3)
    except KeyError:
        stat = None
        
    df1 = float(anova_res_h1["ddof1"][2])
    df2 = float(anova_res_h1["ddof2"][2])
    eff_size = str(round(float(anova_res_h1["n2"][2]), 3)) + " (eta squared)"
        
    """ Mauchly's Test """ 
    # If the Mauchly Test was significant, apply GG correction
    # --> Problem: in this case, Mauchly's test can't be computed 
    # (at least the pingouin package says it can't) so use corrected p-value
    p = round(float(anova_res_h1["p-GG-corr"][2]), 3)

    # append results to df
    gss_results_df.loc[len(gss_results_df)] = ["one-way repeated measures ANOVA", 
                                               "all sfc x feedback groups", 
                                               p, stat, [df1, df2],
                                               eff_size, None, None]
                 
#%%  
    """ 4.3.2 ANOVA for Hypotheses 2.1 - 2.3: """
    # 1-way repeated measures ANOVA aka ANOVA for dependent groups with 1 independent variable
    
    anova_res_h2 = rm_anova(data =  df_aggregated_h2, 
                            dv = "power", 
                            within = ["feedback"], 
                            subject = "ID",
                            correction = True,
                            effsize = "n2")         
    
    # save results:
    
    # if you don't have enough data / the data in all groups 
    # has range = 0 because you copy & pasted one test df like me
    # the ANOVA won't give you an F value. So it that's the case, 
    # save None instead
    try:
        stat = round(float(anova_res_h2["F"]), 3)
    except KeyError:
        stat = None
        
    df1 = float(anova_res_h2["ddof1"])
    df2 = float(anova_res_h2["ddof2"])
    eff_size = str(round(float(anova_res_h2["n2"]), 3)) + " (eta squared)"
        
    if GG_correction_h2:    
        """ Mauchly's Test """ 
        # If Mauchly's Test was significant, apply GG correction
        # --> Problem: in this case, Mauchly's test can't be computed 
        # (at least the pingouin package says it can't) so use corrected p-value
        p = round(float(anova_res_h2["p-GG-corr"]))
    else:
        p = round(float(anova_res_h2["p-unc"]), 3)
        
    # append results to df
    gss_results_df.loc[len(gss_results_df)] = ["one-way repeated measures ANOVA", 
                                               "all feedback groups", 
                                               p, stat, [df1, df2],
                                               eff_size, None, None]
    

#%%            
    """ 4.3.4 if ANOVA is significant, run t-tests """

    """ 4.3.4.1 for Hypotheses 1.1 - 1.3 """
    
    # I assume more feedback (aka higher sfc) --> higher tremor amplitudes
    # --> this is what I'll test in the t-test (for each feedback condition): 
    #     0.25 > 0.2, 0.3 > 0.2 and 0.3 > 0.25

    feedback_values = list(set(df_aggregated_h1["feedback"]))
    pairs = [(0.25, 0.2), (0.3, 0.2), (0.3, 0.25)]
    
    for feedback in feedback_values:

        # get data
        feedback_df = df_aggregated_h1[df_aggregated_h1["feedback"] == feedback]
        
        for sfc_1, sfc_2 in pairs:   
           
            # get data
            X = np.array(feedback_df[feedback_df["sfc"] == sfc_1]["power"])
            Y = np.array(feedback_df[feedback_df["sfc"] == sfc_2]["power"])
            
            # check if both arrays have at least 2 elements
            if len(X) < 2 or len(Y) < 2:
                # go to the next pair if you can't run tests with these data
                continue
            else:
                # run one-sided t-test, assume x > y 
                res = ttest(x = X, 
                            y = Y, 
                            paired = True, 
                            alternative = "greater")
               
                # use Bonferroni correction for p-values:
                p = float(res["p-val"]) * (len(pairs) * len(feedback_values))
               
                stat = float(res["T"])
                df = float(res["dof"])
                eff_size = str(float(res["cohen-d"])) + " (Cohen's d)"
                bayes_factor = float(res["BF10"])
                power = float(res["power"])
                
                # create description of the data you tested
                data = "for feedback: " + feedback + ", compare power in sfc: " + str(sfc_1) + " > " + str(sfc_2)
               
                # append results to df
                gss_results_df.loc[len(gss_results_df)] = ["one-sided t-test", 
                                                           data,
                                                           p, stat, df,
                                                           eff_size, bayes_factor, power]
               
            
#%%    
    
    """ 4.3.4.2 for Hypotheses 2.1 - 2.3 """
    
    # 1. I assume ao should evoke at least the same tremor intensity as vo, 
    #    so I have to "test" the H0 I guess 
    # 2. I also assume multisensory feedback should lead to higher 
    #    tremor amplitudes than unisensory feedback (ao or vo)
    
    # --> this is what I'll test: 
    #     ao ≠ vo (--> 2-sided, alpha = 20%, hope for non-significant result).     
    pairs = [("ao", "vo"), ("va", "ao"), ("va", "vo")]
    alternative = ["two-sided", "greater", "greater"]
    test_name = ["two-sided t-test (alpha 20%)", "one-sided t-test", "one-sided t-test"]
    data = ["compare power in feedback: ao ≠ vo", 
            "compare power in feedback: va > ao",
            "compare power in feedback: va > vo"]
    
    # I need 1 value for each participant in each feedback condition 
    # so aggregate df again:
        
        
        
         
    # loop our planned tests:
    for idx in range(0, len(pairs)): 
        
        # get data for current pair we'd like to compare
        feedback1, feedback2 = pairs[idx]
        
        X = np.array(df_aggregated_h2[df_aggregated_h2["feedback"] == feedback1]["power"])
        Y = np.array(df_aggregated_h2[df_aggregated_h2["feedback"] == feedback2]["power"])
        
        # check if both arrays have at least 2 elements
        if len(X) < 2 or len(Y) < 2 or len(X) != len(Y):
            # go to the next pair if you can't run tests with these data
            continue
        else:
            # run one-sided t-test, assume x > y 
            res = ttest(x = X, 
                        y = Y, 
                        paired = True, 
                        alternative = alternative[idx])
           
            # use Bonferroni correction for p-values:
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

    """ 5. save dataframe with results as .csv """
    gss_results_df.to_csv(path_or_buf = working_directory + "gss_test_results.csv")

          
#%% 

    """ 6. Create "I'm done!"-message: """
    print("\n\n- - - - - - - - - - - - - - - - - - - - - \n\nHey girl, I saved the results for the\ngss data as 'gss_test_results.csv' in\nthe working directory. I also returned it\nas a new variable called 'gss_results_df'!\n\n- - - - - - - - - - - - - - - - - - - - - ")

          
#%% 

    """ 7. return gss_results_df as function output """
    return(gss_results_df)


# END OF FUNCTION    
