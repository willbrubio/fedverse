"""
EXTRACTED FROM UPSTREAM — re-validate against the original before editing logic.

Package : fed3bandit 0.0.5
Install : fed3bandit==0.0.5
Requested: ['binned_paction', 'true_probs', 'count_pellets', 'count_pokes', 'pokes_per_pellet', 'reversal_peh']
Helpers pulled in: ['filter_data', 'count_left_pokes', 'count_right_pokes']
Constants pulled in: []
Sibling imports: {}
Extracted: 2026-06-30
"""


import copy
import pandas as pd
import numpy as np
import statsmodels.api as sm


def filter_data(data_choices, skip=[]):
    """Filters the data to only show pokes "Left" or "Right" events, which are pokes that did not occur 
    during time out, or during pellet dispensing.
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    filtered_data : pd.DataFrame
        Filtered fed3 data file
    """
    
    event_types = ["Pellet", "LeftinTimeOut", "RightinTimeout", "LeftDuringDispense", "RightDuringDispense", "LeftWithPellet", "RightWithPellet", "LeftShort", "RightShort"]
    if len(skip) != 0:
        for event_type in skip:
            event_types.remove(event_type)

    filtered_data = copy.deepcopy(data_choices)
    for event_type in event_types:
        filtered_data = filtered_data[filtered_data["Event"] != event_type]
    
    filtered_data.iloc[:,0] = pd.to_datetime(filtered_data.iloc[:,0])
    
    return filtered_data.reset_index(drop=True)


def binned_paction(data_choices, window=5):
    """Bins actions from fed3 bandit file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
    window : int
        Sliding window by which the probability of choosing left will be calculated

    Returns
    --------
    p_left : pandas.Series
        Probability of choosing left. Returns pandas.Series of length data_choices.shape[0] - window
    
    """
    f_data_choices = filter_data(data_choices)
    actions = f_data_choices["Event"]
    p_left = []
    for i in range(len(actions)-window):
        c_slice = actions[i:i+window]
        n_left = 0
        for action in c_slice:
            if action == "Left":
                n_left += 1
            
        c_p_left = n_left / window
        p_left.append(c_p_left)
        
    return p_left


def true_probs(data_choices, offset=5, alt_left="Session_type", alt_right="Device_Number"):
    """Extracts true reward probabilities from Fed3bandit file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
    offset : int
        Event number in which the extraction will start

    Returns
    --------
    left_probs : pandas.Series
        True reward probabilities of left port

    right_probs : pandas.Series
        True reward probabilities of right port
    """

    f_data_choices = filter_data(data_choices)
    try:
        left_probs = f_data_choices["Prob_left"].iloc[offset:] / 100
        right_probs = f_data_choices["Prob_right"].iloc[offset:] / 100
    except:
        left_probs = f_data_choices[alt_left].iloc[offset:] / 100
        right_probs = f_data_choices[alt_right].iloc[offset:] / 100

    return left_probs, right_probs


def count_pellets(data_choices):
    """Counts the number of pellets in fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    c_pellets : int
        total number of pellets
    """

    f_data_choices = filter_data(data_choices)
    pellet_count = f_data_choices["Pellet_Count"]
      
    c_diff = np.diff(pellet_count)
    c_diff2 = np.where(c_diff < 0, 1, c_diff)
    c_pellets = int(c_diff2.sum())
    
    return c_pellets


def count_left_pokes(data_choices):
    """Counts the number of left pokes in fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
        
    Returns
    --------
    all_pokes : int
        total number of left pokes
    """

    f_data_choices = filter_data(data_choices)
    left_count = f_data_choices["Left_Poke_Count"]
    c_left_diff = np.diff(left_count)
    c_left_diff2 = np.where(np.logical_or(c_left_diff < 0, c_left_diff > 1), 1, c_left_diff)

    all_left_pokes = c_left_diff2.sum()

    return all_left_pokes


def count_right_pokes(data_choices):
    """Counts the number of right pokes in fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
        
    Returns
    --------
    all_right_pokes : int
        total number of right pokes
    """

    f_data_choices = filter_data(data_choices)
    right_count = f_data_choices["Right_Poke_Count"]
    c_right_diff = np.diff(right_count)
    c_right_diff2 = np.where(np.logical_or(c_right_diff < 0, c_right_diff > 1), 1, c_right_diff)

    all_right_pokes = c_right_diff2.sum()

    return all_right_pokes


def count_pokes(data_choices):
    """Counts the number of pokes in fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
        
    Returns
    --------
    all_pokes : int
        total number of pellets
    """

    all_left_pokes = count_left_pokes(data_choices)
    all_right_pokes = count_right_pokes(data_choices)
    all_pokes = all_left_pokes + all_right_pokes

    return all_pokes


def pokes_per_pellet(data_choices):
    """Calculates pokes per pellets from fed3 bandit file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file

    Returns
    --------
    ppp : int
        pokes per pellets
    """

    f_data_choices = filter_data(data_choices)
    pellets = count_pellets(f_data_choices)
    pokes = count_pokes(f_data_choices)
    
    if (pellets == 0) | (pokes == 0):
        ppp = np.nan
    else:
        ppp = pokes/pellets
    
    return ppp


def reversal_peh(data_choices, min_max, return_avg = False, alt_right = "Device_Number"):
    """Calculates the probability of poking in the high probability port around contingency switches
    from fed3 data file
    
    Parameters
    ----------
    data_choices : pandas.DataFrame
        The fed3 data file
    min_max : tuple
        Event window around the switches to analyze. E.g. min_max = (-10,11) will use a time window
        of 10 events before the switch to 10 events after the switch.
    return_avg : bool
        If True, returns only the average trace. If False, returns all the trials.

    Returns
    --------
    c_days : int
        days in data file
    c_pellets : int
        total number of pellets
    c_ppd : int
        average pellets per day
    
    """
    f_data_choices = filter_data(data_choices)
    try:
        prob_right = f_data_choices["Prob_right"]
    except:
        prob_right = f_data_choices[alt_right]

    event = f_data_choices["Event"]
    switches = np.where(np.diff(prob_right) != 0)[0] + 1
    switches = switches[np.logical_and(switches+min_max[0] > 0, switches+min_max[1] < f_data_choices.shape[0])]

    all_trials = []
    for switch in switches:
        c_trial = np.zeros(np.abs(min_max[0])+min_max[1])
        counter = 0
        for i in range(min_max[0],min_max[1]):
            c_choice = event.iloc[switch+i]
            c_prob_right = prob_right.iloc[switch+i]
            if c_prob_right < 50:
                c_high = "Left"
            elif c_prob_right > 50:
                c_high = "Right"
            else:
                print("Error")
                
            if c_choice == c_high:
                c_trial[counter] += 1
                
            counter += 1
        
        all_trials.append(c_trial)

    aall_trials = np.vstack(all_trials)
    
    if return_avg:
        avg_trial = aall_trials.mean(axis=0)
        return avg_trial
        
    else:
        return aall_trials
