import numpy as np
import numbers

class EMA():
    def __init__(self, initial_val, smoothing_level = 0.05):
        self.mean = initial_val
        self.smoothing_level = smoothing_level
        self.mean_list = [initial_val]
    
    def compute_EMA_and_add(self, newvalues):
        # computes EMA given new value and add data renew self.mean
        if (isinstance(newvalues, numbers.Number)): newvalues = [newvalues]
        newmeans = []
        for val in newvalues:
            self.mean = self.smoothing_level * val + (1-self.smoothing_level) * self.mean
            self.mean_list.append(self.mean)
            newmeans.append(self.mean)
        return newmeans

def tick_imbalance_bar(ts, init_num = 10, e0t_init = 10):
    """ Creates tick imbalance bars
     Parameters
    ----------
    ts : 1-dim ndarray of time series data
    init_num : #frames used to compute initial estimate of Pr[b_t = 1]
    e0t_init : initial E0[T]

    Returns
    -------
    tick_time : List of tick time
    T_list : List of T's
    p1_list : List of estimated Pr[b_t=1]
    """
    diff = np.diff(ts) 
    signs = np.sign(diff)
    mask = signs==0
    idx = np.where(~mask,np.arange(len(mask)),0)
    np.maximum.accumulate(idx, out=idx)
    signs = signs[idx] # filling signs forward

    # initialization
    tick_time = [init_num]
    T_list = [e0t_init]
    imbalance = 0
    e0t = e0t_init
    p1 = (np.mean(signs[:init_num] + 1))/2 # the estimate of Pr[b_t = 1]
    p1_list = [p1]
    threshold = np.abs(2*p1 - 1) * e0t
    ema_T = EMA(e0t)
    ema_p1 = EMA(p1)
    for i in range(init_num + 1,len(ts)):
        imbalance += signs[i-1]
        if np.abs(imbalance) >= threshold:
            # imbalance exceeds threshold, so store the time and reset
            duration = i-tick_time[-1]
            T_list.append(duration)
            tick_time.append(i)
            e0t = ema_T.compute_EMA_and_add(duration)[0]
            p1 = ema_p1.compute_EMA_and_add((signs[tick_time[-2]:tick_time[-1]]+1)/2)[-1]
            p1_list.append(p1)
            threshold = np.abs(2*p1 - 1) * e0t
            imbalance = 0
    return tick_time, T_list, p1_list
