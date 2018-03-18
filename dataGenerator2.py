from __future__ import division
from __future__ import print_function
import numpy as np
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=True)
def _seed_numba(seed):
    np.random.seed(seed)

def seed(seed):
    np.random.seed(seed)
    _seed_numba(seed)

@jit(nopython=True)
def log_mean_prob(log_Prob):
    max_log_prob = np.max(log_Prob)
    return np.log(np.mean(np.exp(log_Prob - max_log_prob))) + max_log_prob

@jit(nopython=True)
def normalize_prob(log_Prob):
    Prob = np.exp(log_Prob - np.max(log_Prob))
    Prob /= np.sum(Prob)
    return Prob

@jit(nopython=True)
def residual_resample(Prob):
    M = Prob.size
    res_M = M
    res_Prob = Prob * M
    count = np.zeros(M, dtype=np.int64)
    for i in range(M):
        count[i] = int(res_Prob[i])
        res_Prob[i] -= count[i]
        res_M -= count[i]
    if res_M > 0:
        j = 0
        cum_prob = 0.0
        res_Prob /= res_M
        sampled_float = np.sort(np.random.rand(res_M))
        for i in range(M):
            cum_prob += res_Prob[i]
            while j < res_M and sampled_float[j] <= cum_prob:
                count[i] += 1
                j += 1
    j = 0
    sampled_int = np.arange(M)
    for i in range(M):
        if count[i] == 0:
            while count[j] <= 1: j += 1
            sampled_int[i] = j
            count[j] -= 1
    return sampled_int


@jit(nopython=True)
def generate_data(theta, T):
  """
  model:
  s_0 = N(0,5)
  s_{t+1} = s_t/2 + 25s_t/(1+s_t**2) + 8cos(1.2t) + N(0,theta[0]**2)
  o_{t+1} = s_{t+1}^2/2 + N(0,theta[1]**2)
  """
  s = np.zeros((T))
  o = np.zeros((T))
  s[0] = 0.0
  o[0] = s[0]**2/2 + np.random.normal()*theta[1]
  for t in range(1,T):
    s[t] = s[t-1]/2 + 25*s[t-1]/(1+s[t-1]**2) + 8*np.cos(1.2*t) + np.random.normal()*theta[0]
    o[t] = s[t]**2/2 + np.random.normal()*theta[1]
  return s, o


@jit(nopython=True)
def particlefilter(theta, o, M, lag):
    """
    input:
        o[0~T]
        M: number of particles
        lag: for pre-fit residual lag

    output: s_pre, s_post, y,
            where, 
                s_pre[t] = E[s_t|o_{1:t-1}]
                s_post[t] = E[s_t|o_{1~t}]
                y[t,k] = o_t - E[O_t|o_{1~t-k}], k=0,1,...,lag-1,lag
    """
    T = o.shape[0]
    s_pre = np.zeros((T,M))
    s_post = np.zeros((T,M))
    y = np.zeros((T,lag))
    W = np.zeros((T,M))

    # initilazation
    for i in range(M):
        s_pre[0,i] = 0.0
        W[0,i] = -np.log(theta[1])-((s_pre[0,i]**2/2-o[0])/theta[1])**2/2

    for t in range(1,T):
        
        # resample according to last step's weight
        probability = normalize_prob(W[t-1,:])
        sampled_index = residual_resample(probability)
        for j in range(M):
            s_post[t-1,j] = s_pre[t-1,sampled_index[j]]

        # transition & reweight
        for i in range(M):
            s_pre[t,i] = s_post[t-1,i]/2 + 25*s_post[t-1,i]/(1+s_post[t-1,i]**2) + \
                           8*np.cos(1.2*t) + np.random.normal() * theta[0]
            W[t,i] = -np.log(theta[1])-((s_pre[t,i]**2/2-o[t])/theta[1])**2/2

    y = np.zeros((T,lag+1)) #y[t,lag] = o_t - E[O_t|o_{1:t-lag}]
    for t in range(T-lag):
        tmp = s_post[t].copy()
        for i in range(1,lag+1):
            o_hat = 0.0
            for j in range(M):
                tmp[j] = tmp[j]/2 + 25*tmp[j]/(1+tmp[j]**2) + \
                           8*np.cos(1.2*(t+i)) + np.random.normal() * theta[0]
                o_hat += tmp[j]**2/2/M
            y[t+i,i] = o[t+i] - o_hat
        y[t,0] = o[t] - np.mean(s_post[t,:]**2/2)

    return s_pre, s_post, y

if __name__ == '__main__':
    
    seed(123)
    
    theta = np.array([1,1])
    T = 500
    M = 1000
    K = 10

    s, o = generate_data(theta,T)
    s_pre, s_post, y = particlefilter(theta, o, M, K)
    
    for i in range(0,10):
        print('average of (o_t-E[O_t|o_0:t-{}])^2'.format(i),np.mean(y[:,i]**2))

    plt.plot(y[:,0])
    plt.plot(y[:,1])
    plt.show()



  