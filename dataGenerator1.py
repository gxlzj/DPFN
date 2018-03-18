from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

"""
s[0] ~ N(c1,Q)
s[t] ~ F.dot(s[t-1]) + N(c1,Q)
o[t] ~ H.dot(s[ t ]) + N(c2,R)
"""

def dataGenerate(F,H,Q,R,c1,c2,T):
    """
    Assume initial state s[0] ~ N(c1,Q), o[0] ~ H*s[0] + c2 + noise
    Return s[0~T] and o[0~T]
    """
    s = np.zeros((T,c1.size),dtype=np.float64)
    o = np.zeros((T,c2.size),dtype=np.float64)

    s[0] = np.random.multivariate_normal( np.linalg.inv(np.eye(F.shape[0])-F).dot(c1), Q)
    o[0] = np.random.multivariate_normal(H.dot(s[0])+c2,R)
    for t in range(1,T):
        s[t] = F.dot(s[t-1]) + np.random.multivariate_normal(c1,Q)
        o[t] = H.dot(s[t]) + np.random.multivariate_normal(c2,R)
    return s,o

from numba import jit
@jit
def kalmanFilter(F,H,Q,R,c1,c2,o,lag=10):
    """
    input o[0~T]
    output: s_pre, s_post, y, see definition below
            where, 
                s_pre[t] = E[s_t|o_{1:t-1}]
                s_post[t] = E[s_t|o_{1~t}]
                y[t,k] = o_t - E[O_t|o_{1:t-k}], k=0~lag
    """
    dim_s = c1.size
    dim_o = c2.size
    T = o.shape[0]

    I = np.eye(dim_s)
    s_pre = np.zeros((T,dim_s)) #s_pre[t] = E[s_t|o_{1:t-1}]
    s_post = np.zeros((T,dim_s))#s_post[t] = E[s_t|o_{1~t}]
    y_pre = np.zeros((T,dim_o)) #y_pre[t] = o_t - (H*s_pre[t]+c2)
    y_post = np.zeros((T,dim_o))#y_post[t] = o_t - (H*s_post[t]+c2)

    s_pre[0] = np.linalg.inv(np.eye(F.shape[0])-F).dot(c1)
    P_pre = Q

    y = np.zeros((T,lag+1)) #y[t,lag] = np.sum(o_t-E[O_t|o_{1:t-lag}])**2

    for t in range(o.shape[0]):
        y_pre[t] = o[t] - (H.dot(s_pre[t])+c2)
        S = R + H.dot(P_pre).dot(H.T)
        K = P_pre.dot(H).dot(np.linalg.inv(S))
        s_post[t] = s_pre[t] + K.dot(y_pre[t])
        tmp = I-K.dot(H)
        P_post = tmp.dot(P_pre).dot(tmp.T) + K.dot(R).dot(K.T)
        y_post[t] = o[t] - (H.dot(s_post[t])+c2)
        y[t,0] = np.sum(y_post[t]**2)
        if t<o.shape[0]-1:
            s_pre[t+1] = F.dot(s_post[t]) + c1
            P_pre = F.dot(P_post).dot(F.T) + Q

    
    for t in range(T-lag):
        tmp = s_post[t]
        for i in range(1,lag+1):
            tmp = F.dot(tmp)+c1
            y[t,i] = np.sum((o[t+i] - (H.dot(tmp)+c2))**2)

    return s_pre, s_post, y
    
if __name__ == '__main__':

    # np.random.seed(1234)

    T = 100
    K = 8

    # F = np.array([[0.9]])
    # c1 = np.array([1.0])
    # c2 = np.array([0])
    # H = np.array([[1.0]])
    # Q = np.array([[1.0]])
    # R = np.array([[1.0]])

    # F = np.array([[-0.9,0.0],[0.0,-0.9]])
    # c1 = np.array([1.0,1.0])
    # c2 = np.array([0,0])
    # H = np.array([[1.0,0.0],[0.0,1.0]])
    # Q = np.array([[0.1,0.0],[0.0,0.1]])
    # R = np.array([[1.0,0.0],[0.0,1.0]])

    F = np.array([[-0.9,0,0],[0,-0.9,0],[0,0,-0.9]])
    c1 = np.array([1,1,1])
    c2 = np.array([0,0,0])
    H = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Q = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])

    error = np.zeros((T,K+1))
    num_sim = 1000
    for _ in range(num_sim):
        s,o = dataGenerate(F,H,Q,R,c1,c2,T)
        s_pre, s_post, y = kalmanFilter(F,H,Q,R,c1,c2,o,K)
        for i in range(K+1):
            error[:,i] += y[:,i]/num_sim
    for i in range(K+1):
        print(i,np.mean(error[:-K,i]),np.std(error[:-K,i])/np.sqrt(num_sim),sep='\t')
    

    # for i in range(0,K):
        # print("average of (o_t-E[O_t|o_0:t-{}])^2 ".format(i),np.mean(y[K+1:,i]**2))

    # s,o = dataGenerate(F,H,Q,R,c1,c2,T)        
    # plt.plot(s[:,0])
    # # plt.plot(s_post[:,0])
    # plt.plot(o[:,0])
    # plt.show()
