""" 
Deep Particle Filter Network
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import random
import sys
sys.path.append('..')
import time
import numpy as np
import matplotlib.pyplot as plt

import dataGenerator1 as sim
import util
import tensorflow as tf

class DPFN(object):
    def __init__(self,M,dim_s,dim_o,dim_h,T,K,lr,pre_weight):
        self.o_data = None
        self.s_data = np.zeros((T,M,dim_s))
        self.pre_weight = pre_weight

        self.dim_s = dim_s
        self.dim_o = dim_o
        self.dim_h = dim_h
        self.M = M
        self.T = T
        self.K = K
        self.lr = lr
        
        with tf.variable_scope('DFPN', reuse=tf.AUTO_REUSE):

            self.W0 = tf.get_variable('W0', (self.dim_s, self.dim_s),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b0 = tf.get_variable('b0', (self.dim_s,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            self.W1 = tf.get_variable('W1', (self.dim_s+self.dim_s, self.dim_s),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b1 = tf.get_variable('b1', (self.dim_s,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)
            self.W2 = tf.get_variable('W2', (self.dim_s, self.dim_o),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b2 = tf.get_variable('b2', (self.dim_o,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            self.Wo = tf.get_variable('Wo', (self.dim_o, self.dim_h),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.bo = tf.get_variable('bo', (self.dim_h,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            self.Ws = tf.get_variable('Ws', (self.dim_s, self.dim_h),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.bs = tf.get_variable('bs', (self.dim_h,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            self.W3 = tf.get_variable('W3', (2*self.dim_h, self.dim_h),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b3 = tf.get_variable('b3', (self.dim_h,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            self.W4 = tf.get_variable('W4', (self.dim_h, 1),
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            self.b4 = tf.get_variable('b4', (1,), 
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def get_o_data(self,T):
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

        _,self.o_data = sim.dataGenerate(F,H,Q,R,c1,c2,T)

    def buildPrior(self):
        """
        z: tensor with size M x dim_s
        s_0: tensor with size M x dim_s
        """
        z = tf.random_normal( [self.M,self.dim_s], dtype=tf.float64)
        s_0 = tf.matmul(z,self.W0) + self.b0
        return s_0

    def buildT(self,s):
        """
        s: tensor with size M x dim_s
        s_new:  tensor with size M x dim_s
        """
        z = tf.random_normal( [self.M,self.dim_s], dtype=tf.float64)
        s_new = tf.matmul(tf.concat((s,z),axis=1),self.W1) + self.b1
        return s_new

    def buildO(self,snew,o_t):
        """
        snew: tensor with size M x dim_s
        o_t: tensor with size dim_o
        W: a tensor with size M x 1
        """
        
        o_t_M = tf.reshape(tf.tile(o_t,[self.M]),[self.M,-1])

        o1 = tf.nn.leaky_relu(tf.matmul(o_t_M,self.Wo)+self.bo)
        s1 = tf.nn.leaky_relu(tf.matmul(snew,self.Ws)+self.bs)

        z = tf.concat((s1, o1),axis=1)
        z = tf.nn.leaky_relu(tf.matmul(z, self.W3) + self.b3)
        W = tf.nn.leaky_relu(tf.matmul(z, self.W4) + self.b4)

        return W

    def buildF(self,s):
        """
        s: tensor with size M x dim_s
        ohat: tensor with size M x dim_o
        """
        ohat = tf.matmul(s, self.W2) + self.b2
        return ohat

    def buildForwardPropagate(self):
        """
        build placeholder:
            (1) self.s_old: size M x dim_s
            (2) self.o_t: size dim_o
        build tensor
            (1) self.s_new: size M x dim_s
            (2) self.s_new_w: size M
            (3) self.o_forecast: a list of K tensor size dim_o
        """
        self.s_0 = self.buildPrior()
        self.s_old = tf.placeholder(tf.float64, [self.M,self.dim_s],name='s_old')
        self.o_t = tf.placeholder(tf.float64, [self.dim_o],name='o_t')

        self.s_new = self.buildT(self.s_old)
        W = self.buildO(self.s_new, self.o_t)
        W = tf.exp(W - tf.reduce_max(W))
        self.s_new_w = W / tf.reduce_sum(W)

        o_hat = self.buildF(self.s_new)
        self.o_forecast = [tf.reduce_sum( o_hat*self.s_new_w, axis=0 )]
        self.o_forecast.append(tf.reduce_mean(o_hat, axis=0))

        s_old = self.s_new
        for t in range(self.K):
            s_new = self.buildT(s_old)
            self.o_forecast.append(  tf.reduce_sum( self.buildF(s_new)*self.s_new_w, axis=0 )  )
            s_old = s_new


    def ForwardPropagate(self):
        """
            This function will
            (1) fetch observation data from self.o_data (size T x dim_o)
            (2) perform forward propagate
            (3) save results in self.s_data (size T x M x dim_s); also calculate error and save in error (size T x (K+1))
        """
        feed = {  self.o_t   : self.o_data[0],
                  self.s_old : self.sess.run(self.s_0),
                }
        
        error = np.zeros((self.T,self.K+1))

        s_pre, prob, o_forecast=self.sess.run([self.s_new,self.s_new_w,self.o_forecast], feed)
        
        error[0,0] = np.sum((self.o_data[0] - np.array(o_forecast[0]))**2)
        for i in range(1,self.K+1):
            error[0,i]  = np.sum((self.o_data[i-1] - np.array(o_forecast[i]))**2)

        
        util.resample(self.s_data[0,:],s_pre,prob[:,0])

        for t in range(1,self.T-self.K):
            feed={  self.o_t    : self.o_data[t],
                    self.s_old  : self.s_data[t-1,:,:],
                    }
            s_pre, prob, o_forecast = self.sess.run([self.s_new,self.s_new_w,self.o_forecast],feed)
            
            error[t,0] = np.sum((self.o_data[t] - np.array(o_forecast[0]))**2)
            for i in range(1,self.K+1):
                error[t,i]  = np.sum((self.o_data[t+i-1] - np.array(o_forecast[i]))**2)
            
            util.resample(self.s_data[t,:],s_pre,prob[:,0])

        return error


    def buildBackPropagate(self):
        """
        build placeholder:
            (1) self.o_seq: observation sequence for time self.o_data, with size K x dim_o
        build tensor:
            (1) self.o_pre: a list, forecast one step further pre fit value for self.o_seq
            (2) self.o_post: a list, post_fit value for self.o_seq
            (3) self.loss_pre:  MSE for sum((self.o_seq[i]-self.o_pre[i])**2)
            (4) self.loss_post:  MSE for sum((self.o_seq[i]-self.o_post[i])**2)
        """

        self.o_pre = []
        self.o_post = []
        self.loss_pre = []
        self.loss_post = []

        # self.opt1 = []

        self.o_seq = tf.placeholder(tf.float64, [self.K,self.dim_o],name='o_seq')

        
        W = tf.constant(0.0,shape=(self.M,1),dtype=tf.float64)
        s_t = self.buildT(self.buildPrior())
        o_t_hat = self.buildF(s_t)
        self.o_pre.append(  tf.reduce_mean( o_t_hat, axis=0) )
        self.loss_pre.append(  tf.reduce_sum(tf.square(self.o_pre[-1]-self.o_seq[0])) )

        self.totalLoss = self.loss_pre[-1]
        for t in range(self.K-1):

            s_t1 = self.buildT(s_t)
            o_t1_hat = self.buildF(s_t1)

            W += self.buildO(s_t, self.o_seq[t])
            w = tf.exp(W - tf.reduce_max(W))
            weight = w / tf.reduce_sum(w)

            self.o_pre.append( tf.reduce_sum(weight*o_t1_hat,axis=0) )
            self.loss_pre.append(  tf.reduce_sum(tf.square(self.o_pre[-1]-self.o_seq[t+1])) )

            self.o_post.append( tf.reduce_sum(weight*o_t_hat,axis=0) )
            self.loss_post.append(  tf.reduce_sum(tf.square(self.o_post[-1]-self.o_seq[t])) )

            self.totalLoss +=  (self.loss_post[-1] + self.pre_weight*self.loss_pre[-1])#/(t+1)
            # self.opt1.append( tf.train.AdamOptimizer(self.lr).minimize(self.totalLoss) )

            s_t = s_t1
            o_t_hat = o_t1_hat

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.totalLoss)
        # self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.totalLoss)

    def run(self,train,gap,load_name,save_name):

        import time
        if train:
            self.buildBackPropagate()
        else:
            self.buildForwardPropagate()
        
        self.sess.run(tf.global_variables_initializer())
        if load_name:
            self.saver.restore(self.sess, load_name)

        for _ in range(1000000):

            self.get_o_data(self.T)

            if train:
                if _%gap==0:
                    if _>0:
                        print(_,total_loss)
                        print(pre_error)
                        print(post_error)
                        print('------train------')
                    pre_error = np.zeros(self.K)
                    post_error = np.zeros(self.K-1)
                    total_loss = 0
                    if save_name:
                        self.saver.save(self.sess, save_name)
                feed = {  self.o_seq: self.o_data  }
                res = self.sess.run([self.opt,self.totalLoss,self.loss_post,self.loss_pre],feed)
                pre_error += np.array(res[3]).flatten()/gap
                post_error += np.array(res[2]).flatten()/gap
                total_loss += float(res[1])/gap/self.K
            else:
                if _%gap==0:
                    if _>0:
                        print(error)
                        for i in range(self.K+1):
                            print(i,np.mean(error[:-self.K,i]),np.std(error[:-self.K,i])/np.sqrt(gap))
                        print('-------predict---------')
                    error = np.zeros((self.T,self.K+1))
                eval_error = self.ForwardPropagate()
                error += np.array(eval_error)/gap

if __name__ == '__main__':

    M = 10000
    dim_s = 3
    dim_o = 3
    dim_h = 10
    
    T = 100
    K = 8
    lr = 0.0001
    train = False
    gap = 100
    pre_weight=0.5
    load_name='./model3test.ckpt'
    save_name='./model3test.ckpt'
    # load_name=None

    # T = 12
    # K = 12
    # lr = 0.0001
    # train = True
    # gap = 1000
    # pre_weight=0.5
    # load_name='./model3test.ckpt'
    # save_name='./model3test.ckpt'
    # # load_name=None

    a=DPFN(M,dim_s,dim_o,dim_h,T,K,lr,pre_weight)
    a.run(train=train,gap=gap,load_name=load_name,save_name=save_name)

    
    