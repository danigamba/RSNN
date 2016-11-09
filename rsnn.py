# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:16:01 2016
"""

import numpy as np
import matplotlib.pyplot as plt

class NNError(Exception):
    """Exception raised for errors in runtime.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class simplenn:

    def sigma(self, x):
        '''neuron function - sigmodial function'''
        return 1/(1+np.e**(-x))


    def costf(self, yhat):
        z = []
        for i in range(self.nTrain):
            z.append(float(self.yTrain[i]-yhat[i])**2)
        return (1/self.nTot)*sum(z)


    def __init__(self, x, y, n=1, m=1, p=0.8):
        '''
        Initialize the network

        Keyword arguments:

        x -- input (list)
        y -- output (list)
        n -- number of first-layer neuron
        m -- number of entry
        p -- fraction of training example used for training and not for validation
        '''
        self.m = m
        self.n = n
        self.p = p
        self.nTot=len(x)
        n_tmp = len(y)
        if self.nTot != n_tmp:
            raise NNError("InputError", "Dimension of x and y doesn't correspond")

        #Linear correlation with sigma_function
        #self.W = np.random.rand(n,self.m)
        self.W = np.random.rand()
        self.beta = np.random.rand()
        self.alpha = np.random.rand()
        self.gamma = np.random.rand()

        self.xTrain = []
        self.yTrain = []

        self.nTrain = int(np.floor(self.nTot*p))
        #sampling
        for i in range(self.nTrain):
            indx = int(np.floor(np.random.rand()*(self.nTot-i)))
            self.xTrain.append(x[indx])
            self.yTrain.append(y[indx])
            del x[indx]
            del y[indx]

        self.xValid = x
        self.yValid = y

        self.yOut = []


    def _compute(self, theta):
        '''
            y = alpha*sigma(Wx+beta)+gamma

            theta[0] -- alpha
            theta[1] -- beta
            theta[2] -- gamma
            theta[3] -- W
        '''
        yout = []
        for i in range(self.nTrain):
            inp = theta[3]*self.xTrain[i] + theta[1]
            zout = self.sigma(inp)
            yout.append(theta[0]*zout+theta[2])

        return yout
    
    def _gradient(self, thetamin):
        #the increment rate is randomized, this should decrease in time
        a = np.random.rand(4)
        costold = self.costf(self._compute(thetamin))
        theta = thetamin
        for i in range(4):
            theta[i] += a[i]
            cost = self.costf(self._compute(theta))
            print(cost)
            if((float(cost) - float(costold)) < 0):
                continue            
            else:
                theta[i] -= 2*a[i]
                cost = self.costf(self._compute(theta))
                if((float(cost) - float(costold)) < 0):
                    continue                
                else:
                    #dont change that param
                    theta[i] += a[i]                
        
        return theta
        


    def run(self):
        failcount = 0
        theta = [self.alpha, self.beta, self.gamma, self.W]
        yhat = self._compute(theta)
        costmin = self.costf(yhat)
        thetamin = theta
        print("Initial value: "+str(float(costmin)))
        while failcount < 20:
            #TODO: use validation data
            theta = self._gradient(thetamin)

            yhat = self._compute(theta)
            cost = self.costf(yhat)
            if float(cost) < float(costmin):
                print("New min: "+str(float(cost)))
                failcount = 0
                thetamin = theta
                costmin = cost
            else:
                print(" + "+str(float(cost)))
                failcount+=1

        self.alpha = thetamin[0]
        self.beta = thetamin[1]
        self.gamma = thetamin[2]
        self.w = thetamin[3]
        self.yOut = self._compute(thetamin)




if __name__ == '__main__':
    print("Executing main test script")

    # random shaped data from a sigmodial function
    tmp = np.random.rand()*10-5
    x = np.arange(tmp,tmp+3,0.05)
    m = len(x)
    y = 1/(1+np.e**(-x))
    y += np.random.rand(m)/10   #uniform random

    #plt.plot(y,x,'gx')

    nn = simplenn(x.tolist(), y.tolist())
    yTrain=nn.yTrain
    nn.run()

    plt.plot(y,x,'gx')
    plt.plot(nn.yOut, nn.xTrain, 'r*')
    #plt.legend("input data","nonlinear regression output")
