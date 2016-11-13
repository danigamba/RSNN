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
        ''' compute the cost J for a given yhat (mean square error) '''
        z = []
        for i in range(self.nTrain):
            z.append(float(self.yTrain[i]-yhat[i])**2)
        return (1/self.nTrain)*sum(z)


    def __init__(self, x, y, n=1, m=1, p=0.8):
        '''
        Initialize the network

        Keyword arguments:

        x -- input (list)
        y -- output (list)
        n -- number of first-layer neuron
        m -- number of features
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
        self.W = np.random.rand(n*m)
        self.beta = np.random.rand(n)
        self.alpha = np.random.rand(n)
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
            Compute the yOut
            y = alpha*sigma(Wx+beta)+gamma

            theta[0] -- alpha
            theta[1] -- beta
            theta[2] -- gamma
            theta[3] -- W
        '''
        yout = np.dot(self.sigma( np.outer(self.xTrain, theta[3]) + theta[1] ),theta[0])+theta[2]
        return yout
    
    def _gradient(self, thetamin, learning_rate=0.1):
        ''' compute the direction of every parameter of theta (not so good) '''
        #the increment rate is randomized, this should decrease in time
        #a = np.random.rand(4)
        costold = self.costf(self._compute(thetamin))
        theta = thetamin
        for i in range(4):
            if type(theta[i]) != float:
                for j in range(theta[i].shape[0]):
                    theta[i][j] += learning_rate
                    cost = self.costf(self._compute(theta))
                    #print(cost)
                    if((float(cost) - float(costold)) < 0):
                        continue            
                    else:
                        theta[i][j] -= 2*learning_rate
                        cost = self.costf(self._compute(theta))
                        if((float(cost) - float(costold)) < 0):
                            continue                
                        else:
                            #dont change that param
                            theta[i][j] += learning_rate                
            else:
                theta[i] += learning_rate
                cost = self.costf(self._compute(theta))
                #print(cost)
                if((float(cost) - float(costold)) < 0):
                    continue            
                else:
                    theta[i] -= 2*learning_rate
                    cost = self.costf(self._compute(theta))
                    if((float(cost) - float(costold)) < 0):
                        continue                
                    else:
                        #dont change that param
                        theta[i] += learning_rate 
        return theta
        

    def run(self):
        ''' execute the regression '''
        failcount = 0
        theta = [self.alpha, self.beta, self.gamma, self.W]
        yhat = self._compute(theta)
        costmin = self.costf(yhat)
        thetamin = theta
        incr = 1.0
        print("Initial value: "+str(float(costmin)))
        while failcount < 10 and incr > 0.00001:
            #TODO: use validation data
            theta = self._gradient(thetamin)

            yhat = self._compute(theta)
            cost = self.costf(yhat)
            if float(cost) < float(costmin):
                incr = float(costmin) - float(cost)
                #print("New min: "+str(float(cost)))
                failcount = 0
                thetamin = theta
                costmin = cost
            else:
                #print(" + "+str(float(cost)))
                failcount+=1
        
        self.alpha = thetamin[0]
        self.beta = thetamin[1]
        self.gamma = thetamin[2]
        self.w = thetamin[3]
        self.yOut = self._compute(thetamin)
        print("Cost min: ", costmin,"Theta: a=", self.alpha, "b=", self.beta, "W=", self.w, "g=", self.gamma)
        return costmin


if __name__ == '__main__':
    print("Executing main test script")

    # random shaped data from a sigmodial function
    tmp = np.random.rand()*10-5
    x = np.arange(tmp,tmp+7,0.01)
    m = len(x)
    #y = 1/(1+np.e**(-x))    
    y = np.sin(x)    
    #y += np.random.rand(m)/2   #uniform random
    y += np.random.normal(loc=0.0, scale=1.0, size=m)/2
    
    plt.figure()
    plt.plot(x,y,'rx', label='Input data')
    
    epoch = 5
    costmin = float("inf")
    for i in range(epoch):
        nn = simplenn(x.tolist(), y.tolist(), n=3)
        cost = nn.run()
        yModel = np.dot(nn.sigma( np.outer(x , nn.w) + nn.beta ),nn.alpha)+nn.gamma    
        plt.plot(x,yModel,'-', label='Model of the '+str(i+1)+"* iteration")
        if cost < costmin:
            costmin = cost
            best = i
            theta = [nn.alpha, nn.beta, nn.gamma, nn.w]
            
    plt.legend()
    print("The best model is the "+str(best+1)+"* iteration")
    print("Theta: "+str(theta))