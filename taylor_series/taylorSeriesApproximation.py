# Script to generate polynomial approximations to the sine and exp functions

import numpy as np
import math
import matplotlib
from matplotlib import pyplot as mplot

def TS_Prediction_exp(x0,x,numTerms):
    # Function to generate polynomial approximations for the exp function about x0
    prediction = np.exp(x0)*np.ones(x.shape)
    displacement = 1
    for numTerms in range(2,numTerms+1):
        der = np.exp(x0)            
        prediction = prediction + der*np.multiply(displacement,x-x0)/(math.factorial(numTerms-1))
        displacement = np.multiply(displacement,x-x0)
    
    return prediction

def TS_Prediction_sin(x0,x,numTerms):
    # Function to generate polynomial approximations for the sine function about x0
    prediction = np.sin(x0)*np.ones(x.shape)
    displacement = 1
    for numTerms in range(2,numTerms+1):
        if numTerms%2==0:
            der = np.cos(x0)
            if numTerms%4==0:
                der = -1*der
        else:
            der = np.sin(x0)
            if (numTerms-1)%4!=0:
                der = -1*der
            
        prediction = prediction + der*np.multiply(displacement,x-x0)/(math.factorial(numTerms-1))
        displacement = np.multiply(displacement,x-x0)
    
    return prediction

def taylorSeriesCheck(func):
    # Function that allows visual inspection of accuracy of Taylor series approximation to an exact function

    if func=='sin':
        xlim1 = -2*np.pi
        xlim2 = 2*np.pi
    else:
        xlim1 = -2
        xlim2 = 2

    # Point about which the expansion is made    
    anchorPoint = 0

    x = np.linspace(xlim1,xlim2,100)

    # Some hardwired values for visualization purposes
    if func=='sin':
        y_exact = np.sin(x)
        startTerm = 2
        jump = 2
        maxTerms = 10
    else:
        y_exact = np.exp(x)
        startTerm = 1
        jump = 1
        maxTerms= 6

    # The exact curve
    fig, ax = mplot.subplots()
    ax.plot(x,y_exact,'k-', label='Exact Curve',linewidth=3)

    # Polynomial approximations to the exact curve
    for numTerms in range(startTerm,maxTerms-startTerm+1,jump):
        if func=='sin':
            y = TS_Prediction_sin(anchorPoint,x,numTerms)
        else:
            y = TS_Prediction_exp(anchorPoint,x,numTerms)
        ax.plot(x,y,'--',linewidth=3,label='Num terms = %s' %(numTerms))
           
    mplot.title('Taylor Series Approximations', fontsize=24)
    mplot.xlim(xlim1,xlim2)
    
    if func=='sin':
        mplot.ylim(-1.5,1.5)
    else:
        mplot.ylim(-0.5,5)
    mplot.xticks(fontsize=14)
    mplot.yticks(fontsize=14)
    
    mplot.grid()
    mplot.ylabel('Y', fontsize=18, loc='center',rotation=90)
    mplot.xlabel('X', fontsize=18)

    mplot.legend(prop={'size': 14})
        
    mplot.show(block=True)

if __name__ == '__main__':
    #taylorSeriesCheck('sin')
    taylorSeriesCheck('exp')
