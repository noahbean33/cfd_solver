# Script to compute the numerical solution of the 1D (uni-directional) wave equation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def analyticalFunction(x,x_ref,waveLength,typeOfFunction):
    # Function to create a bunch of initial conditions
    if typeOfFunction == 'exponential':
        y = np.exp(-200 * (x-x_ref)**2)
    else:
        if typeOfFunction == 'single sinusoid':
            y = np.sin(2*np.pi*(x-x_ref)/waveLength)
        elif typeOfFunction == 'multiple sinusoids':
            y = (1/3)*(np.sin(2*np.pi*(x-x_ref)/waveLength) + np.sin(4*np.pi*(x-x_ref)/waveLength) + np.sin(8*np.pi*(x-x_ref)/waveLength))
        elif typeOfFunction == 'step function':
            y = np.ones(x.shape)
        elif typeOfFunction == 'polynomial':
            y = 0.5 + (x-x_ref) + (x-x_ref)*(x-x_ref) + (x-x_ref)*(x-x_ref)*(x-x_ref)
        else:
            raise Exception("Function type not recognized")

        # Apply a mask
        lessThanZeroIndices = np.where(x < x_ref)
        greaterThanWavelengthIndices = np.where(x>x_ref+waveLength)
        y[lessThanZeroIndices] = 0
        y[greaterThanWavelengthIndices] = 0        

    return y

# Computational parameters
dx = 0.004
dt = 0.002
numSteps = 101
c = 1.0
waveLength = 0.4

# Save options (for gif creation)
saveFile = 0
if saveFile==1:
    pauseCount = 10
else:
    pauseCount = max(numSteps-1,1)

x = np.arange(-0.5, 1.5, dx)
x_ref_init = 0
functionType = 'step function'#'multiple sinusoids'#'exponential'
y_init = analyticalFunction(x,x_ref_init,waveLength,functionType)

y_MacCormack = np.copy(y_init)
y_FTBS = np.copy(y_init)
y_analytical = np.copy(y_init)

for i in range(0,numSteps):
    if i > 0:

        # MacCormack
        yOld_MacCormack = np.copy(y_MacCormack) 
        y_pred = np.copy(y_MacCormack)
        y_pred[1:] = yOld_MacCormack[1:] - c*(dt/dx)*(yOld_MacCormack[1:]-yOld_MacCormack[0:-1])
        y_MacCormack[1:-1] = yOld_MacCormack[1:-1] - 0.5*c*dt*( (yOld_MacCormack[1:-1]-yOld_MacCormack[0:-2])/dx + (y_pred[2:]-y_pred[1:-1])/dx )
        
        # FTBS
        yOld_FTBS = np.copy(y_FTBS)        
        y_FTBS[1:] = yOld_FTBS[1:] - c*(dt/dx)*(yOld_FTBS[1:]-yOld_FTBS[0:-1])

        # Analytical
        y_analytical = analyticalFunction(x,x_ref_init+i*c*dt,waveLength,functionType)

        
    if i%pauseCount == 0:

        if i<10:
            padding = '000'
        elif i>=10 and i<100:
            padding = '00'
        elif i>=100 and i<1000:
            padding = '0'
        else:
            padding=''

        fig, ax = plt.subplots()
        ax.plot(x, y_MacCormack,'r-',label='MacCormack')
        ax.plot(x, y_FTBS,'g-',label='FTBS')
        ax.plot(x, y_analytical,'k-',label='Analytical')
        ax.legend()

        if functionType == 'single sinusoid' or functionType == 'multiple sinusoids':
            plt.ylim(-1.2,1.2)
        else:
            plt.ylim(-0.2,1.2)
        plt.xlim(-0.5,1.5)
        fileName = 'images/waveEquation/wave_'+padding+str(i)
        plt.suptitle("Wave Equation",fontsize=20)
        if saveFile == 1:
            plt.savefig(fileName,dpi=200)

        # Keep last plot open
        if i < numSteps-1:
            plt.close(fig)

plt.show()
