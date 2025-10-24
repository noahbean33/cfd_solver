# Script to compute the numerical solution of streamfunction-vorticity equations applied to a lid-driven cavity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Geometric params
Lx = 1
Ly = 1
nx = 256
ny = 256
dx = Lx/(nx-1)
dy = Ly/(ny-1)

X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))

# Timestep size and number
dt = 0.0003
numSteps = 20001

# Problem-specific params
reynoldsNumber = 1000
nu = (1/reynoldsNumber)
Uwall = 1

saveFile = 0
if saveFile==1:
    pauseCount = 100
else:
    pauseCount = max(numSteps-1,1)

psi = np.zeros([ny,nx])
omega = np.zeros([ny,ny])
U = np.zeros([ny,nx])
V = np.zeros([ny,nx])

U[ny-1,0:] = 1

for i in range(0,numSteps):

    omega_old = np.copy(omega)
    
    # Set Boundary Conditions
    omega[ny-1,1:-1] = (2/dx/dx)*(psi[ny-1,1:-1]-psi[ny-2,1:-1])-(2/dx)*Uwall
    omega[0,1:-1] = (2/dx/dx)*(psi[0,1:-1]-psi[1,1:-1])
    omega[1:-1,0] = (2/dy/dy)*(psi[1:-1,0]-psi[1:-1,1])
    omega[1:-1,nx-1] = (2/dy/dy)*(psi[1:-1,nx-1]-psi[1:-1,nx-2])

    # Update Omega
    omega[1:-1,1:-1] = omega[1:-1,1:-1] + 0.25*dt/(dx*dy)*(-(psi[2:,1:-1]-psi[0:-2,1:-1])*(omega_old[1:-1,2:]-omega_old[1:-1,0:-2]) + (psi[1:-1,2:]-psi[1:-1,0:-2])*(omega_old[2:,1:-1]-omega_old[0:-2,1:-1])) +dt*nu*( (omega_old[1:-1,2:]-2*omega_old[1:-1,1:-1]+omega_old[1:-1,0:-2])/dx/dx + (omega_old[2:,1:-1]-2*omega_old[1:-1,1:-1]+omega_old[0:-2,1:-1])/dy/dy )
    psi_old = np.copy(psi)
    
    # Update Psi
    psi[1:-1,1:-1] = (-omega_old[1:-1,1:-1] - ((psi_old[2:,1:-1] + psi_old[0:-2,1:-1])/dy/dy + (psi_old[1:-1,2:] + psi_old[1:-1,0:-2])/dx/dx))*(-0.5/((1/dx/dx)+(1/dy/dy)))
   
    if i%pauseCount==0:
        err = np.linalg.norm(psi-psi_old)
        print(err)                                       
        if i<10:
            padding = '000'
        elif i>=10 and i<100:
            padding = '00'
        elif i>=100 and i<1000:
            padding = '0'
        else:
            padding=''
        fileName = 'images/lidDrivenCavity/Re'+str(reynoldsNumber)+'/LC_'+padding+str(i)

        fig, ax = plt.subplots()

        U[1:-1,1:-1] = -(psi[1:-1,1:-1] - psi[2:,1:-1])/dy
        V[1:-1,1:-1] = -(psi[1:-1,1:-1] - psi[1:-1,0:-2])/dx
        speed = np.sqrt(U**2 + V**2)

        ax.streamplot(X,Y,U,V,2.0,color='k',linewidth=0.5,arrowstyle='-',broken_streamlines=False)#, color = np.flipud(speed))#, linewidth = 2, cmap ='autumn')
        cs = ax.contourf(X, Y, omega,np.linspace(-5,5,201),cmap='RdYlGn',extend='both')
        cb = fig.colorbar(cs, ax=ax, shrink=0.9,location="right",ticks = np.linspace(-5,5,11), format='%.0f')
        cb.set_label('Vorticity',fontsize=16)
        ax.set_aspect('equal', adjustable='box')
        plt.suptitle("Lid Driven Cavity at Re = "+str(reynoldsNumber),fontsize=20)
        if saveFile == 1:
            plt.savefig(fileName,dpi=200)
        if i < numSteps-1:
            plt.close(fig)


plt.show()
