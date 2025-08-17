# Script to compute the numerical solution of the compressible Euler equations applied to a Sod shock tube

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from numpy import genfromtxt

# Geometric params
def initializeGrid(Lx,Ly,nx,ny):
    """Create a uniform grid and return mesh and spacings."""
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    X, Y = np.meshgrid(np.linspace(0,Lx,nx), np.linspace(0,Ly,ny))
    return X,Y,dx,dy

def initializeSolution(X,Y,nx,ny,gam,Lx,diaphragmLocation):
    """Initialize Sod tube left/right states around diaphragmLocation (x)."""

    diaLocationIdx = int((nx-1)*diaphragmLocation/Lx)
    rho = np.zeros([ny,nx])
    u = np.zeros([ny,nx])
    v = np.zeros([ny,nx])
    pressure = np.zeros([ny,nx])

    # Conditions on left of diaphragm
    rho[:,0:diaLocationIdx+1] = 1.0
    pressure[:,0:diaLocationIdx+1] = 1.0

    # Conditions on right of diaphragm
    rho[:,diaLocationIdx+1:] = 0.125
    pressure[:,diaLocationIdx+1:] = 0.1
                  
    e = np.divide(pressure,rho)/(gam-1)

    return rho,u,v,e

def marchSolution(X,Y,rho,u,v,e,gam,Lx,dx,dy,dt,nt,saveFile,pauseCount,Cx=0.4,Cy=0.0):
    """MacCormack march with artificial viscosity. Cx,Cy control AV strength."""

    locationOfRareFront = Lx*0.5
    locationOfShockFront = Lx*0.5
    ny, nx = rho.shape
    for t in range(nt):
        p = rho*e*(gam-1)
        locationOfRareFront = locationOfRareFront - np.sqrt(gam*(p[0,0]/rho[0,0]))*dt
        locationOfShockFront = locationOfShockFront + np.sqrt(gam*(p[0,-1]/rho[0,-1]))*dt
                
        drho_dx = np.zeros([ny,nx])
        du_dx = np.zeros([ny,nx])
        dv_dx = np.zeros([ny,nx])
        de_dx = np.zeros([ny,nx])
        dp_dx = np.zeros([ny,nx])

        drho_dy = np.zeros([ny,nx])
        du_dy = np.zeros([ny,nx])
        dv_dy = np.zeros([ny,nx])
        de_dy = np.zeros([ny,nx])
        dp_dy = np.zeros([ny,nx])
                
        drho_dx[:,0:-1] = (rho[:,1:] - rho[:,0:-1])/dx
        du_dx[:,0:-1] = (u[:,1:] - u[:,0:-1])/dx
        dv_dx[:,0:-1] = (v[:,1:] - v[:,0:-1])/dx
        de_dx[:,0:-1] = (e[:,1:] - e[:,0:-1])/dx
        dp_dx[:,0:-1] = (p[:,1:] - p[:,0:-1])/dx

        drho_dy[0:-1,:] = (rho[1:,:] - rho[0:-1,:])/dy
        du_dy[0:-1,:] = (u[1:,:] - u[0:-1,:])/dy
        dv_dy[0:-1,:] = (v[1:,:] - v[0:-1,:])/dy
        de_dy[0:-1,:] = (e[1:,:] - e[0:-1,:])/dy
        dp_dy[0:-1,:] = (p[1:,:] - p[0:-1,:])/dy

        # Predictor step
        drho_dt = -(rho*du_dx+u*drho_dx+rho*dv_dy+v*drho_dy)
        du_dt = -(u*du_dx+v*du_dy+np.divide(dp_dx,rho))
        dv_dt = -(u*dv_dx+v*dv_dy+np.divide(dp_dy,rho))
        de_dt = -(u*de_dx+v*de_dy+np.divide(p,rho)*du_dx+np.divide(p,rho)*dv_dy)

        rho_pred = np.copy(rho)
        u_pred = np.copy(u)
        v_pred = np.copy(v)
        e_pred = np.copy(e)

        # Artificial Viscosity
        pressure_term_y = np.divide(abs(p[2:,1:-1]-2*p[1:-1,1:-1]+p[0:-2,1:-1]),(p[2:,1:-1]+2*p[1:-1,1:-1]+p[0:-2,1:-1]))
        pressure_term_x = np.divide(abs(p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,0:-2]),(p[1:-1,2:]+2*p[1:-1,1:-1]+p[1:-1,0:-2]))
        
        rho_pred[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(rho[1:-1,2:]-2*rho[1:-1,1:-1]+rho[1:-1,0:-2])+ Cy*pressure_term_y*(rho[2:,1:-1]-2*rho[1:-1,1:-1]+rho[0:-2,1:-1])
        u_pred[1:-1,1:-1] = u[1:-1,1:-1] + du_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,0:-2])+ Cy*pressure_term_y*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[0:-2,1:-1])
        v_pred[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,0:-2])+ Cy*pressure_term_y*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[0:-2,1:-1])
        e_pred[1:-1,1:-1] = e[1:-1,1:-1] + de_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(e[1:-1,2:]-2*e[1:-1,1:-1]+e[1:-1,0:-2])+ Cy*pressure_term_y*(e[2:,1:-1]-2*e[1:-1,1:-1]+e[0:-2,1:-1])

        
        # Regular BC update

        rho_pred[-1,1:-1]=rho_pred[-2,1:-1]
        u_pred[-1,1:-1]=u_pred[-2,1:-1]
        v_pred[-1,1:-1]=0.0
        e_pred[-1,1:-1]=e_pred[-2,1:-1]

        rho_pred[0,1:-1]=rho_pred[1,1:-1]
        u_pred[0,1:-1]=u_pred[1,1:-1]
        v_pred[0,1:-1]=0.0
        e_pred[0,1:-1]=e_pred[1,1:-1]
        
        p_pred = rho_pred*e_pred*(gam-1)

        # Corrector step
        drho_dx_pred = np.zeros([ny,nx])
        du_dx_pred = np.zeros([ny,nx])
        dv_dx_pred = np.zeros([ny,nx])
        de_dx_pred = np.zeros([ny,nx])
        dp_dx_pred = np.zeros([ny,nx])

        drho_dy_pred = np.zeros([ny,nx])
        du_dy_pred = np.zeros([ny,nx])
        dv_dy_pred = np.zeros([ny,nx])
        de_dy_pred = np.zeros([ny,nx])
        dp_dy_pred = np.zeros([ny,nx])
           
        drho_dx_pred[:,1:] = (rho_pred[:,1:] - rho_pred[:,0:-1])/dx
        du_dx_pred[:,1:] = (u_pred[:,1:] - u_pred[:,0:-1])/dx
        dv_dx_pred[:,1:] = (v_pred[:,1:] - v_pred[:,0:-1])/dx
        de_dx_pred[:,1:] = (e_pred[:,1:] - e_pred[:,0:-1])/dx
        dp_dx_pred[:,1:] = (p_pred[:,1:] - p_pred[:,0:-1])/dx

        drho_dy_pred[1:,:] = (rho_pred[1:,:] - rho_pred[0:-1,:])/dy
        du_dy_pred[1:,:] = (u_pred[1:,:] - u_pred[0:-1,:])/dy
        dv_dy_pred[1:,:] = (v_pred[1:,:] - v_pred[0:-1,:])/dy
        de_dy_pred[1:,:] = (e_pred[1:,:] - e_pred[0:-1,:])/dy
        dp_dy_pred[1:,:] = (p_pred[1:,:] - p_pred[0:-1,:])/dy

        drho_dt_pred = -(rho_pred*du_dx_pred+u_pred*drho_dx_pred+rho_pred*dv_dy_pred+v_pred*drho_dy_pred)
        du_dt_pred = -(u_pred*du_dx_pred+v_pred*du_dy_pred+np.divide(dp_dx_pred,rho_pred))
        dv_dt_pred = -(u_pred*dv_dx_pred+v_pred*dv_dy_pred+np.divide(dp_dy_pred,rho_pred))
        de_dt_pred = -(u_pred*de_dx_pred+v_pred*de_dy_pred+np.divide(p_pred,rho_pred)*du_dx_pred+np.divide(p_pred,rho_pred)*dv_dy_pred)

        drho_dt_av = 0.5*(drho_dt + drho_dt_pred)
        du_dt_av = 0.5*(du_dt + du_dt_pred)
        dv_dt_av = 0.5*(dv_dt + dv_dt_pred)
        de_dt_av = 0.5*(de_dt + de_dt_pred)

        # Artificial Viscosity
        pressure_term_y = np.divide(abs(p_pred[2:,1:-1]-2*p_pred[1:-1,1:-1]+p_pred[0:-2,1:-1]),(p_pred[2:,1:-1]+2*p_pred[1:-1,1:-1]+p_pred[0:-2,1:-1]))
        pressure_term_x = np.divide(abs(p_pred[1:-1,2:]-2*p_pred[1:-1,1:-1]+p_pred[1:-1,0:-2]),(p_pred[1:-1,2:]+2*p_pred[1:-1,1:-1]+p_pred[1:-1,0:-2]))
        
        # Update variables
        rho[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(rho_pred[1:-1,2:]-2*rho_pred[1:-1,1:-1]+rho_pred[1:-1,0:-2])+ Cy*pressure_term_y*(rho_pred[2:,1:-1]-2*rho_pred[1:-1,1:-1]+rho_pred[0:-2,1:-1])
        u[1:-1,1:-1] = u[1:-1,1:-1] + du_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(u_pred[1:-1,2:]-2*u_pred[1:-1,1:-1]+u_pred[1:-1,0:-2])+ Cy*pressure_term_y*(u_pred[2:,1:-1]-2*u_pred[1:-1,1:-1]+u_pred[0:-2,1:-1])
        v[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(v_pred[1:-1,2:]-2*v_pred[1:-1,1:-1]+v_pred[1:-1,0:-2])+ Cy*pressure_term_y*(v_pred[2:,1:-1]-2*v_pred[1:-1,1:-1]+v_pred[0:-2,1:-1])
        e[1:-1,1:-1] = e[1:-1,1:-1] + de_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(e_pred[1:-1,2:]-2*e_pred[1:-1,1:-1]+e_pred[1:-1,0:-2])+ Cy*pressure_term_y*(e_pred[2:,1:-1]-2*e_pred[1:-1,1:-1]+e_pred[0:-2,1:-1])

        rho[-1,1:-1]=rho[-2,1:-1]
        u[-1,1:-1]=u[-2,1:-1]
        v[-1,1:-1]=0.0
        e[-1,1:-1]=e[-2,1:-1]

        rho[0,1:-1]=rho[1,1:-1]
        u[0,1:-1]=u[1,1:-1]
        v[0,1:-1]=0.0
        e[0,1:-1]=e[1,1:-1]

        
        if t%pauseCount==0:
            p = rho*e*(gam-1)
                                                        
            if t<10:
                padding = '000'
            elif t>=10 and t<100:
                padding = '00'
            elif t>=100 and t<1000:
                padding = '0'
            else:
                padding=''
            import os
            fileName = 'images/sod/Sod_Shock_Tube_'+padding+str(t)
            fig, ax = plt.subplots()
            cs = ax.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], p[1:-1,1:-1],np.linspace(0.1,1,91),cmap='RdYlGn_r', extend='both')
            
            cb = fig.colorbar(cs, ax=ax, shrink=0.9,location="bottom")
            cb.set_label('Non-dimensional Pressure')
            ax.set_title("Sod Shock Tube Simulation")
            ax.axis("Equal")
            ax.set_aspect('equal', 'box')

            if saveFile == 1:
                os.makedirs(os.path.dirname(fileName), exist_ok=True)
                plt.savefig(fileName,dpi=200)
            if t < nt-1:
                plt.close(fig)


    fig1, ax1 = plt.subplots()
    ax1.plot(X[1,0:], p[1,0:]/p[1,0],'r-',linewidth=4,label='Prediction')
    exp_data = genfromtxt('sod_pressure_experimental.csv', delimiter=',')
    ax1.plot(exp_data[:,0], exp_data[:,1],'ko',label='Experiment')
    ax1.set_title("Pressure Variation at Non-dimensional Time = 0.2",fontsize=24)
    ax1.set_xlabel('X',fontsize=18)
    plt.xticks(fontsize=14)
    ax1.set_ylabel('$P/P_{left}$',fontsize=18)
    plt.yticks(fontsize=14)
    ax1.legend(fontsize=16)
    
    plt.show(block=True)
 
if __name__ == '__main__':
    Lx = 1
    Ly = 0.1
    nx = 401
    ny = 10
    nt = 20000
    dt = 0.00001
    gam = 1.4

    saveFile = 0
    if saveFile==1:
        pauseCount = 100
    else:
        pauseCount = max(nt-1,1)
    
    diaphragmLocation = 0.5*Lx
    [X,Y,dx,dy] = initializeGrid(Lx,Ly,nx,ny)
    [rho,u,v,e] = initializeSolution(X,Y,nx,ny,gam,Lx,diaphragmLocation)
    marchSolution(X,Y,rho,u,v,e,gam,Lx,dx,dy,dt,nt,saveFile,pauseCount)
