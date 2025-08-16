# Script to compute the numerical solution of the compressible Euler equations applied to a convecting isentropic vortex

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Geometric params
def initializeGrid(Lx,Ly,nx,ny):
    dx = Lx/(nx-1)
    dy = Ly/(ny-1)
    X, Y = np.meshgrid(np.linspace(0,Lx,nx), np.linspace(0,Ly,ny))
    return X,Y,dx,dy

def initializeSolution(X,Y,nx,ny,gam,vortex_gamma,u_inf):
    rho = np.zeros([ny,nx])
    u = np.zeros([ny,nx])
    v = np.zeros([ny,nx])

    x_vortex = 0.5*Lx
    y_vortex = 0.5*Ly

    dist = np.sqrt(np.multiply(X-x_vortex,X-x_vortex) + np.multiply(Y-y_vortex,Y-y_vortex))

    # The perturbation velocities of the isentropic vortex
    v = 0+np.exp(0.5*(1-np.multiply(dist,dist)))*vortex_gamma*(X-x_vortex)/(2*np.pi)
    u = u_inf-np.exp(0.5*(1-np.multiply(dist,dist)))*vortex_gamma*(Y-y_vortex)/(2*np.pi)
    
    tempVar = 0.125*vortex_gamma*vortex_gamma*(gam-1)/(gam*np.pi*np.pi);
    rho=np.power( (1.0 - tempVar*np.exp(1.0*(1-dist*dist))) , (1.0/(gam-1)) )
    pressure = np.power( (rho),(gam))
    e = np.divide(pressure,rho)/(gam-1)

    return rho,u,v,e

def marchSolution(X,Y,rho,u,v,e,gam,dx,dy,dt,nt,saveFile,pauseCount):

    for t in range(nt):

        p = rho*e*(gam-1)
        
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

        rho_pred[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt[1:-1,1:-1]*dt
        u_pred[1:-1,1:-1] = u[1:-1,1:-1] + du_dt[1:-1,1:-1]*dt
        v_pred[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt[1:-1,1:-1]*dt
        e_pred[1:-1,1:-1] = e[1:-1,1:-1] + de_dt[1:-1,1:-1]*dt
        
        # Regular BC update
        rho_pred[1:-1,-1]=rho_pred[1:-1,1]
        u_pred[1:-1,-1]=u_pred[1:-1,1]
        v_pred[1:-1,-1]=v_pred[1:-1,1]
        e_pred[1:-1,-1]=e_pred[1:-1,1]

        rho_pred[-1,1:-1]=rho_pred[1,1:-1]
        u_pred[-1,1:-1]=u_pred[1,1:-1]
        v_pred[-1,1:-1]=v_pred[1,1:-1]
        e_pred[-1,1:-1]=e_pred[1,1:-1]

        rho_pred[1:-1,0]=rho_pred[1:-1,-2]
        u_pred[1:-1,0]=u_pred[1:-1,-2]
        v_pred[1:-1,0]=v_pred[1:-1,-2]
        e_pred[1:-1,0]=e_pred[1:-1,-2]

        rho_pred[0,1:-1]=rho_pred[-2,1:-1]
        u_pred[0,1:-1]=u_pred[-2,1:-1]
        v_pred[0,1:-1]=v_pred[-2,1:-1]
        e_pred[0,1:-1]=e_pred[-2,1:-1]
        

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

        # Update variables
        rho[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt_av[1:-1,1:-1]*dt
        u[1:-1,1:-1] = u[1:-1,1:-1] + du_dt_av[1:-1,1:-1]*dt
        v[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt_av[1:-1,1:-1]*dt
        e[1:-1,1:-1] = e[1:-1,1:-1] + de_dt_av[1:-1,1:-1]*dt

        # Regular BC update
        rho[1:-1,0]=rho[1:-1,-2]
        u[1:-1,0]=u[1:-1,-2]
        v[1:-1,0]=v[1:-1,-2]
        e[1:-1,0]=e[1:-1,-2]

        rho[0,1:-1]=rho[-2,1:-1]
        u[0,1:-1]=u[-2,1:-1]
        v[0,1:-1]=v[-2,1:-1]
        e[0,1:-1]=e[-2,1:-1]

        rho[1:-1,-1]=rho[1:-1,1]
        u[1:-1,-1]=u[1:-1,1]
        v[1:-1,-1]=v[1:-1,1]
        e[1:-1,-1]=e[1:-1,1]

        rho[-1,1:-1]=rho[1,1:-1]
        u[-1,1:-1]=u[1,1:-1]
        v[-1,1:-1]=v[1,1:-1]
        e[-1,1:-1]=e[1,1:-1]

        if t%pauseCount==0:
            dv_dx[:,1:-1] = (v[:,2:] - v[:,0:-2])/(2*dx)
            du_dy[1:-1,:] = (u[2:,:] - u[0:-2,:])/(2*dy)
            vort = abs(dv_dx - du_dy)
            if t<10:
                padding = '000'
            elif t>=10 and t<100:
                padding = '00'
            elif t>=100 and t<1000:
                padding = '0'
            else:
                padding=''
            fileName = 'images/vortex/isentropicVortex_'+padding+str(t)
            fig, ax = plt.subplots()
            cs = ax.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], vort[1:-1,1:-1],[0,0.08,0.16,0.24,0.32])

            cb = fig.colorbar(cs, ax=ax, shrink=0.9)
            cb.set_label('Absolute Non-dimensional Vorticity',fontsize=14)
            ax.set_title("Convecting Isentropic Vortex",fontsize=20)
            ax.set_xlabel('X',fontsize=18)
            ax.set_ylabel('Y',fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            if saveFile == 1:
                plt.savefig(fileName,dpi=200)
            if t < nt-1:
                plt.close(fig)
    
if __name__ == '__main__':

    # Geometric and Flow parameters
    Lx = 10
    Ly = 10
    nx = 201
    ny = 201
    nt = 10001
    dt = 0.001
    gam = 1.4
    u_inf = 1.0
    vortex_gamma = 0.5

    # Save options for gif creation
    saveFile = 0
    if saveFile==1:
        pauseCount = 100
    else:
        pauseCount = max(nt-1,1)

    # Initialize grid and solution
    [X,Y,dx,dy] = initializeGrid(Lx,Ly,nx,ny)
    [rho,u,v,e] = initializeSolution(X,Y,nx,ny,gam,vortex_gamma,u_inf)

    # March in time
    marchSolution(X,Y,rho,u,v,e,gam,dx,dy,dt,nt,saveFile,pauseCount)
    
    plt.show()
