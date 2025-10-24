# Script to compute the numerical solution of the compressible Euler equations applied to the Shock-Vortex-Interaction problem

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

def initializeSolution(X,Y,nx,ny,gam,vortex_gamma):
    # Function to initialize the shock

    M_inf = 1.2 #Hardwired to correspond to Zhang et al
    shockLocationIdx = int((nx-1)/4)
    rho = np.zeros([ny,nx])
    u = np.zeros([ny,nx])
    v = np.zeros([ny,nx])
    pressure = np.zeros([ny,nx])

    # The Rankine-Hugoniot normal shock relations
    gam_const = (gam+1)/(gam-1)
    p2_by_p1 = (2*gam*M_inf*M_inf-(gam-1))/(gam+1)
    rho2_by_rho1 = (gam_const*p2_by_p1 + 1)/(gam_const + p2_by_p1)
    u2_by_u1 = (gam_const + p2_by_p1)/(gam_const*p2_by_p1 + 1)
    
    rho[:,shockLocationIdx+1:] = rho2_by_rho1
    u[:,shockLocationIdx+1:] = M_inf*u2_by_u1
    v[:,shockLocationIdx+1:] = 0.0
    pressure[:,shockLocationIdx+1:] = (1/gam)*p2_by_p1

    # In an earlier version of this code, I attempted introducing the vortex at the very beginning (it didn't work, haha)
    x_vortex = 0.1*Lx
    y_vortex = 0.5*Ly

    dist = np.sqrt(np.multiply(X-x_vortex,X-x_vortex) + np.multiply(Y-y_vortex,Y-y_vortex))

    v[:,0:shockLocationIdx+1] = 0+np.exp(0.5*(1-np.multiply(dist[:,0:shockLocationIdx+1],dist[:,0:shockLocationIdx+1])))*vortex_gamma*(X[:,0:shockLocationIdx+1]-x_vortex)/(2*np.pi)
    u[:,0:shockLocationIdx+1] = M_inf-np.exp(0.5*(1-np.multiply(dist[:,0:shockLocationIdx+1],dist[:,0:shockLocationIdx+1])))*vortex_gamma*(Y[:,0:shockLocationIdx+1]-y_vortex)/(2*np.pi)
    tempVar = 0.125*vortex_gamma*vortex_gamma*(gam-1)/(gam*np.pi*np.pi);
    rho[:,0:shockLocationIdx+1]=np.power( (1.0 - tempVar*np.exp(1.0*(1-dist[:,0:shockLocationIdx+1]*dist[:,0:shockLocationIdx+1]))) , (1.0/(gam-1)) )

    pressure[:,0:shockLocationIdx+1] = (1/gam)*pow( (rho[:,0:shockLocationIdx+1]),(gam))
    e = np.divide(pressure,rho)/(gam-1)

    return rho,u,v,e

def initializeVortex(X,Y,nx,ny,gam,vortex_gamma,rho,u,v,e):
    # Function to initialize the isentropic vortex perturbation into an already-converged shock solution

    shockLocationIdx = int((nx-1)/4)
    pressure = rho*e*(gam-1)
                  
    x_vortex = 0.1*Lx
    y_vortex = 0.5*Ly

    dist = np.sqrt(np.multiply(X-x_vortex,X-x_vortex) + np.multiply(Y-y_vortex,Y-y_vortex))

    delta_v = np.exp(0.5*(1-np.multiply(dist,dist)))*vortex_gamma*(X-x_vortex)/(2*np.pi)
    delta_u =-np.exp(0.5*(1-np.multiply(dist,dist)))*vortex_gamma*(Y-y_vortex)/(2*np.pi)
    delta_T = -0.125*vortex_gamma*vortex_gamma*(gam-1)/(gam*np.pi*np.pi)
    
    rho=rho*np.power( (1.0 + delta_T*np.exp(1.0*(1-dist*dist))) , (1.0/(gam-1)) )
    pressure=pressure*np.power( (1.0 + delta_T*np.exp(1.0*(1-dist*dist))) , (gam/(gam-1)) )
    u = u + delta_u
    v = v + delta_v

    e = np.divide(pressure,rho)/(gam-1)

    return rho,u,v,e

def marchSolution(X,Y,rho,u,v,e,meanP,gam,dx,dy,dt,nt,plotOption,saveFile,pauseCount):

    midX_index = int(np.round(X.shape[1]/2))

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

        # Artificial Viscosity
        Cx = 0.3
        Cy = 0.3
        pressure_term_y = np.divide(abs(p[2:,1:-1]-2*p[1:-1,1:-1]+p[0:-2,1:-1]),(p[2:,1:-1]+2*p[1:-1,1:-1]+p[0:-2,1:-1]))
        pressure_term_x = np.divide(abs(p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,0:-2]),(p[1:-1,2:]+2*p[1:-1,1:-1]+p[1:-1,0:-2]))

        rho_pred[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(rho[1:-1,2:]-2*rho[1:-1,1:-1]+rho[1:-1,0:-2])+ Cy*pressure_term_y*(rho[2:,1:-1]-2*rho[1:-1,1:-1]+rho[0:-2,1:-1])
        u_pred[1:-1,1:-1] = u[1:-1,1:-1] + du_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,0:-2])+ Cy*pressure_term_y*(u[2:,1:-1]-2*u[1:-1,1:-1]+u[0:-2,1:-1])
        v_pred[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,0:-2])+ Cy*pressure_term_y*(v[2:,1:-1]-2*v[1:-1,1:-1]+v[0:-2,1:-1])
        e_pred[1:-1,1:-1] = e[1:-1,1:-1] + de_dt[1:-1,1:-1]*dt + Cx*pressure_term_x*(e[1:-1,2:]-2*e[1:-1,1:-1]+e[1:-1,0:-2])+ Cy*pressure_term_y*(e[2:,1:-1]-2*e[1:-1,1:-1]+e[0:-2,1:-1])

        rho_pred[0,1:-1]=rho_pred[-2,1:-1]
        u_pred[0,1:-1]=u_pred[-2,1:-1]
        v_pred[0,1:-1]=v_pred[-2,1:-1]
        e_pred[0,1:-1]=e_pred[-2,1:-1]

        rho_pred[-1,1:-1]=rho_pred[1,1:-1]
        u_pred[-1,1:-1]=u_pred[1,1:-1]
        v_pred[-1,1:-1]=v_pred[1,1:-1]
        e_pred[-1,1:-1]=e_pred[1,1:-1]        

        rho_pred[1:-1,-1]=2*rho_pred[1:-1,-2]-rho_pred[1:-1,-3]
        u_pred[1:-1,-1]=2*u_pred[1:-1,-2]-u_pred[1:-1,-3]
        v_pred[1:-1,-1]=2*v_pred[1:-1,-2]-v_pred[1:-1,-3]
        e_pred[1:-1,-1]=2*e_pred[1:-1,-2]-e_pred[1:-1,-3]
        
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

        # Artificial Viscosity (as per Anderson's recommendation, not Chung)
        pressure_term_y = np.divide(abs(p_pred[2:,1:-1]-2*p_pred[1:-1,1:-1]+p_pred[0:-2,1:-1]),(p_pred[2:,1:-1]+2*p_pred[1:-1,1:-1]+p_pred[0:-2,1:-1]))
        pressure_term_x = np.divide(abs(p_pred[1:-1,2:]-2*p_pred[1:-1,1:-1]+p_pred[1:-1,0:-2]),(p_pred[1:-1,2:]+2*p_pred[1:-1,1:-1]+p_pred[1:-1,0:-2]))

        # Update variables
        rho[1:-1,1:-1] = rho[1:-1,1:-1] + drho_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(rho_pred[1:-1,2:]-2*rho_pred[1:-1,1:-1]+rho_pred[1:-1,0:-2])+ Cy*pressure_term_y*(rho_pred[2:,1:-1]-2*rho_pred[1:-1,1:-1]+rho_pred[0:-2,1:-1])
        u[1:-1,1:-1] = u[1:-1,1:-1] + du_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(u_pred[1:-1,2:]-2*u_pred[1:-1,1:-1]+u_pred[1:-1,0:-2])+ Cy*pressure_term_y*(u_pred[2:,1:-1]-2*u_pred[1:-1,1:-1]+u_pred[0:-2,1:-1])
        v[1:-1,1:-1] = v[1:-1,1:-1] + dv_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(v_pred[1:-1,2:]-2*v_pred[1:-1,1:-1]+v_pred[1:-1,0:-2])+ Cy*pressure_term_y*(v_pred[2:,1:-1]-2*v_pred[1:-1,1:-1]+v_pred[0:-2,1:-1])
        e[1:-1,1:-1] = e[1:-1,1:-1] + de_dt_av[1:-1,1:-1]*dt + Cx*pressure_term_x*(e_pred[1:-1,2:]-2*e_pred[1:-1,1:-1]+e_pred[1:-1,0:-2])+ Cy*pressure_term_y*(e_pred[2:,1:-1]-2*e_pred[1:-1,1:-1]+e_pred[0:-2,1:-1])

        rho[0,1:-1]=rho[-2,1:-1]
        u[0,1:-1]=u[-2,1:-1]
        v[0,1:-1]=v[-2,1:-1]
        e[0,1:-1]=e[-2,1:-1]

        rho[-1,1:-1]=rho[1,1:-1]
        u[-1,1:-1]=u[1,1:-1]
        v[-1,1:-1]=v[1,1:-1]
        e[-1,1:-1]=e[1,1:-1]

        rho[1:-1,-1]=2*rho[1:-1,-2]-rho[1:-1,-3]
        u[1:-1,-1]=2*u[1:-1,-2]-u[1:-1,-3]
        v[1:-1,-1]=2*v[1:-1,-2]-v[1:-1,-3]
        e[1:-1,-1]=2*e[1:-1,-2]-e[1:-1,-3]

        p = rho*e*(gam-1)

        if plotOption == 1:    
            if t%pauseCount==0:
                                                            
                if t<10:
                    padding = '000'
                elif t>=10 and t<100:
                    padding = '00'
                elif t>=100 and t<1000:
                    padding = '0'
                else:
                    padding=''
                fileName = 'images/shockVortex/Shock_Vortex_'+padding+str(t)
                fig, ax = plt.subplots()
                
                cs5 = ax.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], p[1:-1,1:-1],np.linspace(meanP[1,midX_index]-0.001,meanP[1,midX_index]+0.001,400),cmap='jet',extend='both')
                cb = fig.colorbar(cs5, ax=ax, shrink=0.9)
                cb.set_label('Non-dimensional Pressure')
                
                ax.axis("Equal")
                ax.set_aspect('equal', 'box')
                plt.suptitle("Shock Vortex Interaction",fontsize=20)

                if saveFile == 1:
                    plt.savefig(fileName,dpi=200)
                if t < nt-1:
                    plt.close(fig)

    return rho,u,v,e,p

   
if __name__ == '__main__':
    Lx = 40
    Ly = 40
    nx = 321
    ny = 321
    nt = 30001
    dt = 0.001
    gam = 1.4
    vortex_gamma = 0.125

    saveFile = 0
    if saveFile==1:
        pauseCount = 200
    else:
        pauseCount = max(nt-1,1)

    # Initialize grid and solution
    [X,Y,dx,dy] = initializeGrid(Lx,Ly,nx,ny)
    [rho,u,v,e] = initializeSolution(X,Y,nx,ny,gam,0)
    p = rho*e*(gam-1)

    # Converge the shock-only solution
    [rho,u,v,e,meanP] = marchSolution(X,Y,rho,u,v,e,p,gam,dx,dy,dt,nt,1,0,nt)
    print('Shock solution initialized')

    # Introduce the isentropic vortex perturbation
    [rho,u,v,e] = initializeVortex(X,Y,nx,ny,gam,vortex_gamma,rho,u,v,e)
    print('Vortex initialized')

    # Run the actual shock-vortex-interaction problem
    [rho,u,v,e,p] = marchSolution(X,Y,rho,u,v,e,meanP,gam,dx,dy,dt,nt,1,saveFile,pauseCount)
    plt.show()
