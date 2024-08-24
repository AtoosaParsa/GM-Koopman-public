""" Functions for molecular dynamics simulation of granular chains with Hertizian contacts and fixed boundaries
Partially developed by Qikai Wu, Dong Wang, Annie Xia from The O'Hern Group at Yale University <https://jamming.research.yale.edu/>

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = ['Atoosa Parsa', 'Qikai Wu', 'Dong Wang', 'Annie Xia']
__license__ = 'MIT License'
__version__ = '0.0.3'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"



import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from FFT_functions import FFT_Fup, FFT_vCorr

def hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp(k_list, k_type, B, B_pp, B_wp, Nt, N, x_ini, y_ini, D0, m0, L, u_0, v_0):
    """
    New MD code: similar to MD_VibrSP_ConstV_Yfixed_DiffK with 
        1) added particle-particle and particle-wall damping (rel. velocities) with coefficients B_pp and B_wp
            * for B_pp = 0 and B_wp = 0, damping behaves same as before
        2) walls in x at 0 and Lx, not periodic

    Parameters:
    ------
    k_list (2x1, or {stiffness classes x 1}): unique ks
    k_type (Nx1): indices per particle to put into above array
    B (sc): 'background fluid' damping coefficient
    B_pp (sc): particle-particle damping coefficient
    B_wp (sc): particle-wall damping coefficient
    Nt (int): number of time points to simulate
    N (int): number of particles
    x_ini (Nx1): initial particle x coordinates
    y_ini (Nx1): initial particle y coordinates
    D0 (Nx1): particle diameters
    m0 (Nx1): particle masses
    L (2x1): box dimensions, [Lx, Ly]
    Freq_Vibr1 (sc): frequency of input 1 signal
    Amp_Vibr1 (sc): amplitude of input 1 signal
    ind_in1 (int): index/location of input 1 particle
    ... similar for 2
    -----

    Returns:
    ------
    freq_fft (Nf x 1): fft frequencies
    fft_in1 (Nf x 1): input 1 fft amplitudes (x)
    fft_in2 (Nf x 1): input 2 fft amplitudes (x)
    fft_x_out (Nf x 1): output fft amplitudes (x)
    fft_y_out (Nf x 1): output fft amplitudes (y)
    np.mean(cont) (sc): mean contacts
    nt_rec (int): times to record
    Ek_now (Nt x 1): instantaneous kinetic energy 
    Ep_now (Nt x 1): instantaneous potential energy
    cont_now (Nt x 1): inst. contact array
    x_out (Nt x 1): output particle x signal
    y_out (Nt x 1): output particle y signal
    x_in1 (Nt x 1) input 1 particle x signal
    ------

        
    """
    Lx = L[0]
    Ly = L[1]

    mark_vibrY = 0
    mark_resonator = 0
    dt = D0[0]/10
    print(dt)
    Nt = int(Nt)
    
    nt_rec = np.linspace(0, Nt, 1)    
    
    nt_rec = nt_rec.astype(int)
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)

    vx = np.zeros(N)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini)
    y = np.array(y_ini)
    
    x_save = np.array(x_ini)
    y_save = np.array(y_ini)
    
    # try to save and visualize
    x_all = np.zeros((Nt, N))
    y_all = np.zeros((Nt, N))
    vx_all = np.zeros((Nt, N))
    vy_all = np.zeros((Nt, N))
    ax_all = np.zeros((Nt, N))
    ay_all = np.zeros((Nt, N))
    
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)

    for nt in np.arange(Nt):
        if nt==0:
            x = x + u_0
            vx = vx + v_0
            
        x = x+vx*dt+ax_old*dt**2/2  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2
        
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx - B*vx
        Fy_all = Fy - B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
                
        vx = vx+(ax_old+ax)*dt/2  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2        
        ax_old = ax
        ay_old = ay
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))

        #old method for recording
# =============================================================================
#     Ek_now = []
#     Ep_now = []
#     cont_now = []
#     for ii in np.arange(len(nt_rec)-1):
#         Ek_now.append(np.mean(Ek[nt_rec[ii]:nt_rec[ii+1]]))
#         Ep_now.append(np.mean(Ep[nt_rec[ii]:nt_rec[ii+1]]))
#         cont_now.append(np.mean(cont[nt_rec[ii]:nt_rec[ii+1]]))        
#     nt_rec = (nt_rec[1:] + nt_rec[:-1]) / 2
# =============================================================================

        # track all instantaneous information, can comment out and use old method above for speed/memory
        Ek_now = Ek
        Ep_now = Ep
        cont_now = cont
        nt_rec = np.arange(Nt)
        
        x_all[nt, :] = x-x_ini
        y_all[nt, :] = y
        vx_all[nt, :] = vx
        vy_all[nt, :] = vy
        ax_all[nt, :] = ax
        ay_all[nt, :] = ay       

    #t_end = time.time()
    #print ("time=%.3e" %(t_end-t_start))
    #CB_ratio = min(cont)/max(cont)
    #print ("freq=%f, cont_min/cont_max=%f, Ek_mean=%.3e, Ep_mean=%.3e\n" %(Freq_Vibr, CB_ratio, np.mean(Ek), np.mean(Ep)))
    
    return x_all, nt_rec, dt, Nt, cont, vx_all, ax_all

def hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3(k_list, k_type, B, B_pp, B_wp, Nt, dt, N, x_disp, x_ini, y_ini, v_ini, D0, m0, L):

    Lx = L[0]
    Ly = L[1]
    
    dt = dt
    Nt = int(Nt)
    
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)

    vx = np.array(v_ini)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini+x_disp)
    y = np.array(y_ini)
    
    x_save = np.array(x_ini)
    y_save = np.array(y_ini)
    
    x_all = np.zeros((Nt, N))
    v_all = np.zeros((Nt, N))    

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)

    for nt in np.arange(Nt):
        x_all[nt, :] = x - x_ini
        v_all[nt, :] = vx
            
        x = x+vx*dt+ax_old*dt**2/2  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2
        
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx - B*vx
        Fy_all = Fy - B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
                
        vx = vx+(ax_old+ax)*dt/2  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2        
        ax_old = ax
        ay_old = ay
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))

    return cont, Ek, Ep, p, x_all, v_all

def hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3_c(k_list, k_type, B, B_pp, B_wp, Nt, dt, N, x_disp, x_ini, y_ini, v_ini, D0, m0, L, Freq_Vibr1, Amp_Vibr1, Freq_Vibr2, Amp_Vibr2):

    Lx = L[0]
    Ly = L[1]
    
    dt = dt
    Nt = int(Nt)
    
    
    Ep = np.zeros(Nt)
    Ek = np.zeros(Nt)   
    cont = np.zeros(Nt)
    p = np.zeros(Nt)
    
    x_out = np.zeros(Nt)
    y_out = np.zeros(Nt)

    x_in1 = Amp_Vibr1*np.sin(Freq_Vibr1*dt*np.arange(Nt))+Amp_Vibr2*np.sin(Freq_Vibr2*dt*np.arange(Nt))+x_ini[0]+x_disp[0]
    vx_in1 = np.gradient(x_in1)
        
    vx = np.array(v_ini)
    vy = np.zeros(N)    
    
    ax_old = np.zeros(N)
    ay_old = np.zeros(N)
    
    x = np.array(x_ini+x_disp)
    y = np.array(y_ini)
    
    x_save = np.array(x_ini)
    y_save = np.array(y_ini)
    
    x_all = np.zeros((Nt, N))
    v_all = np.zeros((Nt, N))    

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)

    for nt in np.arange(Nt):
        x_all[nt, :] = x - x_ini
        v_all[nt, :] = vx
            
        x = x+vx*dt+ax_old*dt**2/2  # first step in Verlet integration
        y = y+vy*dt+ay_old*dt**2/2

        x[0] = x_in1[nt]
        y[0] = y_ini[0]
        vx[0] = vx_in1[nt]
        vy[0] = 0
        
        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp)
        
        Ep[nt] = Ep_now
        cont[nt] = cont_now
        p[nt] = p_now
        
        Fx_all = Fx - B*vx
        Fy_all = Fy - B*vy
        
        ax = np.divide(Fx_all, m0)
        ay = np.divide(Fy_all, m0)
        
        ax[0] = 0
        
        vx = vx+(ax_old+ax)*dt/2  # second step in Verlet integration
        vy = vy+(ay_old+ay)*dt/2        
        ax_old = ax
        ay_old = ay
        Ek[nt] = sum(0.5*np.multiply(m0,np.multiply(vx, vx)+np.multiply(vy, vy)))

    return cont, Ek, Ep, p, x_all, v_all, x_in1, vx_in1

def hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_6(k_list, k_type, B, B_pp, B_wp, Nt, dt, N, x_ini, y_ini, x_disp, v_ini, D0, m0, L):
    
    Lx = L[0]
    Ly = L[1]

    dt = dt
    Nt = int(Nt)
    
    x_all = np.zeros((Nt, N))
    v_all = np.zeros((Nt, N))    

    #print(D0)
    #print(x_ini)
    #print(y_ini)
    
    w, v = hertz_DM_mass_DiffK_RealBoundX_2(N, x_ini, y_ini, D0, m0, 0.0, Lx, 0.0, Ly, k_list, k_type)
    w = np.real(w)
    v = np.real(v)
    freq = np.sqrt(np.absolute(w))
    ind_sort = np.argsort(freq)
    freq = freq[ind_sort]
    v = v[:, ind_sort]
    #ind = freq > 1e-4
    eigen_freq = freq #[ind] # N X 1
    eigen_mode = v #[:, ind] # N X N

    #print(freq)
    #print(v)
    
    for i in np.arange(N):
        d = np.transpose(eigen_mode[:, i])@eigen_mode[:, i]
        eigen_mode[:, i] = eigen_mode[:, i] / np.sqrt(d)
    
    
    A = np.transpose(eigen_mode) @ x_disp #np.linalg.inv(eigen_mode) @ np.transpose(x_disp)
    x_all[0, :] = x_disp
    for nt in np.arange(1, Nt):
        C = np.cos(eigen_freq*nt*dt)
        x_all[nt, :] = eigen_mode @ np.transpose(A*C)
    
    v_all[0, :] = v_ini
    for nt in np.arange(1, Nt):
        C = np.sin(eigen_freq*nt*dt)
        v_all[nt, :] = -eigen_mode @ np.transpose((A*C)*eigen_freq)  


    return eigen_freq, eigen_mode, x_all, v_all

def hertz_FIRE_RealBoundX_ConstV_DiffK(Nt, N, x0, y0, D0, m0, Lx, Ly, k_list, k_type):  
    '''
    Energy minimize via FIRE with real walls in x at {0, Lx}
    '''

    dt_md = 0.01 * D0[0] * np.sqrt(k_list[2])
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    Nt = int(Nt)
    Ep = np.zeros(Nt)
    F_tot = np.zeros(Nt)
    
    vx = np.zeros(N)
    vy = np.zeros(N)    
    x = np.array(x0)
    y = np.array(y0)
    x_save = np.array(x0)
    y_save = np.array(y0)

    Fx = np.zeros(N)
    Fy = np.zeros(N)
    VL_list = np.zeros((N * 10, 2), dtype=int) 
    VL_counter = 0
    VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 1)
    # no damping in energy minimization
    Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, 0, 0)
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy)
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                x -= vx * dt_half
                y -= vy * dt_half
                vx = np.zeros(N)
                vy = np.zeros(N)

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2)) / np.sqrt(np.sum(Fx**2 + Fy**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        x += vx * dt
        y += vy * dt

        Fx = np.zeros(N)
        Fy = np.zeros(N)
        VL_list, VL_counter, x_save, y_save = hertz_VL_RealBoundX_ConstV(N, x, y, D0, Lx, VL_list, VL_counter, x_save, y_save, 0)
        Fx, Fy, Fleft_now, Fright_now, Fup_now, Fbot_now, Ep_now, cont_now, p_now, cont_up = hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D0, 0, Lx, 0, Ly, k_list, k_type, VL_list, VL_counter, vx, vy, 0, 0)
        Ep[nt] = Ep_now

        F_tot[nt] = sum(np.absolute(Fx) + np.absolute(Fy))
        # putting a threshold on total force
        if (F_tot[nt] < 1e-11):
            break

        vx += Fx * dt_half
        vy += Fy * dt_half

    #print(nt)
    #print(F_tot[nt])
    t_end = time.time()
    #print ("F_tot=%.3e" %(F_tot[nt]))
    #print ("time=%.3e" %(t_end-t_start))

    return x, y, p_now

def hertz_force_RealBoundX_DiffK_VL_damp(Fx, Fy, N, x, y, D, x_left, x_right, y_bot, y_up, k_list, k_type, VL_list, VL_counter, vx, vy, B_wp, B_pp):

    """
    Calculates instantaneous forces on a given particle system with
        1) real walls in x at {0, Lx}
        2) 3 kinds of damping, a) background viscous damping b) particle-particle damping
            c) particle-wall damping, controlled respectively by parameters B, B_pp, B_wp
    
    Parameters:
    ------
    Fx (Nx1): forces x 
    Fy (Nx1): forces y
    N (sc): number of particles
    x (Nx1): positions x
    y (Nx1): positions y
    D (Nx1): diameters
    x_left (sc): left wall x position
    x_right(sc): right wall x position
    y_bot (sc): bottom wall y position
    y_up (sc): upper wall y position
    k_list (3x1, or stiffness classes x 1): unique ks
    k_type (Nx1): indices to put into above array
    VL_list
    VL_counter
    -----

    Returns:
    ------
    Fx (Nx1): forces x
    Fy (Nx1): forces y
    Fleft (sc): force on left wall
    Fright (sc): force on right wall
    Fup (sc): force on top wall
    Fbot (sc): force on bottom wall
    Ep (sc): potential energy
    cont (sc): contacts (single time)
    p_now (sc): inst. momentum
    cont_up (sc): contacts on upper wall
    ------
    """
    # initialize
    Fup = 0
    Fbot = 0
    Fleft = 0
    Fright = 0
    Ep = 0
    cont = 0
    cont_up = 0
    cont_left = 0
    cont_right = 0
    p_now = 0
    
    # initialize contacts
    C_left = np.zeros(N)
    C_right = np.zeros(N)
    C_up = np.zeros(N) # upper wall contacts
    C_down = np.zeros(N)
    
    for nn in np.arange(N): # loop through each particle
        d_up = y_up - y[nn] # distance to top wall
        d_bot = y[nn] - y_bot
        d_left = x[nn] - x_left # distance to left wall
        d_right = x_right - x[nn]
        r_now = 0.5 * D[nn] # current radius
        
        if d_up < r_now: # repulsive springs, upper wall
            F = (-k_list[k_type[nn]] * (1 - d_up / r_now)**(3/2) / (r_now)) - (B_wp*vx[nn])
            Fup -= F # increment force on wall
            Fy[nn] += F # increment force on particle
            Ep += 2/5 * k_list[k_type[nn]] * (1 - d_up / r_now)**5/2
            cont_up += 1 # increment contacts with upper wall
            cont += 1 # increment contacts
            C_up[nn] = 1 
            #dbg.set_trace()
            
        if d_bot < r_now: # lower wall
            F = (-k_list[k_type[nn]] * (1 - d_bot / r_now)**(3/2) / (r_now)) - (B_wp*vx[nn])
            Fbot += F
            Fy[nn] -= F
            Ep += 2/5 * k_list[k_type[nn]] * (1 - d_bot / r_now)**5/2
            cont += 1
            C_down[nn] = 1 # set contact matrix
            
        if d_left < r_now: # left wall
            F = (-k_list[k_type[nn]] * (1 - d_left / r_now)**(3/2) / (r_now)) - (B_wp*vx[nn]) # includes wp damping
            Fleft += F # force on wall
            Fx[nn] -= F # force on particle
            Ep += 2/5 * k_list[k_type[nn]] * (1 - d_left / r_now)**5/2
            cont_left += 1 # increment contacts with upper wall
            cont += 1 # increment contacts
            C_left[nn] = 1 
            
        if d_right < r_now: # right wall
            F = (-k_list[k_type[nn]] * (1 - d_right / r_now)**(3/2) / (r_now)) - (B_wp*vx[nn])
            Fright -= F # force on wall
            Fx[nn] += F # force on particle
            Ep += 2/5 * k_list[k_type[nn]] * (1 - d_right / r_now)**5/2
            cont_right += 1 # increment contacts with upper wall
            cont += 1 # increment contacts
            C_right[nn] = 1         

    for vl_idx in np.arange(VL_counter): # particles in verlet list
        nn = VL_list[vl_idx][0]
        mm = VL_list[vl_idx][1]
        dy = y[mm] - y[nn]
        Dmn = 0.5 * (D[mm] + D[nn]) # avg diameter
        if abs(dy) < Dmn: 
            dx = x[mm] - x[nn]
            if abs(dx) < Dmn:
                dmn = np.sqrt(dx**2 + dy**2)
                if dmn < Dmn: # particles in contact
                    k = k_list[(k_type[nn] ^ k_type[mm]) + np.maximum(k_type[nn], k_type[mm])] # cheeky effective spring constant

                    # print("first term is")
                    # print((k_type[nn] ^ k_type[mm]))
                    # print("\n total k is ", k)                  
                    F = -k * (1 - dmn / Dmn)**(3/2) / Dmn / dmn
                    Fx[nn] += F * dx # distribute forces
                    Fx[mm] -= F * dx
                    Fy[nn] += F * dy
                    Fy[mm] -= F * dy
                    
                    # particle-particle damping, relative velocities
                    
                    Fx[nn] -= B_pp * (vx[nn] - vx[mm]) # vector aligned with nn
                    Fy[nn] -= B_pp * (vy[nn] - vy[mm])

                    Fx[mm] += B_pp * (vx[nn] - vx[mm]) # vector aligned with nn
                    Fy[mm] += B_pp * (vy[nn] - vy[mm])                    
                    
                    # damping walls, to 0
                    if C_up[nn] == 1: # upper wall contact
                        Fx[nn] -= B_wp * vx[nn]
                        Fy[nn] -= B_wp * vy[nn]
                        Fup += B_wp * vy[nn]
                        C_up[nn] = 0
                        
                    if C_down[nn] == 1: # lower wall contact
                        Fx[nn] -= B_wp * vx[nn]
                        Fy[nn] -= B_wp * vy[nn]
                        Fbot += B_wp * vy[nn]
                        C_down[nn] = 0                        
                        
                    if C_up[mm] == 1: 
                        Fx[mm] -= B_wp * vx[mm]
                        Fy[mm] -= B_wp * vy[mm]
                        Fup += B_wp * vy[mm]
                        C_up[mm] = 0  
                        
                    if C_down[mm] == 1: 
                        Fx[mm] -= B_wp * vx[mm]
                        Fy[mm] -= B_wp * vy[mm]
                        Fbot += B_wp * vy[mm]
                        C_down[mm] = 0  
                        
                    if C_left[nn] == 1: # left wall contact
                        Fx[nn] -= B_wp * vx[nn]
                        Fy[nn] -= B_wp * vy[nn]
                        Fleft += B_wp * vy[nn]
                        C_up[nn] = 0
                        
                    if C_right[nn] == 1: # right wall contact
                        Fx[nn] -= B_wp * vx[nn]
                        Fy[nn] -= B_wp * vy[nn]
                        Fright += B_wp * vy[nn]
                        C_down[nn] = 0                        
                        
                    if C_left[mm] == 1: 
                        Fx[mm] -= B_wp * vx[mm]
                        Fy[mm] -= B_wp * vy[mm]
                        Fleft += B_wp * vy[mm]
                        C_up[mm] = 0  
                        
                    if C_right[mm] == 1: 
                        Fx[mm] -= B_wp * vx[mm]
                        Fy[mm] -= B_wp * vy[mm]
                        Fright += B_wp * vy[mm]
                        C_down[mm] = 0                          
                    
                    Ep += 2/5 * k * (1 - dmn / Dmn)**5/2  # increment potential energy
                    cont += 1 # increment contact
                    p_now += (-F) * (dx**2 + dy**2) # increment momentum 
    return Fx, Fy, Fleft, Fright, Fup, Fbot, Ep, cont, p_now, cont_up

def hertz_VL_RealBoundX_ConstV(N, x, y, D, Lx, VL_list, VL_counter_old, x_save, y_save, first_call):   
    """
    Calculates the verlet list VL for system with real walls in x at {0, Lx}
    """ 
    
    r_factor = 1.2
    r_cut = np.amax(D)
    r_list = r_factor * r_cut
    r_list_sq = r_list**2
    r_skin_sq = ((r_factor - 1.0) * r_cut)**2

    if first_call == 0:
        dr_sq_max = 0.0
        for nn in np.arange(N):
            dy = y[nn] - y_save[nn]
            dx = x[nn] - x_save[nn]
            # dx = dx - round(dx / Lx) * Lx
            dr_sq = dx**2 + dy**2
            if dr_sq > dr_sq_max:
                dr_sq_max = dr_sq
        if dr_sq_max < r_skin_sq:
            return VL_list, VL_counter_old, x_save, y_save

    VL_counter = 0
    
    for nn in np.arange(N):
        r_now = 0.5*D[nn]

        for mm in np.arange(nn+1, N):
            dy = y[mm]-y[nn]
            Dmn = 0.5*(D[mm]+D[nn])
            if abs(dy) < r_list:
                dx = x[mm]-x[nn]
                # dx = dx - round(dx / Lx) * Lx
                if abs(dx) < r_list:
                    dmn_sq = dx**2 + dy**2
                    if dmn_sq < r_list_sq:
                        VL_list[VL_counter][0] = nn
                        VL_list[VL_counter][1] = mm
                        VL_counter += 1

    return VL_list, VL_counter, x, y

def linear_DM_mass_DiffK_RealBoundX(N, x0, y0, D0, m0, x_left, x_right, y_bot, y_top, k_list, k_type):
    """
    Calculates eigensystem of dynamical matrix for system with real walls at {0, Lx}
    """
    
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        r_now = 0.5*D0[i]
        if y0[i]-y_bot<r_now or y_top-y0[i]<r_now:
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1] + k_list[k_type[i]] / r_now / r_now
        if x0[i]-x_left<r_now or x_right-x0[i]<r_now: # x wall interaction
            M[2*i, 2*i] = M[2*i, 2*i] + k_list[k_type[i]] / r_now / r_now            
        for j in range(i):
            dij = 0.5 * (D0[i] + D0[j])
            dijsq = dij**2
            dx = x0[i] - x0[j]
            # dx = dx - round(dx / Lx) * Lx
            dy = y0[i] - y0[j]
            rijsq = dx**2 + dy**2
            if rijsq < dijsq:
                contactNum += 1  
                k = k_list[(k_type[i] ^ k_type[j]) + np.maximum(k_type[i], k_type[j])]
                # print("k is", k)
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -k * rijmat / rijsq / dijsq
                Mij2 = -k * (1.0 - rij / dij) * (rijmat / rijsq - [[1,0],[0,1]]) / rij / dij
                Mij = Mij1 + Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)

    w,v = np.linalg.eig(M)
    
    return w,v

def hertz_DM_mass_DiffK_RealBoundX(N, x0, y0, D0, m0, x_left, x_right, y_bot, y_top, k_list, k_type):
    """
    Calculates eigensystem of dynamical matrix for system with real walls at {0, Lx}, with hertzian interactions
    """
    
    M = np.zeros((2*N, 2*N))
    contactNum = 0

    for i in range(N):
        r_now = 0.5*D0[i]
        if y0[i]-y_bot<r_now or y_top-y0[i]<r_now: # vertical walls
            M[2*i+1, 2*i+1] = M[2*i+1, 2*i+1] + 1.5*k_list[k_type[i]] / r_now / r_now
        if x0[i]-x_left<r_now or x_right-x0[i]<r_now: # horizontal walls
            M[2*i, 2*i] = M[2*i, 2*i] + 1.5*k_list[k_type[i]] / r_now / r_now            
        for j in range(i):
            dij = 0.5 * (D0[i] + D0[j])
            dijsq = dij**2
            dx = x0[i] - x0[j]
            # dx = dx - round(dx / Lx) * Lx
            dy = y0[i] - y0[j]
            rijsq = dx**2 + dy**2
            if rijsq < dijsq:
                contactNum += 1  
                k = k_list[(k_type[i] ^ k_type[j]) + np.maximum(k_type[i], k_type[j])]
                # print("k is", k)
                rijmat = np.array([[dx*dx, dx*dy], [dx*dy, dy*dy]])
                rij = np.sqrt(rijsq)
                Mij1 = -1.5*k * rijmat / rijsq / dijsq
                Mij2 = -1.5*k * (1.0 - rij / dij)**(3/2) * (rijmat / rijsq - [[1,0],[0,1]]) / rij / dij
                Mij = Mij1 + Mij2
                M[2*i:2*i+2,2*j:2*j+2] = Mij
                M[2*j:2*j+2,2*i:2*i+2] = Mij
                M[2*i:2*i+2,2*i:2*i+2] = M[2*i:2*i+2,2*i:2*i+2] - Mij
                M[2*j:2*j+2,2*j:2*j+2] = M[2*j:2*j+2,2*j:2*j+2] - Mij

    m_sqrt = np.zeros((2*N, 2*N))
    m_inv = np.zeros((2*N, 2*N))
    for i in range(N):
        m_sqrt[2*i, 2*i] = 1/np.sqrt(m0[i])
        m_sqrt[2*i+1, 2*i+1] = 1/np.sqrt(m0[i])
        m_inv[2*i, 2*i] = 1/m0[i]
        m_inv[2*i+1, 2*i+1] = 1/m0[i]

    #M = m_sqrt.dot(M).dot(m_sqrt)

    w,v = np.linalg.eig(M)
    
    return w,v

def hertz_DM_mass_DiffK_RealBoundX_2(N, x0, y0, D0, m0, x_left, x_right, y_bot, y_top, k_list, k_type):
    """
    Calculates eigensystem of dynamical matrix for system with real walls at {0, Lx}, with hertzian interactions
    """
    
    M = np.zeros((N, N))
    contactNum = 0

    for i in range(N):
        r_now = 0.5*D0[i]
        if x0[i]-x_left<r_now: # horizontal walls
            M[i, i] = M[i, i] + 1.5*k_list[k_type[i]]*(1.0 - (x0[i]-x_left)/r_now)**0.5 / r_now / r_now
        if x_right-x0[i]<r_now: # horizontal walls
            M[i, i] = M[i, i] + 1.5*k_list[k_type[i]]*(1.0 - (x_right-x0[i])/r_now)**0.5 / r_now / r_now 
        for j in range(i):
            dij = 0.5 * (D0[i] + D0[j])
            dijsq = dij**2
            dx = x0[i] - x0[j]
            rijsq = dx**2
            if rijsq < dijsq:
                contactNum += 1  
                k = k_list[(k_type[i] ^ k_type[j]) + np.maximum(k_type[i], k_type[j])]
                # print("k is", k)
                rijmat = dx*dx
                rij = np.sqrt(rijsq)
                Mij1 = -1.5*k * (1.0 - rij / dij)**(1/2) * dx / rij / dijsq
                Mij2 = -k * (1.0 - rij / dij)**(3/2) * (dx / rij - 1) / rij / dij
                Mij = Mij1 + Mij2
                M[i, j] = Mij
                M[j, i] = Mij
                M[i, i] = M[i, i] - Mij
                M[j, j] = M[j, j] - Mij


    w,v = np.linalg.eig(M)
    
    return w,v

def ConfigPlot_DiffStiffness_RealX(N, x, y, D, L, m, mark_print, ax, in1, in2, out):
    """
    Plots configurations, same as before, now just with real walls.
    """
    
    m_min = min(m)
    m_max = max(m)
    if (m_min == m_max):
        m_min = 0
    #fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    m_all = []
    for i in range(N):
        x_now = x[i]
        y_now = y[i]
        for k in range(1, 2):
            for l in range(1, 2):                        
                e = Ellipse((x_now, y_now), D[i],D[i],angle=0)
                e.set_edgecolor('k')
                e.set_linewidth(2)
                #if i==in1:
                #    e.set_edgecolor((0, 0, 1))
                #    e.set_linewidth(4)
                #elif i==in2:
                #    e.set_edgecolor((0, 0, 1))
                #    e.set_linewidth(4)
                #elif i==out:
                #    e.set_edgecolor((1, 0, 0))
                #    e.set_linewidth(4)
                ells.append(e)
                m_all.append(m[i])

    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('k')
        
        e.set_alpha(0.2+(m_all[i])*0.5)
        #e.set_alpha(0.2+(m_all[i]-m_min)/(m_max-m_min)*0.3)
 
        i += 1
                
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])

    # draw walls
    ax.plot([0, L[0]], [0, 0], color='black', linewidth=10)
    ax.plot([0, L[0]], [L[1], L[1]], color='black', linewidth=10)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    #plt.show() 
    #if mark_print == 1:
    #    fig.savefig(hn, dpi = 300)
    return ax 
