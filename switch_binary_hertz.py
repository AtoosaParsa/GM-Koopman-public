""" Functions for numerical simulation of granular chains with Hertizian contacts and fixed boundaries
Partially developed by Qikai Wu from The O'Hern Group at Yale University <https://jamming.research.yale.edu/>

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = ['Atoosa Parsa', 'Qikai Wu']
__license__ = 'MIT License'
__version__ = '0.0.3'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from Wall_1D_functions_hertz import hertz_FIRE_RealBoundX_ConstV_DiffK, hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3, hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_6, hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3_c, ConfigPlot_DiffStiffness_RealX

class switch_hertz():
    def evaluate_2(stiffness, damping, compression, timeSteps, dt, initial_x, initial_v):
        # initial_x is the displacement from the equilibrium initial positions
        k1 = 1.
        k2 = 2.3 
        
        n_col = len(stiffness)
        n_row = 1
        N = n_col*n_row
        
        Nt_fire = 1e6
        
        B = damping
        Nt = timeSteps
        
        
        dphi = compression
        d0 = 0.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        k_type = stiffness
        
        # Steepest Descent to get energy minimum
        x_ini, y_ini, p_now = hertz_FIRE_RealBoundX_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
        cont, Ek, Ep, p, x_all, v_all = hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3(k_list, k_type, B, 0, 0, Nt, dt, N, initial_x, x_ini, y_ini, initial_v, D, mass, [Lx, Ly])
    
        return cont, Ek, Ep, p, x_all, v_all

    def evaluate_3(stiffness, damping, compression, timeSteps, dt, initial_x, initial_v):
        # initial_x is the displacement from the equilibrium initial positions
        k1 = 1.
        k2 = 2.3 
        
        n_col = len(stiffness)
        n_row = 1
        N = n_col*n_row
        
        Nt_fire = 1e6
        
        B = damping
        Nt = timeSteps
        
        
        dphi = compression
        d0 = 0.1
        d_ratio = 1.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        k_type = stiffness
        
        # Steepest Descent to get energy minimum
        x_ini, y_ini, p_now = hertz_FIRE_RealBoundX_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)
        

        eigen_freq, eigen_mode, x_all, v_all = hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_6(k_list, k_type, B, 0, 0, Nt, dt, N, x_ini, y_ini, initial_x, initial_v, D, mass, [Lx, Ly])
    
        return eigen_freq, eigen_mode, x_all, v_all

    def evaluate_4(stiffness, damping, compression, timeSteps, dt, initial_x, initial_v, Freq_Vibr1, Amp_Vibr1, Freq_Vibr2, Amp_Vibr2):
        # initial_x is the displacement from the equilibrium initial positions
        k1 = 1.
        k2 = 2.3 
        
        n_col = len(stiffness)
        n_row = 1
        N = n_col*n_row
        
        Nt_fire = 1e6
        
        B = damping
        Nt = timeSteps
        
        
        dphi = compression
        d0 = 0.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        k_type = stiffness
        
        # Steepest Descent to get energy minimum
        x_ini, y_ini, p_now = hertz_FIRE_RealBoundX_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)

        cont, Ek, Ep, p, x_all, v_all, x_in1, vx_in1 = hertz_MD_VibrSP_ConstV_RealBoundX_DiffK_damp_3_c(k_list, k_type, B, 0, 0, Nt, dt, N, initial_x, x_ini, y_ini, initial_v, D, mass, [Lx, Ly], Freq_Vibr1, Amp_Vibr1, Freq_Vibr2, Amp_Vibr2)
    
        return cont, Ek, Ep, p, x_all, v_all, x_in1, vx_in1
    
    def showpacking(stiffness, damping, compression, timeSteps, dt, path=None):
        k1 = 0.1
        k2 = 1.0 
        
        n_col = len(stiffness)
        n_row = 1
        N = n_col*n_row
        
        Nt_fire = 1e6
        
        B = damping
        Nt = timeSteps
        
        
        dphi = compression
        d0 = 0.1
        Lx = d0*n_col
        Ly = (n_row-1)*np.sqrt(3)/2*d0+d0
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        
        phi0 = N*np.pi*d0**2/4/(Lx*Ly)
        d_ini = d0*np.sqrt(1+dphi/phi0)
        D = np.zeros(N)+d_ini
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i_row in range(1, n_row+1):
            for i_col in range(1, n_col+1):
                ind = (i_row-1)*n_col+i_col-1
                x0[ind] = (i_col-1)*d0+0.5*d0
                y0[ind] = (i_row-1)*np.sqrt(3)/2*d0
        y0 = y0+0.5*d0
        
        mass = np.zeros(N) + 1
        k_list = np.array([k1, k2, k1 * k2 / (k1 + k2)])
        k_type = stiffness
        
        # Steepest Descent to get energy minimum
        x_ini, y_ini, p_now = hertz_FIRE_RealBoundX_ConstV_DiffK(Nt_fire, N, x0, y0, D, mass, Lx, Ly, k_list, k_type)

        if path != None:
            ConfigPlot_DiffStiffness_RealX(N, x_ini, y_ini, D, [Lx, Ly], k_list[k_type], 1, path, 0, 0, N-1)    
        return True
