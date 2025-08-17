# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:40:19 2020

@author: Tobias Hangleiter
"""


# %% weak vs strong driving
# We calculate with hbar == 1


def get_envelope(A_m, t_r, t_f, t):
    t_p = t[-1] - t_f - t_r
    A = np.full_like(t, A_m)
    A[t <= t_r] = A_m/2*(1 - np.cos(np.pi*t[t <= t_r]/t_r))
    A[t > t_p + t_r] = A_m/2*(1 + np.cos(np.pi*(t[t > t_p + t_r] - t_p - t_r)/t_f))
    return A


Delta = 0.0   # set detuning to zero
omega_d = 20e9*2*np.pi
omega_d = 20*2*np.pi
omega_d = 2*np.pi*2.288
omega_0 = omega_d + Delta
# Phase shift; phi = 0 gives us a rotation about Y, phi = pi/2 about X
phi = np.pi/2
# Drive amplitude
A_m = omega_d*0.270*1.005

# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.115.133601
# Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.94.032323
Omega_R_STRONG = np.sqrt((omega_d - omega_0*j0(2*A_m/omega_d))**2 +
                         omega_0**2*j1(2*A_m/omega_d)**2)

T = 2*np.pi/Omega_R_STRONG/2
# T += T/100
t_r = 2.6/omega_d*0.99
t_f = 1/omega_d*1.022
t_p = 4.27/omega_d*0.997
t = np.linspace(0, t_p+t_r+t_f, 1001)
dt = np.diff(t)
# Paulis
X, Y, Z = ff.util.P_np[1:]
A = get_envelope(A_m, t_r, t_f, t)
Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

H_c = [[-Z/2, [omega_0-omega_d]*len(dt), r'$F_{zz}$'],
       [Y/2, 2*A[1:]*np.cos(omega_d*t[1:] + phi)*np.sin(omega_d*t[1:]), r'$F_{yy}$'],
       [X/2, 2*A[1:]*np.cos(omega_d*t[1:] + phi)*np.cos(omega_d*t[1:]), r'$F_{xx}$']]
H_n = [[Z/2, np.ones_like(dt), r'$F_{zz}$'],
       [X/2, np.ones_like(dt), r'$F_{xx}$']]

omega_STRONG = np.geomspace(1e-4*Omega_R_STRONG, 5e1*omega_0, 500)
omega_STRONG = np.geomspace(1e-2, 1e4, 504)
# omega_STRONG = omega

NOT_STRONG = ff.PulseSequence(H_c, H_n, dt)
NOT_STRONG.cache_filter_function(omega_STRONG)
ID20_STRONG = ff.concatenate_periodic(NOT_STRONG, 60)

ff.plot_bloch_vector_evolution(NOT_STRONG)
Y2 = np.array([[-1-1j, 1+1j],[-1+1j, -1+1j]])/2
X2 = np.array([[-1-1j, 1+1j],[-1+1j, -1+1j]])/2
print(1 - np.abs(np.trace(NOT_STRONG.total_Q.conj().T @ Y2)/2))

# %% optimize
def fun(x):
    A_m, t_r, t_p, t_f = x
    Delta = 0.0   # set detuning to zero
    omega_d = 20e9*2*np.pi
    omega_d = 20*2*np.pi
    # omega_d = 2*np.pi*2.288
    omega_0 = omega_d + Delta
    # Phase shift; phi = 0 gives us a rotation about Y, phi = pi/2 about X
    phi = 0
    # Drive amplitude

    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.115.133601
    # Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.94.032323
    Omega_R_STRONG = np.sqrt((omega_d - omega_0*j0(2*A_m/omega_d))**2 +
                             omega_0**2*j1(2*A_m/omega_d)**2)

    t = np.linspace(0, t_p+t_r+t_f, 1001)
    dt = np.diff(t)
    # Paulis
    X, Y, Z = ff.util.P_np[1:]
    A = get_envelope(A_m, t_r, t_f, t)
    Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

    H_c = [[Z/2, [omega_0]*len(dt), r'$F_{zz}$'],
           [X/2, 2*A[1:]*np.cos(omega_d*t[1:] + phi), r'$F_{xx}$']]
    H_n = [[Z/2, np.ones_like(dt), r'$F_{zz}$'],
           [X/2, np.ones_like(dt), r'$F_{xx}$']]

    omega_STRONG = np.geomspace(1e-4*Omega_R_STRONG, 5e1*omega_0, 500)
    omega_STRONG = np.geomspace(1e-2, 1e4, 1004)
    # omega_STRONG = omega

    NOT_STRONG = ff.PulseSequence(H_c, H_n, dt)
    return (1 - np.abs(np.trace(NOT_STRONG.total_Q.conj().T @ qt.sigmax().full())/2))

from scipy.optimize import minimize
minimize(fun, [omega_d/4, 0, T, 0], bounds=[(omega_d*0.2, omega_d*10), (1/omega_d, 10/omega_d), (T/10, None), (1/omega_d, 10/omega_d)], method='SLSQP')
