# %% Imports
import pathlib
import sys
import time

import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
from filter_functions import plotting
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_50
from scipy.special import j0, j1

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TOTALWIDTH, PATH, init

LINE_COLORS = list(RWTH_COLORS.values())[1:]
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')
pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else ' (ns$^{-1}$)'
# %% Parameters

# We calculate with hbar == 1
Delta = 0.0   # set detuning to zero
omega_d = 20*2*np.pi
omega_0 = omega_d + Delta
# Phase shift; phi = 0 gives us a rotation about X, phi = pi/2 about Y
phi = 0
# Rabi frequency
Omega_R = 1e-3*2*np.pi
A = np.sqrt(Omega_R**2 - Delta**2)
T = 2*np.pi/omega_d

t = np.linspace(0, T, 101)
dt = np.diff(t)
X, Y, Z = ff.util.paulis[1:]

H_c = [[Z, [omega_0/2]*len(dt), r'$\sigma_{z}$'],
       [X, A*np.cos(omega_d*t[1:] + phi), r'$\sigma_{x}$']]
H_n = [[Z, np.ones_like(dt), r'$\sigma_{z}$'],
       [X, np.ones_like(dt), r'$\sigma_{x}$']]

omega = np.sort(np.geomspace(1e-7, 1e4, 1000).tolist() + [0, Omega_R, omega_d])

# %% Concatenation
tic = []
toc = []

tic.append(time.perf_counter())
X_ATOMIC = ff.PulseSequence(H_c, H_n, dt)
toc.append(time.perf_counter())

tic.append(time.perf_counter())
F1 = X_ATOMIC.get_filter_function(omega)
toc.append(time.perf_counter())

tic.append(time.perf_counter())
NOT_PERIODIC = ff.concatenate_periodic(X_ATOMIC, 10000)
toc.append(time.perf_counter())

ID20 = ff.concatenate_periodic(NOT_PERIODIC, 40)

# %% weak vs strong driving


def get_envelope(A_m, t_r, t_f, t):
    t_p = t[-1] - t_f - t_r
    A = np.full_like(t, A_m)
    with np.errstate(invalid='ignore'):
        A[t <= t_r] = A_m/2*(1 - np.cos(np.pi*t[t <= t_r]/t_r))
        A[t > t_p + t_r] = A_m/2*(1 + np.cos(np.pi*(t[t > t_p + t_r] - t_p - t_r)/t_f))
    A[np.isnan(A)] = 0
    return A


Delta = 0.0   # set detuning to zero
omega_d = 20*2*np.pi
omega_0 = omega_d + Delta
# Drive amplitude
A_m, t_r, t_p, t_f = np.array([36.7054, 0.008, 0.084, 0.008])
A_m = omega_d*0.25

# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.94.032323
Omega_R_STRONG = np.sqrt((omega_d - omega_0*j0(2*A_m/omega_d))**2 +
                         omega_0**2*j1(2*A_m/omega_d)**2)

T = 2*np.pi/Omega_R_STRONG/2
# T += T/100
t_r = 0/omega_d
t_f = 0/omega_d
t_p = T
t = np.linspace(0, t_p+t_r+t_f, 1001)
dt = np.diff(t)
# Paulis
X, Y, Z = ff.util.paulis[1:]
A = get_envelope(A_m, t_r, t_f, t)
# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.115.133601
Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

H_c = [[Z/2, [omega_0]*len(dt), r'$\sigma_{z}$'],
       [X/2, 2*A[1:]*np.cos(omega_d*t[1:] + phi), r'$\sigma_{x}$']]
H_n = [[Z/2, np.ones_like(dt), r'$\sigma_{z}$'],
       [X/2, np.ones_like(dt), r'$\sigma_{x}$']]

omega_STRONG = np.sort(np.geomspace(1e-2, 1.2e3, 1000).tolist()
                       + [0, Omega_R_STRONG, omega_d - Omega_R_STRONG, omega_d,
                          omega_d + Omega_R_STRONG, 2*omega_d - Omega_R_STRONG,
                          2*omega_d, 2*omega_d + Omega_R_STRONG])

NOT_STRONG = ff.PulseSequence(H_c, H_n, dt)
NOT_STRONG.cache_filter_function(omega_STRONG)
ID20_STRONG = ff.concatenate_periodic(NOT_STRONG, 40)

plotting.plot_bloch_vector_evolution(NOT_STRONG)
print('Strong driving infidelity: ',
      1 - np.abs(np.trace(NOT_STRONG.total_propagator.conj().T @ X)/2))

# %%% plot columns
fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(TOTALWIDTH, 2))

for ax, pulse, Omega in zip(axes, (ID20, ID20_STRONG), (Omega_R, Omega_R_STRONG)):
    ax.set_prop_cycle(color=LINE_COLORS)
    ax.loglog(pulse.omega, np.diagonal(pulse.get_filter_function(pulse.omega)).real)

    ax.axvline(Omega, ls=':', color=RWTH_COLORS_50['black'],
               label=r'$\Omega_R$', zorder=0)
    ax.axvline(omega_d, ls='--', color=RWTH_COLORS_50['black'],
               label=r'$\omega_0$', zorder=0)
    ax.xaxis.minorticks_off()
    ax.set_xlabel(r'$\omega$' + pernanosecond)

ax.axvline(2*omega_d, ls='-.', color=RWTH_COLORS_50['black'], label=r'$2\Omega_R$', zorder=0)
ax.legend(pulse.n_oper_identifiers, frameon=False, loc='lower left', ncols=2)

axes[0].set_ylabel(r'$\mathcal{F}_\alpha(\omega)$')
axes[0].set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4])

fig.savefig(SAVE_PATH / 'rabi_driving_weak_vs_strong.pdf')
