# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:18:02 2019

@author: Tobias Hangleiter, tobias.hangleiter@rwth-aachen.de
"""
from math import ceil
from pathlib import Path
from typing import Callable, Tuple

import qutil
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft, ndarray
from numpy.random import randn
from scipy.linalg import expm
from scipy.optimize import curve_fit

plt.style.use('thesis')
# Hotfix latex preamble
for key in ('text.latex.preamble', 'pgf.preamble'):
    plt.rcParams.update({key: '\n'.join(plt.rcParams.get(key).split('|'))})

golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (5.65071, 5.65071*golden_ratio)
exts = ('pdf', 'eps', 'pgf')


def MC_ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return qutil.linalg.abs2(np.einsum('...ll', gates @ target.conj().T)/d)


def MC_avg_gate_fidelity(gates, target):
    """Calculate the average gate fidelity"""
    d = gates.shape[-1]
    return (d*MC_ent_fidelity(gates, target) + 1)/(d + 1)


def rand_herm(d: int, n: int = 1) -> np.ndarray:
    """n random Hermitian matrices of dimension d"""
    A = randn(n, d, d) + 1j*randn(n, d, d)
    return (A + A.conj().transpose([0, 2, 1])).squeeze()


def rand_unit(d: int, n: int = 1) -> np.ndarray:
    """n random unitary matrices of dimension d"""
    H = rand_herm(d, n)
    if n == 1:
        return expm(1j*H)
    else:
        return np.array([expm(1j*h) for h in H])


def white_noise(S0: float, f_max: float, shape: Tuple[int]) -> ndarray:
    """Generate white noise with variance S0*f_max/2."""
    var = S0*f_max/2
    return np.sqrt(var)*randn(*shape)


def fast_colored_noise(spectral_density: Callable, f_max: float, f_min: float,
                       shape: Tuple[int]) -> ndarray:
    """
    Generate fast noise with frequencies between f_min and f_max with the noise
    spectrum like spectral_density.

    Validate like so:
    >>> from scipy import signal
    >>> traces = fast_noise(spectral_density, f_max, f_min, shape)
    >>> f, S = signal.periodogram(traces, f_max, axis=-1)
    >>> plt.loglog(f, spectral_density(f))
    >>> plt.loglog(f, S.mean(axis=0))
    """
    dt = 1/f_max
    f_N = f_max/2
    S0 = 1/f_N
    R = 2**ceil(-np.log2(f_min*dt))

    delta_white = randn(*shape, R)
    # delta_white = randn(*shape)
    delta_white_ft = fft.rfft(delta_white, axis=-1)
    # Only positive frequencies since FFT is real and therefore symmetric
    f = np.linspace(f_min, f_max/2, R // 2 + 1)
    delta_colored = fft.irfft(delta_white_ft*np.sqrt(spectral_density(f)/S0),
                              axis=-1)

    return delta_colored


def monte_carlo_gate(opers: ndarray, coeffs: ndarray,
                     N_MC: int, S0: float, f_max: float, dt: ndarray):
    """
    Return N_MC gates with ceil(dt.min()*f_max) noise steps per time step of
    the gate.
    """
    N_n = ceil(dt.min()*f_max)
    if N_n*dt.size % 2:
        N_n += 1

    dt = np.repeat(dt, N_n)/N_n
    coeffs = np.repeat(coeffs, N_n, axis=1)

    coeffs_delta = (
        coeffs.reshape(1, *coeffs.shape) +
        white_noise(S0, f_max, (N_MC, dt.size)).reshape(N_MC, 1, dt.size)
    )

    H = np.einsum('ijk,mil->mljk', opers, coeffs_delta)
    HD, HV = np.linalg.eigh(H)
    P = np.einsum('mlij,mjl,mlkj->mlik', HV,
                  np.exp(-1j*np.asarray(dt)*HD.swapaxes(-1, -2)), HV.conj())
    Q = qutil.linalg.mdot(P[:, ::-1, ...], axis=1)
    return Q


def f(x, *args):
    a, b = args
    return a*x**b


# %%
folder = 'MC_vs_FF_benchmark'
sha = '3371534'
now = '20190730-114157'

thesis_path = Path('/home/tobias/Physik/Master/Thesis')
# thesis_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis')
# thesis_path = Path('Z:/MA/')
dpath = thesis_path / 'data' / folder / sha[:7] / now
spath = thesis_path / 'thesis' / 'img'

fname = f'benchmark_MC_vs_FF'

with np.load(dpath / 'benchmark_MC_vs_FF.npz') as arch:
    dims = arch['dims']
    dt_MC = arch['dt_MC']
    dt_FF = arch['dt_FF']
    infids_MC = arch['infids_MC']
    infids_FF = arch['infids_FF']

# %%
d_max = 120
dims = np.arange(2, 120+1)
n_alpha = 3
n_dt = 1
n_MC = 100
n_omega = 500

i = 0   # len(dims) // 2
popt_FF, pcov_MC = curve_fit(f, dims[i:], dt_FF[i:], p0=[1, 3], maxfev=10**5)
popt_MC, pcov_FF = curve_fit(f, dims[i:], dt_MC[i:], p0=[1, 2], maxfev=10**5)

# %% loglog
# dims_plot = np.arange(0, d_max + dims[1] - dims[0])
# fig, ax = plt.subplots(figsize=figsize_narrow)
# ax.loglog(dims, dt_FF, '.', markersize=1.5)
# ax.loglog(dims_plot, f(dims_plot, *popt_FF),
#           color='tab:blue', label=rf'FF: $\mathcal{{O}}(d^{{{popt_FF[1]:.2f}}})$')
# ax.loglog(dims, dt_MC, '.', markersize=1.5)
# ax.loglog(dims_plot, f(dims_plot, *popt_MC),
#           color='tab:orange', label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')
# ax.legend(framealpha=1, loc='lower right')
# ax.grid()
# ax.set_xlim(1, d_max)
# ax.set_xlabel('$d$')
# ax.set_ylabel('$t$ (s)')

# fig.tight_layout(h_pad=0)
# for ext in ('pgf', 'pdf', 'eps'):
#     fig.savefig(spath / ext / (fname + '_loglog.' + ext))
# # %% linear
# dims_plot = np.arange(0, d_max + dims[1] - dims[0])
# fig, ax = plt.subplots(figsize=figsize_narrow)
# ax.plot(dims, dt_FF, '.', markersize=1.5)
# ax.plot(dims_plot, f(dims_plot, *popt_FF),
#         color='tab:blue', label=rf'FF: $\mathcal{{O}}(d^{{{popt_FF[1]:.2f}}})$')
# ax.plot(dims, dt_MC, '.', markersize=1.5)
# ax.plot(dims_plot, f(dims_plot, *popt_MC),
#         color='tab:orange', label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')
# ax.legend(framealpha=1)
# ax.grid()
# ax.set_xlim(0, d_max)
# ax.set_xlabel('$d$')
# ax.set_ylabel('$t$ (s)')

# fig.tight_layout(h_pad=0)
# for ext in ('pgf', 'pdf', 'eps'):
#     fig.savefig(spath / ext / (fname + '.' + ext))

# # %% loglog inset in linear plot
# dims_plot = np.arange(0, d_max + dims[1] - dims[0])
# fig, ax = plt.subplots(figsize=figsize_narrow)
# ax.plot(dims, dt_FF, '.', markersize=1.5, color='tab:blue')
# ax.plot(dims_plot, f(dims_plot, *popt_FF), color='tab:blue',
#         label=rf'FF: $\mathcal{{O}}(d^{{{popt_FF[1]:.2f}}})$')
# ax.plot(dims, dt_MC, '.', markersize=1.5, color='tab:orange')
# ax.plot(dims_plot, f(dims_plot, *popt_MC), color='tab:orange',
#         label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')

# ax.set_xlim(0, d_max)
# ax.set_xlabel('$d$')
# ax.set_ylabel('$t$ (s)')

# ins_ax = inset_axes(ax, 1, 1)
# inset_position = InsetPosition(ax, [0.125, 0.425, 0.5, 0.5])
# ins_ax.set_axes_locator(inset_position)

# ins_ax.loglog(dims, dt_FF, '.', markersize=1, color='tab:blue')
# ins_ax.loglog(dims_plot, f(dims_plot, *popt_FF), color='tab:blue')
# ins_ax.loglog(dims, dt_MC, '.', markersize=1, color='tab:orange')
# ins_ax.loglog(dims_plot, f(dims_plot, *popt_MC), color='tab:orange')

# ins_ax.set_xlim(0, d_max)
# ins_ax.tick_params(direction='in', which='both', labelsize=6)
# ins_ax.spines['right'].set_visible(False)
# ins_ax.spines['top'].set_visible(False)
# ins_ax.patch.set_alpha(0)

# ax.legend(framealpha=1, bbox_to_anchor=(0., 1.04, 1., .104),
#           mode="expand", loc="lower left", ncol=2, borderaxespad=0.)

# fig.tight_layout()
# for ext in ('pgf', 'pdf', 'eps'):
#     fig.savefig(spath / ext / (fname + '_loglog-inset.' + ext))

# %% loglog linear subplots
dims_plot = np.arange(0, d_max + dims[1] - dims[0])
fig, ax = plt.subplots(1, 2, figsize=(figsize_wide[0], figsize_wide[0]/3))

ax[0].plot(dims, dt_FF, '.', markersize=1.5, color='tab:blue')
ax[0].plot(dims_plot, f(dims_plot, *popt_FF), color='tab:blue',
           label=rf'FF: $\mathcal{{O}}(d^{{{popt_FF[1]:.2f}}})$')
ax[0].plot(dims, dt_MC, '.', markersize=1.5, color='tab:orange')
ax[0].plot(dims_plot, f(dims_plot, *popt_MC), color='tab:orange',
           label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')

ax[0].set_xlim(0, d_max)
ax[0].set_xlabel('$d$')
ax[0].set_ylabel('$t$ (s)')
ax[0].grid()
ax[0].legend(framealpha=1)
ax[0].text(0.875, 0.1, '(a)', transform=ax[0].transAxes, fontsize=10)

ax[1].loglog(dims, dt_FF, '.', markersize=1.5, color='tab:blue')
ax[1].loglog(dims_plot, f(dims_plot, *popt_FF), color='tab:blue')
ax[1].loglog(dims, dt_MC, '.', markersize=1.5, color='tab:orange')
ax[1].loglog(dims_plot, f(dims_plot, *popt_MC), color='tab:orange')

ax[1].set_xlim(1, d_max)
ax[1].set_ylim(1e-3, 1e3)
ax[1].set_xlabel('$d$')
# ax[1].set_ylabel('$t$ (s)')
ax[1].grid()
ax[1].text(0.875, 0.1, '(b)', transform=ax[1].transAxes, fontsize=10)

fig.tight_layout()
for ext in exts:
    fig.savefig(spath / ext / (fname + '_linear-loglog.' + ext))
