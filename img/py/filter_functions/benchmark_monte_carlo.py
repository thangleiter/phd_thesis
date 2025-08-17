# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:18:02 2019

@author: Tobias Hangleiter, tobias.hangleiter@rwth-aachen.de
"""
from math import ceil
from pathlib import Path
import datetime
import git
import shutil
from time import perf_counter
from typing import Callable, Tuple

from qutil import linalg as qla
import filter_functions as ff
from filter_functions import util
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler
from numpy import ndarray, random
from numpy import linalg as nla
from scipy import linalg as sla
from scipy.optimize import curve_fit

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition)

plt.style.use('publication')
# Hotfix latex preamble
for key in ('text.latex.preamble', 'pgf.preamble'):
    plt.rcParams.update({key: '\n'.join(plt.rcParams.get(key).split('|'))})

golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
ls_cycle = plt.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=('-','-.','--',':',))
ms_cycle = plt.rcParams['axes.prop_cycle'][:4] + cycler(marker=('s', 'D', 'v', '.'))
rng = random.RandomState()


def MC_ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return qla.abs2(np.einsum('...ll', gates @ target.conj().T)/d)


def MC_avg_gate_fidelity(gates, target):
    """Calculate the average gate fidelity"""
    d = gates.shape[-1]
    return (d*MC_ent_fidelity(gates, target) + 1)/(d + 1)


def rand_herm(d: int, n: int = 1) -> np.ndarray:
    """n random Hermitian matrices of dimension d"""
    A = rng.randn(n, d, d) + 1j*rng.randn(n, d, d)
    return (A + A.conj().transpose([0, 2, 1]))/2


def rand_herm_traceless(d: int, n: int = 1) -> np.ndarray:
    """n random traceless Hermitian matrices of dimension d"""
    A = rand_herm(d, n).transpose()
    A -= A.trace(axis1=0, axis2=1)/d
    return A.transpose()


def rand_unit(d: int, n: int = 1) -> np.ndarray:
    """n random unitary matrices of dimension d"""
    H = rand_herm(d, n)
    if n == 1:
        return sla.expm(1j*H)
    else:
        return np.array([sla.expm(1j*h) for h in H])


def white_noise(S0: float, f_max: float, shape: Tuple[int]) -> ndarray:
    """Generate white noise with variance S0*f_max/2."""
    var = S0*f_max/2  # S0 * f_nyquist
    return np.sqrt(var)*rng.randn(*shape)


def fast_colored_noise(spectral_density: Callable, dt: ndarray, f_max: float,
                       f_min: float, output_shape: Tuple,
                       r_power_of_two=False) -> ndarray:
    """
    Generates noise traces of arbitrary colored noise.

    Use this code for validation:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> traces = fast_colored_noise(spectral_density, dt, f_max, f_min,
    >>>                             output_shape)
    >>> f, S = signal.welch(traces, f_max, axis=-1)
    >>> plt.loglog(f, spectral_density(f))
    >>> plt.loglog(f, S.mean(axis=0))

    Parameters
    ----------
    spectral_density: Callable
        The spectral density as function of frequency.

    dt: ndarray
        Original time trace delta

    f_max: float
        Highest frequency to resolve

    f_min: float
        Smallest frequency to resolve

    output_shape: Tuple[int]
        Shape of the noise traces to be returned.

    r_power_of_two: bool
        If true, then n_samples is rounded downwards to the next power of 2 for
        an efficient fast fourier transform.

    Returns
    -------
    delta_colored: np.ndarray, shape(output_shape, sliced_n_samples)
        Where sliced_n_samples is n_samples or the largest power of 2 smaller
        than n_samples if r_power_of_two is true.

    """
    # f_max = fast_multiple / dt
    fast_multiple = ceil(f_max*dt.min())
    dt_fast = dt.min()/fast_multiple

    f_nyquist = 1/2/dt_fast

    # f_min = 1 / T / slow_multiple
    slow_multiple = ceil(1/(f_min*dt.sum()))

    sliced_n_samples = int(len(dt)*fast_multiple*slow_multiple)

    add_one = sliced_n_samples % 2

    delta_white = np.random.randn(*output_shape, sliced_n_samples + add_one)
    delta_white_ft = np.fft.rfft(delta_white, axis=-1)
    # Only positive frequencies since FFT is real and therefore symmetric
    f = np.linspace(0, f_nyquist, (sliced_n_samples + add_one) // 2 + 1)
    f[1:] = spectral_density(f[1:])
    f[0] = 0
    delta_colored = np.fft.irfft(delta_white_ft * np.sqrt(f * f_nyquist),
                                 axis=-1)
    # the ifft takes r//2 + 1 inputs to generate r outputs

    if add_one:
        delta_colored = delta_colored[..., :-1]

    if slow_multiple != 1:
        s = delta_colored.shape[-1]
        idx = np.arange(s // slow_multiple) + s // 2 - s // slow_multiple // 2
        delta_colored = delta_colored[..., idx]

    return delta_colored


def monte_carlo_gate(N_MC: int, S0: float, f_min: float, f_max: float,
                     dt: ndarray, T: float, alpha: float, c_opers: ndarray,
                     c_coeffs: ndarray, n_opers: ndarray, n_coeffs: ndarray,
                     loop: bool = True):

    def evolution(H, dt):
        HD, HV = nla.eigh(H)

        P = np.einsum('...lij,...jl,...lkj->...lik',
                      HV, util.cexp(-np.asarray(dt)*HD.swapaxes(-1, -2)),
                      HV.conj())

        return qla.mdot(np.flip(P, axis=-3), axis=-3)

    if alpha == 0.0:
        N_n = ceil(dt.min()*f_max)
        if N_n*dt.size % 2:
            N_n += 1
    else:
        N_n = fast_colored_noise(lambda f: S0/f**alpha, dt, f_max, f_min,
                                 (1,)).shape[-1] // dt.size

    a = np.repeat(c_coeffs, N_n, axis=1)
    b = np.repeat(n_coeffs, N_n, axis=1)
    dt_fast = np.repeat(dt, N_n)/N_n

    if loop:
        U_tot = np.empty((N_MC, *c_opers.shape[-2:]), dtype=complex)
        for n in range(N_MC):
            if alpha == 0.0:
                noise = white_noise(S0, f_max, b.shape)
            else:
                noise = fast_colored_noise(lambda f: S0/f**alpha,
                                           dt, f_max, f_min,
                                           n_coeffs.shape[:1])

            H_c = np.tensordot(a, c_opers, axes=[0, 0])
            H_n = np.tensordot(b*noise, n_opers, axes=[0, 0])
            H = H_c + H_n
            U_tot[n] = evolution(H, dt_fast)

        return U_tot
    else:
        noise = white_noise(S0, f_max, (N_MC, *b.shape))
        H_c = np.tensordot(a, c_opers, axes=[0, 0])
        H_n = np.tensordot(b*noise, n_opers, axes=[-2, 0])
        H = H_c + H_n

        return evolution(H, dt_fast)


# %% Run benchmark
# d_max = 120
# dims = np.arange(2, 120+1, 2)
# n_alpha = 3
# n_dt = 1
# n_MC = 100
# n_omega = 500

# tic_MC = []
# toc_MC = []
# tic_FF_H = []
# toc_FF_H = []
# tic_FF_L = []
# toc_FF_L = []
# infids_MC = []
# infids_FF_H = []
# infids_FF_L = []
# t_start = perf_counter()
# print(f'Dim.\t\tMC\t\t\t\tFF (H)\t\t\tFF (L)\t\t\tTotal elapsed')
# print(73*'-')
# for d in dims:
#     opers = rand_herm(d, n_alpha)
#     coeffs = rng.randn(n_alpha, n_dt)
#     dt = np.abs(rng.randn())*np.ones(n_dt)
#     T = dt.sum()
#     f_min = 1e-1/T
#     f_max = 1e2/T
#     S0 = abs(rng.randn())/1e4

#     pulse = ff.PulseSequence(list(zip(opers, coeffs)),
#                              list(zip(opers, np.ones_like(coeffs))),
#                              dt)
#     omega = np.geomspace(f_min, f_max, n_omega // 2)*2*np.pi
#     S = S0*np.ones_like(omega)
#     S, omega = ff.util.symmetrize_spectrum(S, omega)

#     tic_FF_L.append(perf_counter())
#     R = ff.numeric.calculate_control_matrix_from_scratch(
#         pulse.HD, pulse.HV, pulse.Q, omega, pulse.basis, pulse.n_opers,
#         pulse.n_coeffs, pulse.dt, pulse.t, False
#     )
#     pulse.cache_filter_function(omega, R=R)
#     infids_FF_L.append(1 - ff.infidelity(pulse, S, omega).sum())
#     toc_FF_L.append(perf_counter())

#     pulse.cleanup('all')

#     tic_FF_H.append(perf_counter())
#     B = ff.numeric.calculate_noise_operators_from_scratch(
#         pulse.HD, pulse.HV, pulse.Q, omega, pulse.basis, pulse.n_opers,
#         pulse.n_coeffs, pulse.dt, pulse.t, False
#     )
#     pulse.cache_filter_function(omega,
#                                 F=np.einsum('oaji,obji->abo', B.conj(), B))
#     infids_FF_H.append(1 - ff.infidelity(pulse, S, omega).sum())
#     toc_FF_H.append(perf_counter())

#     tic_MC.append(perf_counter())
#     MC_propagators = monte_carlo_gate(n_MC, S0, f_min, f_max, dt, T, 0.0,
#                                       opers, coeffs, opers,
#                                       np.ones_like(coeffs), loop=False)
#     infids_MC.append(MC_ent_fidelity(MC_propagators, pulse.total_Q).mean())
#     toc_MC.append(perf_counter())

#     print(f'd = {d:3d}\t\t'
#           f'{toc_MC[-1]-tic_MC[-1]:.1e} s\t\t'
#           f'{toc_FF_H[-1]-tic_FF_H[-1]:.1e} s\t\t'
#           f'{toc_FF_L[-1]-tic_FF_L[-1]:.1e} s\t\t'
#           f'{perf_counter() - t_start:.1e} s')

# dt_MC = np.array(toc_MC) - np.array(tic_MC)
# dt_FF_H = np.array(toc_FF_H) - np.array(tic_FF_H)
# dt_FF_L = np.array(toc_FF_L) - np.array(tic_FF_L)
# %% Save data
# repo = git.Repo(Path(ff.__file__).parent.parent)
# sha = repo.head.object.hexsha
# now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# folder = 'MC_vs_FF_benchmark'
# dpath = Path('Y:/GaAs/Hangleiter/benchmark/data') / folder / sha[:7] / now
# dpath.mkdir(parents=True, exist_ok=True)
# fname = Path(__file__).name
# shutil.copyfile(__file__, dpath / fname)

# pname = ('efficient_calculation_of_generalized_filter_functions_' +
#          'for_sequences_of_quantum_gates')
# spath = Path(r'Z:/Publication/') / pname / 'img'
# spath.mkdir(parents=True, exist_ok=True)
# fname = f'benchmark_MC_vs_FF'

# np.savez(dpath / fname, dims=dims,
#          dt_FF_H=dt_FF_H, dt_FF_L=dt_FF_L, dt_MC=dt_MC,
#          infids_FF_H=infids_FF_H, infids_FF_L=infids_FF_L,
#          infids_MC=infids_MC)

# %% Load data
folder = 'MC_vs_FF_benchmark'
sha = 'c2ddb9e'
now = '20200526-153408'

# dpath = Path('/home/tobias/Janeway/Common/GaAs/Hangleiter/benchmark/data') / folder / sha[:7] / now
dpath = Path('Y:/GaAs/Hangleiter/benchmark/data') / folder / sha[:7] / now
# paper_path = Path('/home/tobias/Physik/Publication/efficient_calculation_of_generalized_filter_functions')
paper_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Publication/efficient_calculation_of_generalized_filter_functions')
# paper_path = Path('Z:/Publication/efficient_calculation_of_generalized_filter_functions')
spath = paper_path / 'img'

fname = f'benchmark_MC_vs_FF'

with np.load(dpath / 'benchmark_MC_vs_FF.npz') as arch:
    dims = arch['dims']
    dt_MC = arch['dt_MC']
    dt_FF_H = arch['dt_FF_H']
    dt_FF_L = arch['dt_FF_L']
    infids_MC = arch['infids_MC']
    infids_FF_H = arch['infids_FF_H']
    infids_FF_L = arch['infids_FF_L']

d_max = 120
# %% Fit


def f(x, *args):
    a, b = args
    return a*x**b


i = 0   # len(dims) // 2
popt_FF_H, _ = curve_fit(f, dims[i:], dt_FF_H[i:], p0=[1, 3], maxfev=10**5)
popt_FF_L, _ = curve_fit(f, dims[i:], dt_FF_L[i:], p0=[1, 3], maxfev=10**5)
popt_MC, _ = curve_fit(f, dims[i:], dt_MC[i:], p0=[1, 2], maxfev=10**5)


# %% Plot
# %%% loglog
dims_plot = np.arange(1, d_max + dims[1] - dims[0], d_max // dt_MC.size)
fig, ax = plt.subplots(figsize=figsize_narrow)
ax.loglog(dims, dt_MC, '.', markersize=1.5)
ax.loglog(dims_plot, f(dims_plot, *popt_MC), color='tab:blue',
          label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')
ax.loglog(dims, dt_FF_H, '.', markersize=1.5)
ax.loglog(dims_plot, f(dims_plot, *popt_FF_H), color='tab:orange',
          label=rf'FF ($\mathscr{{H}}$): $\mathcal{{O}}(d^{{{popt_FF_H[1]:.2f}}})$')
ax.loglog(dims, dt_FF_L, '.', markersize=1.5)
ax.loglog(dims_plot, f(dims_plot, *popt_FF_L), color='tab:green',
          label=rf'FF ($\mathscr{{L}}$): $\mathcal{{O}}(d^{{{popt_FF_L[1]:.2f}}})$')
ax.legend(framealpha=1, loc='lower right')
ax.grid()
ax.set_xlim(1, d_max)
ax.set_xlabel('$d$')
ax.set_ylabel('$t$ (s)')

fig.tight_layout(h_pad=0)
fig.savefig(spath / (fname + '_loglog.eps'), dpi=600)
fig.savefig(spath / (fname + '_loglog.pdf'), dpi=600)
# %%% linear
dims_plot = np.arange(0, d_max + dims[1] - dims[0])
fig, ax = plt.subplots(figsize=figsize_narrow)
ax.plot(dims, dt_MC, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_MC), color='tab:blue',
        label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')
ax.plot(dims, dt_FF_H, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_FF_H), color='tab:orange',
        label=rf'FF ($\mathscr{{H}}$): $\mathcal{{O}}(d^{{{popt_FF_H[1]:.2f}}})$')
ax.plot(dims, dt_FF_L, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_FF_L),
        color='tab:green', label=rf'FF ($\mathscr{{L}}$): $\mathcal{{O}}(d^{{{popt_FF_L[1]:.2f}}})$')
ax.legend(framealpha=1)
ax.grid()
ax.set_xlim(0, d_max)
ax.set_xlabel('$d$')
ax.set_ylabel('$t$ (s)')

fig.tight_layout(h_pad=0)
fig.savefig(spath / (fname + '.eps'), dpi=600)
fig.savefig(spath / (fname + '.pdf'), dpi=600)

# %%% loglog inset in linear plot
dims_plot = np.arange(0, d_max + dims[1] - dims[0])
fig, ax = plt.subplots(figsize=figsize_narrow)
ax.plot(dims, dt_MC, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_MC), color='tab:blue',
        label=rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$')
ax.plot(dims, dt_FF_H, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_FF_H), color='tab:orange',
        label=rf'FF ($\mathscr{{H}}$): $\mathcal{{O}}(d^{{{popt_FF_H[1]:.2f}}})$')
ax.plot(dims, dt_FF_L, '.', markersize=1.5)
ax.plot(dims_plot, f(dims_plot, *popt_FF_L), color='tab:green',
        label=rf'FF ($\mathscr{{L}}$): $\mathcal{{O}}(d^{{{popt_FF_L[1]:.2f}}})$')

ax.set_xlim(0, d_max)
ax.set_xlabel('$d$')
ax.set_ylabel('$t$ (s)')

ins_ax = inset_axes(ax, 1, 1)
inset_position = InsetPosition(ax, [0.15, 0.45, 0.5, 0.5])
ins_ax.set_axes_locator(inset_position)

dims_plot = np.arange(1, d_max + dims[1] - dims[0])
ins_ax.loglog(dims, dt_MC, '.', markersize=0.5, color='tab:blue')
ins_ax.loglog(dims_plot, f(dims_plot, *popt_MC), linewidth=0.5, color='tab:blue')
ins_ax.loglog(dims, dt_FF_H, '.', markersize=0.5, color='tab:orange')
ins_ax.loglog(dims_plot, f(dims_plot, *popt_FF_H), linewidth=0.5, color='tab:orange')
ins_ax.loglog(dims, dt_FF_L, '.', markersize=0.5, color='tab:green')
ins_ax.loglog(dims_plot, f(dims_plot, *popt_FF_L), linewidth=0.5, color='tab:green')

ins_ax.set_xlim(1, d_max)
ins_ax.set_ylim(1e-3)
ins_ax.tick_params(direction='out', which='both', labelsize=6)
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

# ax.legend(framealpha=1, bbox_to_anchor=(0., 1.04, 1., .104),
ax.legend(framealpha=0, bbox_to_anchor=(1.04, 0.0, 0.5, 1.0),
          mode="expand", loc="lower left", ncol=1, borderaxespad=0.)

fig.tight_layout()
fig.savefig(spath / (fname + '_loglog-inset.eps'), dpi=600)
fig.savefig(spath / (fname + '_loglog-inset.pdf'), dpi=600)

plt.close('all')

# %%% linear inset in loglog plot
dims_plot = np.arange(1, d_max + dims[1] - dims[0])
fig, ax = plt.subplots(figsize=figsize_narrow)
ax.set_prop_cycle(ls_cycle)

for popt, dt, label, ms_props, ls_props in zip([popt_FF_L, popt_FF_H, popt_MC],
                                                [dt_FF_L, dt_FF_H, dt_MC],
                                                [rf'FF ($\mathscr{{L}}$): $\mathcal{{O}}(d^{{{popt_FF_L[1]:.2f}}})$',
                                                 rf'FF ($\mathscr{{H}}$): $\mathcal{{O}}(d^{{{popt_FF_H[1]:.2f}}})$',
                                                 rf'MC: $\mathcal{{O}}(d^{{{popt_MC[1]:.2f}}})$'],
                                                ms_cycle,
                                                ls_cycle):
    ax.loglog(dims_plot, f(dims_plot, *popt), **ls_props)
    ax.loglog(dims, dt, markersize=2, label=label, linestyle='None', **ms_props)

ax.set_xlim(1, d_max)
ax.set_ylim(1e-5, 1e3)
ax.set_xlabel('$d$')
ax.set_ylabel('$t$ (s)')

ins_ax = ax.inset_axes([0.45, 0.13, 0.45, 0.45])
ins_ax2 = ins_ax.twinx()

dims_plot = np.arange(0, d_max + dims[1] - dims[0])

for popt, dt, ms_props, ls_props in zip([popt_FF_L, popt_FF_H, popt_MC],
                                        [dt_FF_L, dt_FF_H, dt_MC],
                                        ms_cycle,
                                        ls_cycle):
    ins_ax2.plot(dims, dt, linestyle='None', markersize=0.5, **ms_props)
    ins_ax2.plot(dims_plot, f(dims_plot, *popt), linewidth=0.5, **ls_props)

ins_ax.set_yticks([])
ins_ax2.set_xlim(0, d_max)
ins_ax.tick_params(direction='out', which='both', labelsize=6)
ins_ax2.tick_params(direction='out', which='both', labelsize=6)
ins_ax.spines['left'].set_visible(False)
ins_ax2.spines['left'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax2.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)
ins_ax2.patch.set_alpha(0)

ax.legend(framealpha=0, handletextpad=0.)

fig.tight_layout()
fig.savefig(spath / (fname + '_linear-inset.eps'), dpi=600)
fig.savefig(spath / (fname + '_linear-inset.pdf'), dpi=600)
