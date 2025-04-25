# %%
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy as sc
from matplotlib import colors

from lindblad_mc_tools.noise import FFTSpectralSampler
from lindblad_mc_tools.noise.real_space import MultithreadedRNG
from qutil import functools, const, signal_processing as sp

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import PATH, MARGINWIDTH  # noqa

mpl.use('pgf')
mpl.rcdefaults()
mpl.style.use('margin.mplstyle')
# %%


def psd(f, σ, τ_c):
    # two-sided definition
    return 2 * σ ** 2 * τ_c / (1 + (2 * np.pi * f * τ_c) ** 2)


def corr(τ, σ, τ_c):
    return σ ** 2 * np.exp(-abs(τ) / τ_c)


def noise(σ, τ_c, N, method='lmt'):
    match method:
        case 'lmt':
            return FFTSpectralSampler(
                (N,), functools.partial(functools.scaled(2)(psd), σ=σ, τ_c=τ_c),
                dt=np.full(L, Δt), seed=SEED,
            ).values.squeeze()
        case 'bartosch':
            Z = np.empty((N, L))
            mrng = MultithreadedRNG(SEED)
            mrng.fill(Z)
            return bartosch_2001_I(Z, Δt, σ, τ_c).squeeze()


@nb.njit(parallel=True)
def bartosch_2001_I(Z, dt, σ=1, τ=1):
    ρ = np.exp(-dt/τ)
    ξ = np.sqrt(1 - ρ ** 2) * σ

    X = np.empty_like(Z)
    X[:, 0] = σ * Z[:, 0]
    for m in nb.prange(Z.shape[0]):
        for n in range(1, Z.shape[1]):
            X[m, n] = ρ * X[m, n-1] + ξ * Z[m, n]

    return X


# %% Parameters
SEED = 1
O = 1000
L = 1000
Δt = 1
T = L * Δt
τ_cs = np.array([1e-2, 1, 1e2])
σs = .5 * τ_cs ** 0.25

# %% Both in same figure
alpha = 0.5
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

with mpl.style.context(['./margin.mplstyle'], after_reset=True):
    fig, axes = plt.subplots(3, 1, figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 3),
                             layout='constrained')

    for σ, τ_c in zip(σs, τ_cs):
        X = noise(σ, τ_c, O, method='lmt')

        # Timetrace
        ax = axes[0]
        t = np.linspace(0, 1, L)
        ax.plot(t, X[0] / sp.real_space.rms(X[0]), '-')

        # Correlation
        ax = axes[1]
        τ = np.insert(np.geomspace(1e-3, L, L), 0, 0)
        ln, = ax.plot(τ / τ_cs[1], corr(τ, σ, τ_c) / σs[1] ** 2, '-')

        C = np.array([sc.signal.correlate(*[x]*2) for x in X])
        τ = sc.signal.correlation_lags(L, L)
        # select only a few of the data points
        log_indices = np.logspace(0, np.log10(L), num=25, endpoint=True) - 1
        idx = np.unique(np.round(log_indices).astype(int))

        ax.errorbar(τ[τ >= 0][idx] / τ_cs[1],
                    (C.mean(0)[τ >= 0] / L)[idx] / σs[1] ** 2,
                    (C.std(0)[τ >= 0] / L)[idx] / σs[1] ** 2 / np.sqrt(O),
                    color=ln.get_color(),
                    ecolor=colors.to_rgb(ln.get_color()) + (alpha,),
                    marker='.',
                    ls='',
                    markersize=5,
                    markeredgecolor=ln.get_color(),
                    markerfacecolor=colors.to_rgb(ln.get_color()) + (alpha,))

        # PSD
        ax = axes[2]
        f = np.insert(np.geomspace(1e-4 * τ_cs[1], (L - 1) / (τ_cs[1] * 2 * np.pi), L), 0, 0)
        ln, = ax.plot(f * 2 * np.pi,
                      psd(f, σ, τ_c) / (2 * τ_cs[1] * σs[1] ** 2))

        fx, Sx = sc.signal.periodogram(X, fs=1/Δt, axis=-1, detrend=False, return_onesided=False)
        # select only a few of the data points
        log_indices = np.logspace(0, np.log10(len(fx)), num=15, endpoint=True) - 1
        idx = np.unique(np.round(log_indices).astype(int))
        ax.errorbar(fx[idx]*2*np.pi,
                    Sx.mean(0)[idx] / (2 * τ_cs[1] * σs[1] ** 2),
                    Sx.std(0)[idx] / (np.sqrt(O) * 2 * τ_cs[1] * σs[1] ** 2),
                    color=ln.get_color(),
                    ecolor=colors.to_rgb(ln.get_color()) + (alpha,),
                    marker='.',
                    ls='',
                    markersize=5,
                    markeredgecolor=ln.get_color(),
                    markerfacecolor=colors.to_rgb(ln.get_color()) + (alpha,))

    ax = axes[0]
    ax.margins(x=0)
    ax.set_xticks([0, 1])
    ax.set_xlabel(r'$\flatfrac{t}{T}$', labelpad=-8.5)
    ax.set_ylabel(r'$\flatfrac{x(t)}{\mathrm{RMS}_x}$')

    ax = axes[1]
    ax.set_xscale('asinh', linear_width=1e-3)
    ax.set_yscale('asinh', linear_width=2.275e-2)
    ax.set_xlim(0, τ[-1] / τ_cs[1])
    ax.set_ylim(-1e-2)
    ax.set_xticks([0, *τ_cs])
    ax.tick_params('x', pad=10)
    for label in ax.get_xticklabels():
        label.set_verticalalignment('bottom')
    ax.set_yticks([0, 1e-1, 1, 1e1])
    ax.set_xlabel(r'$\flatfrac{\tau}{\tau_c}$')
    ax.set_ylabel(r'$\flatfrac{C(\tau)}{\sigma^2}$')
    ax.grid()

    ax = axes[2]
    ax.set_xscale('asinh', linear_width=1e-3)
    ax.set_yscale('log')
    ax.margins(y=0.075)
    ax.set_xlim(axes[1].get_xlim())
    # ax.set_ylim(2e-9)
    ax.set_xticks([0, *τ_cs])
    ax.tick_params('x', pad=10)
    for label in ax.get_xticklabels():
        label.set_verticalalignment('bottom')
    ax.set_yticks([1e-6, 1e-3, 1, 1e3])
    ax.set_xlabel(r'$\omega\tau_c$')
    ax.set_ylabel(r'$\flatfrac{S(\omega)}{2\tau_c\sigma^2}$')
    ax.grid()

    fig.savefig(PATH / 'pdf/spectrometer/lorentzian_psdcorr.pdf', backend='pgf')
