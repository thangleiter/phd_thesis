# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy as sc
from matplotlib import colors
from matplotlib.ticker import FuncFormatter

from lindblad_mc_tools.noise import FFTSpectralSampler
from lindblad_mc_tools.noise.real_space import MultithreadedRNG
from qopt.noise import fast_colored_noise
from qutil import functools

from common import PATH

mpl.use('pgf')
# mpl.style.use('margin.mplstyle')
# %%


def psd(f, σ, τ_c):
    return 4 * σ ** 2 * τ_c / (1 + (2 * np.pi * f * τ_c) ** 2)


def corr(τ, σ, τ_c):
    return σ ** 2 * np.exp(-abs(τ) / τ_c)


def noise(σ, τ_c, N, method):
    match method:
        case 'qopt':
            return fast_colored_noise(functools.partial(psd, σ=σ, τ_c=τ_c), 1, L, (N,))
        case 'lmt':
            return FFTSpectralSampler(
                (N,), functools.partial(psd, σ=σ, τ_c=τ_c), dt=np.full(L, 1), seed=SEED
            ).values.squeeze()
        case 'bartosch':
            Z = np.empty((N, L))
            mrng = MultithreadedRNG(SEED)
            mrng.fill(Z)
            return bartosch_2001_I_nb_mt(Z, 1, σ, τ_c).squeeze()


@nb.njit
def bartosch_2001_I_nb_st(rng, t, σ=1, τ=1):
    ρ = np.exp(-(t[1] - t[0])/τ)
    ξ = np.sqrt(1 - ρ ** 2) * σ

    N = len(t)
    Z = rng.standard_normal(N)
    X = np.empty_like(Z)
    X[0] = σ * Z[0]
    for n in range(1, N):
        X[n] = ρ * X[n-1] + ξ * Z[n]

    return X


@nb.njit(parallel=True)
def bartosch_2001_I_nb_mt(Z, dt, σ=1, τ=1):
    ρ = np.exp(-dt/τ)
    ξ = np.sqrt(1 - ρ ** 2) * σ

    X = np.empty_like(Z)
    X[:, 0] = σ * Z[:, 0]
    for m in nb.prange(Z.shape[0]):
        for n in range(1, Z.shape[1]):
            X[m, n] = ρ * X[m, n-1] + ξ * Z[m, n]

    return X


# %%
L = 1000
SEED = 1
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
exp_formatter = FuncFormatter(lambda x, pos: f'$10^{{{int(np.log10(x))}}}$')

τ_cs = np.array([1e-2, 1, 1e2])
σs = .5 * τ_cs ** 0.25
# %% PSDs

with mpl.style.context(['./margin.mplstyle'], after_reset=True):
    fig, ax = plt.subplots(layout='constrained')
    for σ, τ_c in zip(σs, τ_cs):
        ax.plot((f := np.insert(np.geomspace(1e-4, 1e2, 1000), 0, 0)) * 2 * np.pi * τ_cs[1],
                psd(f, σ, τ_c) / (4 * τ_cs[1] * σs[1] ** 2),
                label=exp_formatter(τ_c))

    ax.set_xscale('asinh', linear_width=7e-4)
    ax.set_yscale('log')
    ax.margins(x=0, y=0.075)
    ax.set_xticks([0, *τ_cs])
    ax.tick_params('x', pad=10)
    for label in ax.get_xticklabels():
        label.set_verticalalignment('bottom')
    ax.set_yticks([1e-6, 1e-3, 1, 1e3])
    ax.set_xlabel(r'$2\pi f\tau_c$')
    ax.set_ylabel(r'$\flatfrac{S(f)}{4\tau_c\sigma^2}$')
    ax.grid()
    # ax.legend()

    fig.savefig(PATH / 'pdf/spectrometer/lorentzian_psd.pdf', backend='pgf')

# %% Correlators
alpha = 0.5

with mpl.style.context(['./margin.mplstyle'], after_reset=True):
    fig, ax = plt.subplots(layout='constrained')
    for σ, τ_c in zip(σs, τ_cs):
        τ = np.insert(np.geomspace(1e-3, L, 1000), 0, 0)
        # τ = np.insert(np.geomspace(1e-2, L, 1000), 0, 0)
        ln, = ax.plot(τ / τ_cs[1], corr(τ, σ, τ_c) / σs[1] ** 2, '--')

        C = np.mean([sc.signal.correlate(*[n]*2) for n in noise(σ, τ_c, 1000, 'lmt')], axis=0)
        τ = sc.signal.correlation_lags(L, L)
        log_indices = np.logspace(0, np.log10(L), num=25, endpoint=True) - 1
        # Round to the nearest integer and convert to int
        idx = np.unique(np.round(log_indices).astype(int))
        ax.plot(τ[τ >= 0][idx] / τ_cs[1], (C[τ >= 0] / np.arange(L, 0, -1))[idx] / σs[1] ** 2,
                color=ln.get_color(),
                marker='.',
                ls='',
                markersize=5,
                markeredgecolor=ln.get_color(),
                markerfacecolor=colors.to_rgb(ln.get_color()) + (alpha,))

    ax.set_xscale('asinh', linear_width=1e-3)
    # ax.set_xscale('asinh', linear_width=8.5e-2)
    ax.set_yscale('asinh', linear_width=3e-2)
    ax.set_xlim(0, τ[-1] / τ_cs[1])
    ax.set_xticks([0, *τ_cs])
    ax.tick_params('x', pad=10)
    for label in ax.get_xticklabels():
        label.set_verticalalignment('bottom')
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), va='bottom', pad=10)
    # ax.set_xticks([t for t in ax.get_xticks() if t != 0])
    ax.set_yticks([0, 1e-1, 1, 1e1])
    ax.set_xlabel(r'$\flatfrac{\tau}{\tau_c}$')
    ax.set_ylabel(r'$\flatfrac{C(\tau)}{\sigma^2}$')
    ax.grid()

    fig.savefig(PATH / 'pdf/spectrometer/lorentzian_corr.pdf', backend='pgf')
