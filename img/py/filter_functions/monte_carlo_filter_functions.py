# %% Imports
import os
import pathlib
import sys
import warnings
from typing import Literal

import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
import scipy as sp
from qutil import misc, ui
from qutil.plotting.colors import RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_50

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, TOTALWIDTH, TEXTWIDTH, PATH, init, rand_pulse_sequence

try:
    import matlab.engine
except ImportError:
    eng = False
else:
    eng = None

FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)
RUN_SIMULATION = os.environ.get('RUN_SIMULATION', 'False') == 'True'

init(MAINSTYLE, backend := 'pgf')

rng = np.random.default_rng(42)
pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else r' (ns$^{-1}$)'
# %% Functions


def monte_carlo_filter_function(pulse, omega, sigma=1.0, traces_shape=400,
                                confidence_method: Literal['frechet', 'bootstrap'] = 'frechet',
                                confidence_level: float = 0.9545,
                                **solver_kwargs):
    import lindblad_mc_tools as lmt

    global eng
    if eng is None:
        try:
            eng = matlab.engine.start_matlab()
            # We require the Matrix Function Toolbox and the Logm Frechet package:
            # https://www.mathworks.com/matlabcentral/fileexchange/20820-the-matrix-function-toolbox
            eng.sqrtm_real(np.eye(2))
            # https://www.mathworks.com/matlabcentral/fileexchange/38894-matrix-logarithm-with-frechet-derivatives-and-condition-number  noqa
            eng.logm_frechet_real(np.eye(2), np.eye(2))
        except Exception:
            eng = False

    if eng is False and confidence_method == 'frechet':
        warnings.warn('matlab engine / logm_frechet not installed.')
        confidence_method = 'bootstrap'

    expr = oe.contract_expression('...ba,ibc,...cd,jda->...ij',
                                  (traces_shape, pulse.d, pulse.d), pulse.basis,
                                  (traces_shape, pulse.d, pulse.d), pulse.basis,
                                  optimize=[(0, 1), (0, 1), (0, 1)],
                                  constants=[1, 3])

    def to_liouville(U):
        return expr(U.conj(), U).real

    K_mean = np.empty((pulse.d**2, pulse.d**2, len(omega)))
    K_conf = np.empty((2, pulse.d**2, pulse.d**2, len(omega)))
    Us = []

    for i, freq in enumerate(ui.progressbar(omega)):
        solver = lmt.MonochromaticSchroedingerSolver(
            pulse, lmt.noise.MonochromaticSpectralFunction(sigma, freq, frequency_type='angular'),
            sampling_method=solver_kwargs.pop('sampling_method', 'monte carlo'), **solver_kwargs
        )
        U = to_liouville(solver.complete_propagator)
        U_mean = solver.mean(U)
        U_conf = solver.confidence_interval(U, confidence_level=confidence_level)

        Uerr_mean = pulse.total_propagator_liouville.T @ U_mean
        Uerr_conf = pulse.total_propagator_liouville.T @ U_conf
        try:
            if confidence_method == 'frechet':
                for j in range(2):
                    X, L = map(np.array, eng.logm_frechet_real(
                        Uerr_mean, Uerr_conf[j] - Uerr_mean, nargout=2
                    ))
                    K_conf[j, ..., i] = X + L
                K_mean[..., i] = X
            else:
                with misc.filter_warnings(action='error', category=np.exceptions.ComplexWarning):
                    K_mean[..., i] = sp.linalg.logm(Uerr_mean)
                    K_conf[..., i] = solver.confidence_interval(
                        sp.linalg.logm(pulse.total_propagator_liouville.T @ U),
                        confidence_level=confidence_level
                    )
        except (np.exceptions.ComplexWarning, Exception) as err:
            raise ValueError('U is not positive semidefinite; reduce sigma') from err

        Us.append(U)

    return -K_mean / sigma**2, -K_conf[::-1] / sigma**2, np.array(Us)


def incoherent_filter_function(pulse, omega):
    traces = pulse.basis.four_element_traces
    basis_FF = pulse.get_filter_function(omega, 'generalized', order=1)
    return (
        + oe.contract('...klo,klji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,kjli->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,kilj->...ijo', basis_FF, traces, backend='sparse').real
        + oe.contract('...klo,kijl->...ijo', basis_FF, traces, backend='sparse').real
    ) / 2


def coherent_filter_function(pulse, omega):
    traces = pulse.basis.four_element_traces
    basis_FF = pulse.get_filter_function(omega, 'generalized', order=2)
    return (
        + oe.contract('...klo,klji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,lkji->...ijo', basis_FF, traces, backend='sparse').real
        - oe.contract('...klo,klij->...ijo', basis_FF, traces, backend='sparse').real
        + oe.contract('...klo,lkij->...ijo', basis_FF, traces, backend='sparse').real
    ) / 2


def to_symmetric(X, axes=(0, 1)):
    return 0.5*(X + X.conj().swapaxes(*axes))


def to_antisymmetric(X, axes=(0, 1)):
    return 0.5*(X - X.conj().swapaxes(*axes))


def plot_separate(pulse, omega, FF_inc, FF_coh, MC_mean, MC_conf, cix=None, rix=None,
                  figsize=(TOTALWIDTH, 3.5)):
    if cix is None:
        cix = list(range(1, 4))
    if rix is None:
        rix = list(range(1, 4))

    fig, axes = plt.subplots(len(rix), len(cix), sharex=True, layout='constrained',
                             figsize=figsize)

    diag_axes = np.diagonal(axes)
    off_diag_axes = np.concatenate((axes[np.triu_indices_from(axes, +1)],
                                    axes[np.tril_indices_from(axes, -1)]))
    for ax in diag_axes[1:]:
        ax.sharey(diag_axes[0])
    for ax in off_diag_axes[1:]:
        ax.sharey(off_diag_axes[0])

    P = 'XYZ'
    for r, i in enumerate(rix):
        for c, j in enumerate(cix):
            ax1 = axes[r, c]
            ax2 = ax1

            if r == c:
                ax1.set_yscale('log')
            else:
                ax1.set_yscale('asinh', linear_width=0.075)

            ax1.semilogx(omega, to_symmetric(MC_mean)[i, j], color=RWTH_COLORS_75['blue'],
                         alpha=0.66, linewidth=0.5)
            ax1.fill_between(omega, *to_symmetric(MC_conf, (1, 2))[:, i, j],
                             alpha=0.33, facecolor=RWTH_COLORS_50['blue'])
            ax2.semilogx(omega, to_antisymmetric(MC_mean)[i, j], color=RWTH_COLORS_75['magenta'],
                         alpha=0.66, linewidth=0.5, ls='--')
            ax2.fill_between(omega, *to_antisymmetric(MC_conf, (1, 2))[:, i, j],
                             alpha=0.33, facecolor=RWTH_COLORS_50['magenta'])

            ax1.semilogx(omega, FF_inc[0, 0, i, j], '-', color=RWTH_COLORS['blue'],
                         linewidth=0.5)
            ax2.semilogx(omega, FF_coh[0, 0, i, j], '--', color=RWTH_COLORS['magenta'],
                         linewidth=0.5)

            ax1.annotate(P[r] + P[c], (0.965, 0.925), xycoords='axes fraction',
                         verticalalignment='top', horizontalalignment='right')

    axes[0, 1].set_ylim(-10, 10)
    axes[1, 0].set_ylabel(r'$\mathcal{F}_{\Gamma,\Delta}(\omega;\tau)$')
    axes[-1, 1].set_xlabel(r'$\omega$' + pernanosecond)
    for ax in axes.flat:
        for loc, txt in zip(ax.get_yticks(), ax.get_yticklabels()):
            if loc == 0.0:
                txt.set_visible(False)

    return fig, axes


def plot_complete(pulse, omega, FF_inc, FF_coh, MC_mean, MC_conf, cix=None, rix=None,
                  figsize=(TOTALWIDTH, 3.5)):
    if cix is None:
        cix = list(range(1, 4))
    if rix is None:
        rix = list(range(1, 4))

    fig, axes = plt.subplots(len(rix), len(cix), sharex=True, layout='constrained',
                             figsize=figsize)

    diag_axes = np.diagonal(axes)
    off_diag_axes = np.concatenate((axes[np.triu_indices_from(axes, +1)],
                                    axes[np.tril_indices_from(axes, -1)]))
    for ax in diag_axes[1:]:
        ax.sharey(diag_axes[0])
    for ax in off_diag_axes[1:]:
        ax.sharey(off_diag_axes[0])

    P = 'XYZ'
    for r, i in enumerate(rix):
        for c, j in enumerate(cix):
            ax = axes[r, c]

            if r == c:
                ax.set_yscale('log')
            else:
                ax.set_yscale('asinh', linear_width=0.075)

            ax.semilogx(omega, MC_mean[i, j], color=RWTH_COLORS['blue'], alpha=0.66, linewidth=0.5)
            ax.fill_between(omega, *MC_conf[:, i, j], facecolor=RWTH_COLORS_50['blue'], alpha=0.33)
            ax.semilogx(omega, (FF_inc + FF_coh)[0, 0, i, j], color=RWTH_COLORS['magenta'],
                        linewidth=0.5)

            ax.annotate(P[r] + P[c], (0.965, 0.925), xycoords='axes fraction',
                        verticalalignment='top', horizontalalignment='right')

    axes[0, 1].set_ylim(-10, 10)
    axes[1, 0].set_ylabel(r'$\mathcal{F}(\omega;\tau)$')
    axes[-1, 1].set_xlabel(r'$\omega$' + pernanosecond)
    for ax in axes.flat:
        for loc, txt in zip(ax.get_yticks(), ax.get_yticklabels()):
            if loc == 0.0:
                txt.set_visible(False)

    return fig, axes


def simulate(pulse):
    omega = ff.util.get_sample_frequencies(pulse, n_samples=300, include_quasistatic=True)
    FF_inc = incoherent_filter_function(pulse, omega)
    FF_coh = coherent_filter_function(pulse, omega)
    # Too large a sigma results in crass overrotations and bad convergence.
    # Too small results in statistical noise
    MC_mean, MC_conf, U = monte_carlo_filter_function(
        pulse, omega, sigma=0.5 / pulse.tau,
        confidence_method='frechet',
        oversampling=10, traces_shape=400,
        threads=None
    )
    return {'omega': omega,
            'FF_inc': FF_inc, 'FF_coh': FF_coh,
            'MC_mean': MC_mean, 'MC_conf': MC_conf}


# %% Define atomic pulses
n_dt = 1
t_pi = 1
t_idle = 10

X_pi = ff.PulseSequence(
    [[ff.util.paulis[1]/2, [np.pi]*n_dt, 'X']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [t_pi / n_dt]*n_dt
)
Z_pi = ff.PulseSequence(
    [[ff.util.paulis[3]/2, [np.pi]*n_dt, 'Z']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [t_pi / n_dt]*n_dt
)
Idle = ff.PulseSequence(
    [[ff.util.paulis[3]/2, [0]*n_dt, 'Z']],
    [[ff.util.paulis[3]/2, [1]*n_dt, 'Z']],
    [t_idle / n_dt]*n_dt
)
# %% X
pulse = X_pi
if RUN_SIMULATION:
    results = simulate(pulse)
    np.savez_compressed(DATA_PATH / 'monte_carlo_FF_X.npz', **results)
else:
    with np.load(DATA_PATH / 'monte_carlo_FF_X.npz') as arch:
        results = dict(arch)
# %%% Plot
fig, axes = plot_complete(pulse, **results)
axes[0, 1].set_yscale('asinh', linear_width=2e-2)
axes[0, 1].set_ylim(-3e-1, 3e-1)
fig.get_layout_engine().set(h_pad=1/72, w_pad=1/72, hspace=0, wspace=0)
fig.savefig(SAVE_PATH / 'monte_carlo_FF_X.pdf')
# %% Z
pulse = Z_pi
if RUN_SIMULATION:
    results = simulate(pulse)
    np.savez_compressed(DATA_PATH / 'monte_carlo_FF_Z.npz', **results)
else:
    with np.load(DATA_PATH / 'monte_carlo_FF_Z.npz') as arch:
        results = dict(arch)
# %%% Plot
fig, axes = plot_complete(pulse, **results, cix=[1, 2], rix=[1, 2], figsize=(TEXTWIDTH, 2.5))
axes[0, 1].set_yscale('asinh', linear_width=2e-2)
axes[0, 1].set_ylim(-5e-1, 5e-1)
axes[0, 0].set_ylim(1e-7)
axes[1, 0].set_ylabel(None)
axes[1, 1].set_xlabel(None)
fig.supylabel(r'$\mathcal{F}(\omega;\tau)$', fontsize='medium')
fig.supxlabel(r'$\omega$' + pernanosecond, fontsize='medium')
fig.get_layout_engine().set(h_pad=1/72, w_pad=1/72, hspace=0, wspace=0)
fig.savefig(SAVE_PATH / 'monte_carlo_FF_Z.pdf')
# %% Spin echo
pulse = Idle @ X_pi @ Idle
if RUN_SIMULATION:
    results = simulate(pulse)
    np.savez_compressed(DATA_PATH / 'monte_carlo_FF_spin_echo.npz', **results)
else:
    with np.load(DATA_PATH / 'monte_carlo_FF_spin_echo.npz') as arch:
        results = dict(arch)
# %%% Plot
fig, axes = plot_complete(pulse, **results)
fig.get_layout_engine().set(h_pad=1/72, w_pad=1/72, hspace=0, wspace=0)
fig.savefig(SAVE_PATH / 'monte_carlo_FF_spin_echo.pdf')
# %% Random pulse (unused)
# pulse = rand_pulse_sequence(2, 10, n_nops=1, commensurable_timesteps=True, rng=rng)
# if RUN_SIMULATION:
#     results = simulate(pulse)
#     np.savez_compressed(DATA_PATH / 'monte_carlo_FF_rand.npz', **results)
# else:
#     with np.load(DATA_PATH / 'monte_carlo_FF_rand.npz') as arch:
#         results = dict(arch)
# %%% Plot
# fig, axes = plot_complete(pulse, **results)
# fig.get_layout_engine().set(h_pad=1/72, w_pad=1/72, hspace=0, wspace=0)
# fig.savefig(SAVE_PATH / 'monte_carlo_FF_rand.pdf')
