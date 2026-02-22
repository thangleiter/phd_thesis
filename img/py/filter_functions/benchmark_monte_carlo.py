# %% Imports
import datetime
import os
import sys
import pathlib
from time import perf_counter

import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from filter_functions import util
from numpy import ndarray
from qutil import linalg as qla, math
from qutil.plotting.colors import RWTH_COLORS
from scipy.optimize import curve_fit
from scipy import linalg as sla

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, PATH, init, markerprops

LINE_COLORS = list(RWTH_COLORS.values())
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)
RUN_SIMULATION = os.environ.get('RUN_SIMULATION', 'False') == 'True'

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def MC_ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return math.abs2(np.einsum('...ll', gates @ target.conj().T)/d)


def rand_herm(d: int, n: int = 1) -> np.ndarray:
    """n random Hermitian matrices of dimension d"""
    A = (np.random.default_rng().standard_normal((n, d, d))
         + 1j*np.random.default_rng().standard_normal((n, d, d)))
    return (A + A.conj().transpose([0, 2, 1]))/2


def rand_herm_traceless(d: int, n: int = 1) -> np.ndarray:
    """n random traceless Hermitian matrices of dimension d"""
    A = rand_herm(d, n).transpose()
    A -= A.trace(axis1=0, axis2=1)/d
    return A.transpose()


def monte_carlo_gate(N_MC: int, S0: float, f_min: float, f_max: float,
                     dt: ndarray, T: float, alpha: float, c_opers: ndarray,
                     c_coeffs: ndarray, n_opers: ndarray, n_coeffs: ndarray,
                     loop: bool = True, seed: int | None = None, threads: int = 1):

    def evolution(H, dt):
        # np.linalg.eigh (2.2.6) sometimes does not converge
        HD, HV = sla.eigh(H, overwrite_a=True)

        P = np.einsum('...lij,...jl,...lkj->...lik',
                      HV, util.cexp(-np.asarray(dt)*HD.swapaxes(-1, -2)),
                      HV.conj())

        return qla.mdot(np.flip(P, axis=-3), axis=-3)

    sampler = lmt.noise.FFTSpectralSampler((1 if loop else N_MC,),
                                           lmt.noise.WhiteSpectralFunction([S0]*3),
                                           dt, f_max=f_max, f_min=f_min, seed=seed,
                                           threads=threads)
    N_n = sampler.shape[-1] // dt.size
    a = np.repeat(c_coeffs, N_n, axis=1)
    b = np.repeat(n_coeffs, N_n, axis=1)
    dt_fast = np.repeat(dt, N_n)/N_n

    if loop:
        U_tot = np.empty((N_MC, *c_opers.shape[-2:]), dtype=complex)
        for n in range(N_MC):
            H_c = np.tensordot(a, c_opers, axes=[0, 0])
            H_n = np.tensordot(b*sampler(), n_opers, axes=[0, 0])
            H = H_c + H_n
            U_tot[n] = evolution(H, dt_fast)

        return U_tot
    else:
        H_c = np.tensordot(a, c_opers, axes=[0, 0])
        H_n = np.tensordot(b*sampler(), n_opers, axes=[-2, 0])
        H = H_c + H_n

        return evolution(H, dt_fast)


def run_simulation(d_max=120, n_alpha=3, n_dt=1, n_MC=100, n_omega=500, seed=42, threads=1,
                   alpha=0, append_date=True):
    dims = np.arange(2, d_max+1, 2)
    rng = np.random.default_rng(seed)

    tic_MC = []
    toc_MC = []
    tic_FF_H = []
    toc_FF_H = []
    tic_FF_L = []
    toc_FF_L = []
    infids_MC = []
    infids_FF_H = []
    infids_FF_L = []
    t_start = perf_counter()
    print('Dim.\t\tMC\t\t\t\tFF (H)\t\t\tFF (L)\t\t\tTotal elapsed')
    print(73*'-')
    for d in dims:
        opers = rand_herm_traceless(d, n_alpha)
        coeffs = rng.standard_normal((n_alpha, n_dt))
        dt = np.abs(rng.standard_normal())*np.ones(n_dt)
        T = dt.sum()
        f_min = 1e-1/T
        f_max = 1e2/T
        S0 = abs(rng.standard_normal())/1e4

        pulse = ff.PulseSequence(list(zip(opers, coeffs)),
                                 list(zip(opers, np.ones_like(coeffs))),
                                 dt)

        omega = np.geomspace(f_min, f_max, n_omega)*2*np.pi
        S = S0*np.ones_like(omega)

        tic_FF_L.append(perf_counter())
        control_matrix = ff.numeric.calculate_control_matrix_from_scratch(
            pulse.eigvals, pulse.eigvecs, pulse.propagators, omega, pulse.basis, pulse.n_opers,
            pulse.n_coeffs, pulse.dt, pulse.t, show_progressbar=False
        )

        pulse.cache_filter_function(omega, control_matrix=control_matrix)
        infids_FF_L.append(1 - ff.infidelity(pulse, S, omega).sum())
        toc_FF_L.append(perf_counter())

        pulse.cleanup('all')

        tic_FF_H.append(perf_counter())
        B = ff.numeric.calculate_noise_operators_from_scratch(
            pulse.eigvals, pulse.eigvecs, pulse.propagators, omega, pulse.n_opers, pulse.n_coeffs,
            pulse.dt, pulse.t, show_progressbar=False
        )
        pulse.cache_filter_function(
            omega, filter_function=np.einsum('oaji,obji->abo', B.conj(), B)
        )
        infids_FF_H.append(1 - ff.infidelity(pulse, S, omega).sum())
        toc_FF_H.append(perf_counter())

        tic_MC.append(perf_counter())
        MC_propagators = monte_carlo_gate(n_MC, S0, f_min, f_max, dt, T, alpha,
                                          opers, coeffs, opers, np.ones_like(coeffs), loop=False,
                                          seed=seed, threads=threads)
        infids_MC.append(MC_ent_fidelity(MC_propagators, pulse.total_propagator).mean())
        toc_MC.append(perf_counter())

        print(f'd = {d:3d}\t\t'
              f'{toc_MC[-1]-tic_MC[-1]:.1e} s\t\t'
              f'{toc_FF_H[-1]-tic_FF_H[-1]:.1e} s\t\t'
              f'{toc_FF_L[-1]-tic_FF_L[-1]:.1e} s\t\t'
              f'{perf_counter() - t_start:.1e} s')

    dt_MC = np.array(toc_MC) - np.array(tic_MC)
    dt_FF_H = np.array(toc_FF_H) - np.array(tic_FF_H)
    dt_FF_L = np.array(toc_FF_L) - np.array(tic_FF_L)

    fname = 'benchmark_MC_vs_FF'
    if append_date:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = '_'.join([fname, now])

    np.savez(p := (DATA_PATH / fname).with_suffix('.npz'), dims=dims,
             dt_FF_H=dt_FF_H, dt_FF_L=dt_FF_L, dt_MC=dt_MC,
             infids_FF_H=infids_FF_H, infids_FF_L=infids_FF_L,
             infids_MC=infids_MC)
    print(f'Saved data to {p}.')


# %% Run benchmark
if RUN_SIMULATION:
    import lindblad_mc_tools as lmt
    run_simulation(threads=None, append_date=False)

# %% Load data
with np.load(DATA_PATH / 'benchmark_MC_vs_FF.npz') as arch:
    dims = arch['dims']
    dt_MC = arch['dt_MC']
    dt_FF_H = arch['dt_FF_H']
    dt_FF_L = arch['dt_FF_L']
    infids_MC = arch['infids_MC']
    infids_FF_H = arch['infids_FF_H']
    infids_FF_L = arch['infids_FF_L']

# %% Fit


def f(x, *args):
    a, b = args
    return a*x**b


popt_FF_H, _ = curve_fit(f, dims, dt_FF_H, p0=[1, 3], maxfev=10**5)
popt_FF_L, _ = curve_fit(f, dims, dt_FF_L, p0=[1, 3], maxfev=10**5)
popt_MC, _ = curve_fit(f, dims, dt_MC, p0=[1, 2], maxfev=10**5)
# %% Plot
dims_plot = np.linspace(dims[0], dims[-1], 1001)
cycle = cycler(color=LINE_COLORS[:3], marker=['s', 'd', 'v'])

fig, ax = plt.subplots(layout='constrained')
ins_ax = ax.inset_axes((0.46, 0.1, 0.45, 0.44))

for dt, popt, space, sty in zip([dt_MC, dt_FF_H, dt_FF_L],
                                [popt_MC, popt_FF_H, popt_FF_L],
                                ['MC', r'FF ($\mathscr{{H}}$)', r'FF ($\mathscr{{L}}$)'],
                                cycle):
    ax.loglog(dims_plot, f(dims_plot, *popt), color=sty['color'])
    ax.loglog(dims, dt, label=rf'{space}: $\mathcal{{O}}(d^{{{popt[1]:.2f}}})$',
              **markerprops(**sty))

    ins_ax.plot(dims_plot, f(dims_plot, *popt), color=sty['color'], linewidth=0.5)
    ins_ax.plot(dims, dt, **markerprops(**sty, markersize=1))

ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('$d$')
ax.set_ylabel('$t$ (s)')
ax.set_ylim(bottom=1e-3)

ins_ax.set_xlim(0)
ins_ax.tick_params(direction='out', which='both', labelsize='small')
ins_ax.spines['left'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.yaxis.tick_right()
ins_ax.patch.set_alpha(0)

fig.savefig(SAVE_PATH / 'benchmark_MC_vs_FF.pdf')
