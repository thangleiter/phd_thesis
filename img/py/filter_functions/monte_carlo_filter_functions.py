# %% Imports
import filter_functions as ff
import lindblad_mc_tools as lmt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import opt_einsum as oe
from qutil import ui

try:
    import matlab.engine

    eng = matlab.engine.start_matlab()
except ImportError:
    eng = None

# os.chdir(pathlib.Path(ff.__file__).parents[1])
#
# from tests import testutil

# %%


def monte_carlo_filter_function(pulse, omega, sigma=1.0, sampling_method='discrete',
                                **solver_kwargs):

    mean = np.empty((pulse.d**2, pulse.d**2, len(omega)))
    std = np.empty_like(mean)
    Us = []

    for i, freq in enumerate(ui.progressbar(omega)):
        solver = lmt.MonochromaticSchroedingerSolver(
            pulse, lmt.noise.MonochromaticSpectralFunction(sigma, freq, frequency_type='angular'),
            sampling_method=sampling_method, **solver_kwargs
        )
        if i == 0:
            expr = oe.contract_expression('...ba,ibc,...cd,jda->...ij',
                                          solver.complete_propagator.shape, pulse.basis,
                                          solver.complete_propagator.shape, pulse.basis,
                                          optimize=[(0, 1), (0, 1), (0, 1)],
                                          constants=[1, 3])

            def to_liouville(U):
                return expr(U.conj(), U).real

        U = to_liouville(solver.complete_propagator)

        match sampling_method:
            case 'discrete':
                U_mean = solver.mean(U)
            case 'monte carlo':
                U_mean = solver.mean(U)
                U_std = solver.standard_error(U)
            case _:
                raise ValueError(f'Unknown sampling method: {sampling_method}')

        Uerr_mean = pulse.total_propagator_liouville.T @ U_mean
        Uerr_std = pulse.total_propagator_liouville.T @ U_std
        if eng is not None:
            X, L, cond = map(np.array, eng.logm_frechet(Uerr_mean, Uerr_std, nargout=3))
            mean[..., i] = -X / sigma ** 2
            std[..., i] = L
        else:
            mean[..., i] = -1 / sigma ** 2 * lmt.util.logm(Uerr_mean)
            std[..., i] = sp.linalg.solve(Uerr_mean, Uerr_std) / np.sqrt(solver.traces_shape[0])
        Us.append(U)

    return mean, std, np.array(Us)


def incoherent_filter_function(pulse, omega):
    traces = pulse.basis.four_element_traces
    basis_FF = pulse.get_filter_function(omega, 'generalized')
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


# %%
# pulse = testutil.rand_pulse_sequence(2, 10, n_nops=1, commensurable_timesteps=True)
n_dt = 100
pulse = ff.PulseSequence(
    # [[ff.util.paulis[3], [2*np.pi]*n_dt, 'Z']],
    [[ff.util.paulis[1], [2*np.pi]*n_dt, 'X']],
    # [[ff.util.paulis[1], [0]*n_dt, 'X']],
    [[ff.util.paulis[3], [1]*n_dt, 'Z']],
    [10 / n_dt]*n_dt
)
# pulse.dt *= 1e-2
# pulse.c_coeffs *= 1e2
omega = ff.util.get_sample_frequencies(pulse, n_samples=300, include_quasistatic=True)
# %%%
mean, std, Us = monte_carlo_filter_function(
    pulse, omega, sigma=1e-2,
    sampling_method='monte carlo',
    oversampling=1, traces_shape=400,
    threads=None
)
# mean, std, Us = monte_carlo_filter_function(pulse, omega[-10:], oversampling=1, n_MC=10000)
FF_Γ = incoherent_filter_function(pulse, omega)
FF_Δ = coherent_filter_function(pulse, omega)
# %%%
fig, ax = plt.subplots(pulse.d**2 - 1, pulse.d**2 - 1, sharex=True, sharey=False)
for i in range(1, pulse.d**2):
    for j in range(1, pulse.d**2):
        if i == j:
            ax[i-1, j-1].set_yscale('log')

        ax[i-1, j-1].grid()

        ax[i-1, j-1].semilogx(omega, FF_Γ[0, 0, i, j], '-.', color='tab:blue', alpha=0.5)
        ax[i-1, j-1].semilogx(omega, FF_Δ[0, 0, i, j], '--', color='tab:blue', alpha=0.5)
        ax[i-1, j-1].semilogx(omega, FF_Γ[0, 0, i, j] + FF_Δ[0, 0, i, j], color='tab:blue')
        ax[i-1, j-1].errorbar(omega, mean[i, j], abs(std[i, j]), fmt='.-',
                              color=mpl.colors.to_rgba('tab:red', 0.3),
                              ecolor=mpl.colors.to_rgba('tab:red', 0.3),
                              markeredgecolor=mpl.colors.to_rgba('tab:red', 0.7))

        # ax[i-1, j-1].plot(omega, mean[i, j], '.')
        # ax[i-1, j-1].fill_between(omega, (mean - std)[i, j], (mean + std)[i, j],
        #                           color='tab:orange', alpha=0.3)

# %%%
fig, ax = plt.subplots(pulse.d**2 - 1, pulse.d**2 - 1, sharex=True, sharey=False)
for i in range(1, pulse.d**2):
    for j in range(1, pulse.d**2):
        if i == j:
            ax[i-1, j-1].set_yscale('log')

        ax[i-1, j-1].grid()
        ax[i-1, j-1].semilogx(omega, FF_Γ[0, 0, i, j], color='tab:blue')
        ax[i-1, j-1].errorbar(omega, (mean[i, j] + mean[j, i])/2, abs(std[i, j]), fmt='.-',
                              color=mpl.colors.to_rgba('tab:red', 0.3),
                              ecolor=mpl.colors.to_rgba('tab:red', 0.3),
                              markeredgecolor=mpl.colors.to_rgba('tab:red', 0.7))

        # ax[i-1, j-1].plot(omega, mean[i, j], '.')
        # ax[i-1, j-1].fill_between(omega, (mean - std)[i, j], (mean + std)[i, j],
        #                           color='tab:orange', alpha=0.3)

# %%%
fig, ax = plt.subplots(pulse.d**2 - 1, pulse.d**2 - 1, sharex=True, sharey=False)
for i in range(1, pulse.d**2):
    for j in range(1, pulse.d**2):
        ax[i-1, j-1].grid()
        ax[i-1, j-1].semilogx(omega, FF_Δ[0, 0, i, j], color='tab:blue')
        ax[i-1, j-1].errorbar(omega, (mean[i, j] - mean[j, i])/2, abs(std[i, j]), fmt='.',
                              color=mpl.colors.to_rgba('tab:red', 0.3),
                              ecolor=mpl.colors.to_rgba('tab:red', 0.3),
                              markeredgecolor=mpl.colors.to_rgba('tab:red', 0.7))

        # ax[i-1, j-1].plot(omega, mean[i, j], '.')
        # ax[i-1, j-1].fill_between(omega, (mean - std)[i, j], (mean + std)[i, j],
        #                           color='tab:orange', alpha=0.3)
