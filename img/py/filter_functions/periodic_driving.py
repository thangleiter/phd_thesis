import time
import git

import numpy as np
import filter_functions as ff
import matplotlib.pyplot as plt
from filter_functions import plotting

from pathlib import Path
from itertools import repeat

from scipy.special import j0, j1

plt.style.use('publication')
golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (7.05826, 7.05826*golden_ratio)
# %% paths
# thesis_path = Path('/home/tobias/Physik/Master/Thesis')
thesis_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis')
# thesis_path = Path('Z:/MA/')
# project_path = Path('/home/tobias/Physik/Publication/')
# project_path = Path('Z:/Publication/')
project_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Publication')
name = ('efficient_calculation_of_generalized_filter_functions')
save_path = project_path / name / 'img'

# %% parameters

# We calculate with hbar == 1
Delta = 0.0   # set detuning to zero
omega_d = 20e9*2*np.pi
omega_d = 20*2*np.pi
omega_0 = omega_d + Delta
# Phase shift; phi = 0 gives us a rotation about Y, phi = pi/2 about X
phi = np.pi/2
# Rabi frequency
Omega_R = 1e6*2*np.pi
Omega_R = 1e-3*2*np.pi
A = np.sqrt(Omega_R**2 - Delta**2)
T = 2*np.pi/omega_d

t = np.linspace(0, T, 101)
dt = np.diff(t)
# Paulis
X, Y, Z = ff.util.paulis[1:]

H_c = [[Z, [omega_0/2]*len(dt), r'$F_{zz}$'],
       [X, A*np.sin(omega_d*t[1:] + phi), r'$F_{xx}$']]
H_n = [[Z, np.ones_like(dt), r'$F_{zz}$'],
       [X, np.ones_like(dt), r'$F_{xx}$']]

omega = np.geomspace(1e-5*Omega_R, 5e1*omega_0, 500)
omega = np.geomspace(1e-7, 1e4, 500)
# omega = np.geomspace(1e-1, 1e9, 501)/400000/T

# %% concatenation
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

# tic.append(time.perf_counter())
# NOT_STANDARD = ff.concatenate(repeat(X_ATOMIC, 10000))
# toc.append(time.perf_counter())

ID20 = ff.concatenate_periodic(NOT_PERIODIC, 40)

# %%% brute force
# t = np.linspace(0, T*10000, len(X_ATOMIC.dt)*10000+1)
# dt = np.diff(t)
# H_c = [[Z, [omega_0/2]*len(dt)],
#         [X, A*np.sin(omega_d*t[1:] + phi)]]
# H_n = [[Z, np.ones_like(dt)],
#         [X, np.ones_like(dt)]]

# tic.append(time.perf_counter())
# NOT_FULL = ff.PulseSequence(H_c, H_n, dt)
# toc.append(time.perf_counter())

# tic.append(time.perf_counter())
# F2 = NOT_FULL.get_filter_function(omega, show_progressbar=True)
# toc.append(time.perf_counter())

# tictoc = np.array(toc) - np.array(tic)

# labels = ['Concatenation (periodic)', 'Concatenation (standard)',
#           'Brute force\t\t']

# repo = git.Repo('Z:/Code/filter_functions')
# sha = repo.head.object.hexsha
# fname = 'periodic_driving_benchmark_commit-{}.txt'.format(sha[:7])
# with open(save_path / fname, 'w+') as f:
#     f.write(f'Git commit hash: {sha}\n\n')
#     f.write('=========================================================\n')
#     for l, tito in zip(labels, (tictoc[:3].sum(),
#                                 tictoc[3:4].sum(),
#                                 tictoc[4:].sum())):
#         f.write(f'{l} : {tito:>8.4f} s\n')

# del NOT_FULL
# %%% Plot FF of 20 identities

# fig, ax, legend = ff.plot_filter_function(
#     ID20, omega_in_units_of_tau=False,
#     xscale='log', yscale='log', plot_kw=dict(linewidth=1),
#     figsize=figsize_narrow
# )
# ax.axvline(Omega_R, ls=':', color='tab:green',
#            label=r'$\Omega_R$', zorder=0, linewidth=1)
# ax.axvline(omega_d, ls='--', color='tab:green',
#            label=r'$\omega_0$', zorder=0, linewidth=1)
# legend = ax.legend(loc='lower left', ncol=2, frameon=True, framealpha=1)

# ax.tick_params(direction='in', which='both', labelsize=8)
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.grid(False)
# ax.set_xlabel(r'$\omega$ (\si{\per\second})')
# ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')

# fig.tight_layout(h_pad=0)

# fname = f'rabi_driving'
# fig.savefig(save_path / Path(fname + '.eps'), dpi=600)
# fig.savefig(save_path / Path(fname + '.pdf'), dpi=600)
# %% RWA
# t = np.linspace(0, T, 101)
# dt = np.diff(t)
# H_c = [[X, A/2*np.cos(Delta)*np.ones_like(dt), 'X'],
#        [Y, A/2*np.sin(Delta)*np.ones_like(dt), 'Y']]
# H_n = [[Z, np.ones_like(dt), '$F_{zz}$'],
#        [X, np.ones_like(dt), '$F_{xx}$']]

# X_ATOMIC_RWA = ff.PulseSequence(H_c, H_n, dt)
# X_ATOMIC_RWA.cache_control_matrix(omega)

# NOT_RWA = ff.concatenate_periodic(X_ATOMIC_RWA, 10000)
# ID20_RWA = ff.concatenate_periodic(NOT_RWA, 40)
# %%% Plot Lab vs RWA

# fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
#                          figsize=(figsize_narrow[0], figsize_narrow[0]))

# _ = ff.plot_filter_function(
#     ID20, omega_in_units_of_tau=False,
#     xscale='log', yscale='log', plot_kw=dict(linewidth=1),
#     fig=fig, axes=axes[0]
# )
# _ = ff.plot_filter_function(
#     ID20_RWA, omega_in_units_of_tau=False,
#     xscale='log', yscale='log', plot_kw=dict(linewidth=1),
#     fig=fig, axes=axes[1]
# )

# subfigs = ('(a)', '(b)')
# for ax, subfig in zip(axes, subfigs):
#     ax.axvline(Omega_R, ls=':', color='tab:green',
#                label=r'$\Omega_R$', zorder=0, linewidth=1)
#     ax.axvline(omega_d, ls='--', color='tab:green',
#                label=r'$\omega_0$', zorder=0, linewidth=1)
#     ax.tick_params(direction='in', which='both', labelsize=8, zorder=10)
#     ax.xaxis.set_ticks_position('both')
#     ax.yaxis.set_ticks_position('both')
#     ax.grid(False)
#     ax.set_xlabel(r'$\omega$ (\si{\per\second})')
#     ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')
#     ax.text(0.85, 0.85, subfig, transform=ax.transAxes)

# axes[0].get_legend().set_visible(False)
# axes[0].set_xlabel('')
# legend = axes[1].legend(loc='lower left', ncol=2, frameon=True, framealpha=1)

# fig.tight_layout(h_pad=0.4)

# fname = f'rabi_driving_LAB_vs_RWA'
# fig.savefig(save_path / Path(fname + '.eps'), dpi=600)
# fig.savefig(save_path / Path(fname + '.pdf'), dpi=600)


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
# omega_d = 2*np.pi*2.288
omega_0 = omega_d + Delta
# Phase shift; phi = 0 gives us a rotation about Y, phi = pi/2 about X
phi = 0
# Drive amplitude
A_m, t_r, t_p, t_f = np.array([3.14159265e+01,
                               2.46263371e-02,
                               7.57653209e-02,
                               2.44662190e-02])
A_m, t_r, t_p, t_f = np.array([31.2786,  0.    ,  0.853 ,  0.    ])
A_m, t_r, t_p, t_f = np.array([36.7054,  0.008 ,  0.084 ,  0.008 ])
A_m = omega_d*0.25#0.270

# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.115.133601
# Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

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
Delta_eps = omega_d*np.sqrt((1 - j0(2*A/omega_0))**2 + j1(2*A/omega_0))

H_c = [[Z/2, [omega_0]*len(dt), r'$F_{zz}$'],
       [X/2, 2*A[1:]*np.cos(omega_d*t[1:] + phi), r'$F_{xx}$']]
H_n = [[Z/2, np.ones_like(dt), r'$F_{zz}$'],
       [X/2, np.ones_like(dt), r'$F_{xx}$']]

omega_STRONG = np.geomspace(1e-4*Omega_R_STRONG, 5e1*omega_0, 500)
omega_STRONG = np.geomspace(1e-2, 1e4, 1004)
# omega_STRONG = omega

NOT_STRONG = ff.PulseSequence(H_c, H_n, dt)
# NOT_STRONG = NOT_STRONG@NOT_STRONG
NOT_STRONG.cache_filter_function(omega_STRONG)
ID20_STRONG = ff.concatenate_periodic(NOT_STRONG, 40)

plotting.plot_bloch_vector_evolution(NOT_STRONG)
print(1 - np.abs(np.trace(NOT_STRONG.total_propagator.conj().T @ X)/2))

# %%% plot
fig, axes = plt.subplots(2, 1, sharex=False, sharey=False,
                         figsize=(figsize_narrow[0], figsize_narrow[0]))

_ = plotting.plot_filter_function(
    ID20, omega_in_units_of_tau=False,
    xscale='log', yscale='log', plot_kw=dict(linewidth=1),
    fig=fig, axes=axes[0]
)
_ = plotting.plot_filter_function(
    ID20_STRONG, omega_in_units_of_tau=False,
    xscale='log', yscale='log', plot_kw=dict(linewidth=1),
    fig=fig, axes=axes[1]
)

subfigs = ('(a)', '(b)')
for ax, subfig, O in zip(axes, subfigs, (Omega_R, Omega_R_STRONG)):
    ax.axvline(O, ls=':', color='tab:grey',
               label=r'$\Omega_R$', zorder=0, linewidth=1)
    ax.axvline(omega_d, ls='--', color='tab:grey',
               label=r'$\omega_0$', zorder=0, linewidth=1)
    ax.tick_params(direction='in', which='both', labelsize=8, zorder=10)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(False)
    ax.set_xlabel(r'$\omega$ (\si{\per\second})')
    ax.set_xlabel(r'$\omega$ (\si{\per\nano\second})')
    # ax.set_ylim(bottom=1e-30, top=1e-10)
    ax.text(0.9, 0.825, subfig, transform=ax.transAxes, fontsize=10)

# ax.axvline(omega_d-O, ls='-.', color='tab:grey', zorder=0, linewidth=1)
# ax.axvline(omega_d+O, ls='-.', color='tab:grey',
#            label=r'$\omega_0\pm\Omega_R$', zorder=0, linewidth=1)
axes[0].get_legend().set_visible(False)
axes[0].set_xlabel('')
# axes[0].set_ylim(top=1e11)
# axes[1].set_ylim(top=1e3)
legend = axes[1].legend(loc='lower left', ncol=2, frameon=False, framealpha=1,
                        bbox_to_anchor=(0, 0.05), fancybox=False, handlelength=1.95)

fig.tight_layout()

fname = 'rabi_driving_weak_vs_strong'
fig.savefig(save_path / Path(fname + '.eps'))
fig.savefig(save_path / Path(fname + '.pdf'))
