# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:57:50 2018

@author: Tobias Hangleiter (tobias.hangleiter@rwth-aachen.de)
"""
import git
import matplotlib
# matplotlib.use('ps')

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib import cycler
from scipy import constants, io, odr

import filter_functions as ff
from filter_functions import plotting, util

from matplotlib import colors, lines
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition)

plt.style.use('publication')
golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (7.05826, 7.05826*golden_ratio)

cycle = plt.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=('-','-.','--',':'))

_force_overwrite = True
# %% Load the data
# thesis_path = Path('/home/tobias/Physik/Master/Thesis')
# thesis_path = Path('Z:/MA/')
# thesis_path = Path('../../../MA')
thesis_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis/')
# project_path = Path('/home/tobias/Physik/Publication/')
# project_path = Path('Z:/Publication/')
# project_path = Path('../../')
project_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Publication/')
name = 'efficient_calculation_of_generalized_filter_functions'
save_path = project_path / name / 'img'

# %% set up the operators
gates = ['X2ID', 'Y2ID', 'CNOT']
struct = {'X2ID': io.loadmat(str(thesis_path / 'data' / 'X2ID.mat')),
          'Y2ID': io.loadmat(str(thesis_path / 'data' / 'Y2ID.mat')),
          'CNOT': io.loadmat(str(thesis_path / 'data' / 'CNOT.mat'))}
eps = {key: np.asarray(struct[key]['eps'], order='C', dtype=float)
       for key in gates}
dt = {key: np.asarray(struct[key]['t'].ravel(), order='C', dtype=float)
      for key in gates}
B = {key: np.asarray(struct[key]['B'].ravel(), order='C', dtype=float)
     for key in gates}
B_avg = {key: struct[key]['BAvg'].ravel() for key in gates}
infid_fast = {key: struct[key]['infid_fast'].ravel() for key in gates}
# B_avg same for all
B_avg = B_avg['X2ID']
T = {key: val.sum() for key, val in dt.items()}

J = {key: np.exp(eps[key]) for key in gates}
ndt = {key: len(dt[key]) for key in gates}

d = 16
H = np.empty((6, d, d), dtype=complex)

Id, Px, Py, Pz = util.paulis
# Exchange Hamiltonians
H[0] = 1/4*sum(util.tensor(P, P, Id, Id) for P in (Px, Py, Pz)).real
H[1] = 1/4*sum(util.tensor(Id, P, P, Id) for P in (Px, Py, Pz)).real
H[2] = 1/4*sum(util.tensor(Id, Id, P, P) for P in (Px, Py, Pz)).real
# Zeeman Hamiltonians
H[3] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-3) +
            util.tensor(Id, Pz, Id, Id) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[4] = 1/4*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[5] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id)*(-1) +
            util.tensor(Id, Id, Id, Pz)*3).real

# Technically there would also be H_0 (the mean magnetic field), but on the
# m_s = 0 subspace it is zero.

# %% Set up Hamiltonian
opers = list(H)

# Reduce to 6x6 subspace
zerospin_subspace_inds = ((3, 5, 6, 9, 10, 12), (3, 5, 6, 9, 10, 12))
d_zerospin_subspace = 6
qubit_subspace_inds = ((1, 2, 3, 4), (1, 2, 3, 4))
opers = [H[np.ix_(*zerospin_subspace_inds)] for H in H]

# Subtract identity to make Hamiltonian traceless (always allowed since we are
# not interested in absolute energies)
opers = [oper - np.trace(oper)/d_zerospin_subspace*np.eye(d_zerospin_subspace)
         for oper in opers]

# The coefficients are the exchange couplings and B-field gradients
c_coeffs = {key: [J[key][0],
                  J[key][1],
                  J[key][2],
                  B[key][0]*np.ones(ndt[key]),
                  B[key][1]*np.ones(ndt[key]),
                  B[key][2]*np.ones(ndt[key])] for key in gates}
# We include the exponential dependence of J on epsilon by a first-order
# derivative (just J back) as noise sensitivity.
n_coeffs = {key: [J[key][0],
                  J[key][1],
                  J[key][2],
                  np.ones(ndt[key]),
                  np.ones(ndt[key]),
                  np.ones(ndt[key])] for key in gates}

identifiers = [r'$\epsilon_{12}$', r'$\epsilon_{23}$', r'$\epsilon_{34}$',
               '$b_{12}$', '$b_{23}$', '$b_{34}$']
H_c = {key: list(zip(opers, val, identifiers))
       for key, val in c_coeffs.items()}
H_n = {key: list(zip(opers, val, identifiers))
       for key, val in n_coeffs.items()}

# Basis for qubit subspace (just pad a two-qubit pauli basis with zeros on the
# leakage space)
qubit_subspace_mask = np.pad(np.ones((4, 4)), 1, 'constant').astype(bool)
qubit_subspace_basis = ff.Basis(
    [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)],
    btype='Pauli'
)
complete_basis = ff.Basis.from_partial(qubit_subspace_basis, btype='Pauli')
# Initialize the PulseSequences
X2ID_complete = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'],
                                 basis=complete_basis)
Y2ID_complete = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'],
                                 basis=complete_basis)
CNOT_complete = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],
                                 basis=complete_basis)
pulses_complete = {'X2ID': X2ID_complete, 'Y2ID': Y2ID_complete,
                   'CNOT': CNOT_complete}

X2ID_subspace = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'],
                                 basis=qubit_subspace_basis)
Y2ID_subspace = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'],
                                 basis=qubit_subspace_basis)
CNOT_subspace = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],
                                 basis=qubit_subspace_basis)
pulses_subspace = {'X2ID': X2ID_subspace, 'Y2ID': Y2ID_subspace,
                   'CNOT': CNOT_subspace}

X2ID_ggm = ff.PulseSequence(H_c['X2ID'], H_n['X2ID'], dt['X2ID'])
Y2ID_ggm = ff.PulseSequence(H_c['Y2ID'], H_n['Y2ID'], dt['Y2ID'])
CNOT_ggm = ff.PulseSequence(H_c['CNOT'], H_n['CNOT'], dt['CNOT'],)
pulses_ggm = {'X2ID': X2ID_ggm, 'Y2ID': Y2ID_ggm, 'CNOT': CNOT_ggm}

for key in gates:
    pulses_subspace[key].diagonalize()
    pulses_complete[key].diagonalize()
    pulses_ggm[key].diagonalize()

# for pulse in pulses_complete.values():
#     pulse._Q = ff.pulse_sequence.closest_unitary(pulse._Q, qubit_subspace_inds)
#     pulse.total_propagator = ff.pulse_sequence.closest_unitary([pulse.total_propagator],
#                                                       qubit_subspace_inds)[0]

# %% constants
delta = 20
n_samples = np.arange(21, 261+delta, delta)
infid = {key: [np.ones(2)] for key in gates}
delta_infid = {key: [] for key in gates}
xi = {key: [] for key in gates}

eps0 = 2.7241e-4
mu_B = constants.value('Bohr magneton')
t_unit = 1e-9
h_unit = 1/(2*np.pi*t_unit)
B_unit = mu_B*0.44/constants.h/h_unit
sigma_eps = 8e-6/eps0
sigma_B = 3e-4*B_unit
# S(f) = S_0 f^{-\alpha}
alpha = np.array([0, 0.7])
# At f = 1 MHz = 1e-3 GHz, S = S_0 = 4e-20 V^2/Hz = 4e-11 1/GHz
# Correspondingly, S(\omega) = A \omega^{-\alpha} such that at
# \omega = 2\pi 10^{-3} GHz, S_0 = 4e-11 1/GHz
S_0 = 4e-11/eps0**2
A = S_0*(2*np.pi*1e-3)**alpha

n = 1001
# %% function to cache projected filter function

d_c = 4
d_l = 2
d = d_c + d_l
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1


def cache_filter_function(pulse, omega, proj=None):
    if proj is None:
        proj = ff.superoperator.liouville_representation(PI_c_op, pulse.basis)

    pulse.cache_filter_function(omega, filter_function=ff.numeric.calculate_filter_function(
        np.einsum('aio,ij->ajo', pulse.get_control_matrix(omega), proj)
    ))


# %% calculate transfer matrix
K_complete = {}
K_ggm = {}
P_complete = {}
P_ggm = {}
for key, pulse in pulses_complete.items():
    K_complete[key] = {}
    K_ggm[key] = {}
    P_complete[key] = {}
    P_ggm[key] = {}
    for i, a in enumerate(alpha):
        omega = np.geomspace(1e-2/T[key], 1e2, n)
        S = A[i]/omega**a
        K_complete[key][a] = ff.numeric.calculate_cumulant_function(pulse, S, omega,
                                                                    show_progressbar=True)
        K_ggm[key][a] = ff.numeric.calculate_cumulant_function(pulses_ggm[key], S, omega,
                                                               show_progressbar=True)
        P_complete[key][a] = ff.error_transfer_matrix(pulse,
                                                      cumulant_function=K_complete[key][a][:3])
        P_ggm[key][a] = ff.error_transfer_matrix(pulses_ggm[key],
                                                 cumulant_function=K_ggm[key][a][:3])

basis_labels = ['']
basis_labels.extend([''.join(tup) for tup in
                     product(['I', 'X', 'Y', 'Z'], repeat=2)][1:])
basis_labels.extend(['$C_{{{}}}$'.format(i) for i in range(16, 36)])

# %% fit high frequency behavior


# def log_fitfun(beta, x):
#     return beta[0]*np.cos(x*beta[1]) + beta[2]


# def fjacb(beta, x):
#     return np.array([np.cos(x*beta[1]),
#                      -beta[0]*x*np.sin(beta[1]*x),
#                      np.ones_like(x)])


# def fjacd(beta, x):
#     return np.array([-beta[0]*beta[1]*np.sin(beta[1]*x)])


# # plt.close('all')
# key = 'CNOT'
# pulse = pulses_subspace[key]
# labels = [rf'$\epsilon_{{{i}{i+1}}}$' for i in range(1, 4)]
# omega = np.linspace(1e-2, 1e2, 300)
# omega = np.geomspace(1/T[key], 1e2, n)
# F = pulses_subspace[key].get_filter_function(omega)
# beta0 = [2, dt[key][0]/pulses_subspace[key].t[-1], 1]
# log_model = odr.Model(log_fitfun, fjacb=fjacb, fjacd=fjacd)
# for i in range(3):
#     logdata = odr.Data(omega*T[key], np.log10(F[i, i].real*omega**2))
#     ODR = odr.ODR(logdata, log_model, beta0, ifixb=[0, 0, 1])
#     output = ODR.run()
#     print(output.beta)

#     fig, ax = plt.subplots()
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.plot(omega*T[key], F[i, i].real, '.--', label=labels[i])
#     ax.plot(omega*T[key], 10**log_fitfun(output.beta, omega*T[key])/omega**2,
#             label=r'$C e^{A\cos(b\omega)}/\omega^2$')
#     ax.plot(omega*T[key], F[i, i].real / 10**log_fitfun(output.beta, omega*T[key]),
#             label=r'FF / fit')
#     ax.legend()
#     ax.set_xlim(min(omega*T[key]), max(omega*T[key]))
#     ax.grid()

# %% inset plot, linear
# omega = np.linspace(0, 1e2, n)
# fig, ax, leg = plotting.plot_filter_function(pulses_subspace[key], omega, [0, 1, 2],
#                                        n_oper_labels=labels,
#                                        xscale='linear', yscale='log',
#                                        figsize=figsize_narrow,
#                                        plot_kw=dict(linewidth=1))
# ax.set_ylim(ax.get_ylim()[0], top=1e8)
# ax.grid()
# ax.legend(frameon=False)
# ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# plt.rcParams['font.size'] = 8
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# ax.tick_params(direction='in', which='both', labelsize=8)
#
# omega = np.linspace(0, 1e2/T[key], n)
# F = pulses_subspace[key].get_filter_function(omega)
# ins_ax = inset_axes(ax, 1, 1)
# inset_position = InsetPosition(ax, [0.15, 0.6, 0.5, 0.4])
# ins_ax.set_axes_locator(inset_position)
#
# for i in range(3):
#     ins_ax.plot(omega*T[key], F[i], linewidth=1)
#
# ins_ax.set_xlim(0, omega.max()*T[key])
# ins_ax.set_yscale('linear')
# ins_ax.tick_params(direction='in', which='both', labelsize=6)
#
# fig.tight_layout()
#
# fname = f'filter_function_{key}_linear_with_inset'
# fig.savefig(save_path / Path(fname + '.png'), dpi=300)

# %% Pulse Train
# plt.rcParams['font.size'] = 8
# fig, ax, leg = plotting.plot_pulse_train(
#     pulses_subspace[key], c_oper_identifiers=identifiers[:3],
#     figsize=figsize_narrow, plot_kw=dict(linewidth=1)
# )
# ax.set_xlim(0, 50)
# ax.set_xlabel(r'$t$ (ns)')
# ax.set_ylabel(r'$J(\epsilon_{ij})$ (ns$^{-1}$)')
# fig.tight_layout()
# fname = f'pulse_train_{key}'
# fig.savefig(save_path / Path(fname + '.eps'))

# %% inset plot, log
# omega = np.geomspace(1/T[key], 1e2, n)
# fig, ax, leg = plotting.plot_filter_function(
#     pulses_subspace[key], omega, n_oper_identifiers=identifiers[:3],
#     xscale='log', yscale='log', omega_in_units_of_tau=False,
#     figsize=figsize_narrow, plot_kw=dict(linewidth=1)
# )
# # ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# # ax.set_ylim(1e-15, ax.get_ylim()[1])
# ax.legend(identifiers[:3], frameon=False, loc='upper right')
# ax.grid()
# # ax.set_xlabel(ax.get_xlabel() + ' (GHz)')
# ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# # ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# ax.tick_params(direction='out', which='both', labelsize=8)
# ax.set_ylim(bottom=1.0501675711991102e-09, top=1e3)

# omega = np.linspace(0, 1e2/T[key], n)
# F = pulses_subspace[key].get_filter_function(omega)
# ins_ax = inset_axes(ax, 1, 1)
# inset_position = InsetPosition(ax, [0.1, 0.1, 0.5, 0.5])
# ins_ax.set_axes_locator(inset_position)

# for i in range(3):
#     ins_ax.plot(omega, F[i, i], linewidth=1)

# ins_ax.set_xlim(0, omega.max())
# ins_ax.set_yscale('linear')
# ins_ax.tick_params(direction='in', which='both', labelsize=6)
# ins_ax.spines['right'].set_visible(False)
# ins_ax.spines['top'].set_visible(False)
# ins_ax.patch.set_alpha(0)

# fig.tight_layout()

# fname = f'filter_function_{key}_log_with_inset'
# fig.savefig(save_path / Path(fname + '.eps'))

# %% Pulse train + filter function in one figure
# fig, ax = plt.subplots(2, 1,
#                        figsize=(figsize_narrow[0], figsize_narrow[1]*1.5),
#                        gridspec_kw={'height_ratios': [1, 2]})

# *_, leg = plotting.plot_pulse_train(
#     pulses_subspace[key], c_oper_identifiers=identifiers[:3], axes=ax[0],
#     fig=fig, plot_kw=dict(linewidth=1)
# )

# ax[0].set_xlabel(r'$t$ (ns)')
# ax[0].set_ylabel(r'$J(\epsilon_{ij})$ (ns$^{-1}$)')
# ax[0].grid(False)
# ax[0].text(0.02, 0.8, '(a)', transform=ax[0].transAxes)
# leg.remove()

# omega = np.geomspace(1/T[key], 1e2, n)
# *_, leg = plotting.plot_filter_function(
#     pulses_subspace[key], omega, n_oper_identifiers=identifiers[:3],
#     yscale='log', omega_in_units_of_tau=False, axes=ax[1], fig=fig,
#     plot_kw=dict(linewidth=1)
# )
# # ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# # ax.set_ylim(1e-15, ax.get_ylim()[1])
# ax[1].legend(identifiers[:3], frameon=False, loc='upper right')
# ax[1].grid(False)
# ax[1].set_xlabel(ax[1].get_xlabel() + r' (ns$^{-1}$)')
# ax[1].set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# # ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# ax[1].tick_params(direction='out', which='both', labelsize=8)
# ax[1].set_ylim(bottom=1.0501675711991102e-09, top=1e3)

# omega = np.linspace(0, 1e2/T[key], n)
# F = pulses_subspace[key].get_filter_function(omega)
# ins_ax = inset_axes(ax[1], 1, 1)
# inset_position = InsetPosition(ax[1], [0.1, 0.15, 0.5, 0.5])
# ins_ax.set_axes_locator(inset_position)

# for i in range(3):
#     ins_ax.plot(omega, F[i, i], linewidth=1)

# ins_ax.set_xlim(0, omega.max())
# ins_ax.set_yscale('linear')
# ins_ax.tick_params(direction='in', which='both', labelsize=6)
# ins_ax.spines['right'].set_visible(False)
# ins_ax.spines['top'].set_visible(False)
# ins_ax.patch.set_alpha(0)

# ax[1].text(0.02, 0.9, '(b)', transform=ax[1].transAxes)

# fig.tight_layout(h_pad=0)

# fname = f'pulse_train_filter_function_{key}'
# fig.savefig(save_path / Path(fname + '.eps'))
# fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# %% transfer matrices all in one, logscale
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = plotting.plot_error_transfer_matrix(
#             P=P[key][a][:3], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='log', basis_labels=basis_labels, basis_labelsize=4,
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_log_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'))
#         fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# %% transfer matrices all in one, linear scale
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = plotting.plot_error_transfer_matrix(
#             P=P[key][a][:3], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='linear', basis_labels=basis_labels, basis_labelsize=4,
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_linear_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'))
#         fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# %% transfer matrices all in one, logscale, subspace only
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = plotting.plot_error_transfer_matrix(
#             P=P[key][a][:3, :16, :16], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='log', basis_labels=basis_labels[:16],
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_log_subspace_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'))
#         fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# %% transfer matrices all in one, linear scale, subspace only
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = plotting.plot_error_transfer_matrix(
#             P=P[key][a][:3, :16, :16], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='linear', basis_labels=basis_labels[:16],
#             imshow_kw=dict(origin='upper')
#         )
#
#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_linear_subspace_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'))
#         fig.savefig(save_path / Path(fname + '.pdf'))
#
# plt.close('all')

# %% everything together
key = 'CNOT'
a = 0.7
K = K_complete[key][a][:3].real
colorscale = 'linear'
# colorscale = 'log'

fig = plt.figure(constrained_layout=False, figsize=figsize_wide)
gs = GridSpec(5, 9, figure=fig)

# Pulse train
pt_ax = fig.add_subplot(gs[:2, :4])
*_, leg = plotting.plot_pulse_train(
    pulses_complete[key], c_oper_identifiers=identifiers[:3],
    axes=pt_ax, fig=fig, cycler=cycle
)
pt_ax.set_xlabel(r'$t$ (ns)')
pt_ax.set_ylabel(r'$J(\epsilon_{ij})$ (ns$^{-1}$)')
pt_ax.grid(False)
pt_ax.text(0.015, 0.875, '(a)', transform=pt_ax.transAxes, fontsize=10)
leg.remove()

# Filter functions
ff_ax = fig.add_subplot(gs[:2, 4:])
omega = np.geomspace(1/T[key], 1e2, n)

cache_idx = np.diag([0, *[1]*15, *[0]*20])
cache_filter_function(pulses_complete[key], omega, cache_idx)

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=ff_ax, fig=fig,
    cycler=cycle
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
ff_ax.legend(identifiers[:3], frameon=False, loc='upper right')
ff_ax.grid(False)
ff_ax.set_xlabel(ff_ax.get_xlabel() + r' (ns$^{-1}$)')
ff_ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
ff_ax.tick_params(direction='out', which='both', labelsize=8)
ff_ax.set_ylim(bottom=1e-9, top=1e3)

omega = np.linspace(0, 1e2/T[key], n)
cache_filter_function(pulses_complete[key], omega, cache_idx)

F = pulses_complete[key].get_filter_function(omega)
ins_ax = inset_axes(ff_ax, 1, 1)
inset_position = InsetPosition(ff_ax, [0.1, 0.15, 0.5, 0.5])
ins_ax.set_axes_locator(inset_position)

for i, props in enumerate(cycle[:3]):
    ins_ax.plot(omega, F[i, i].real, linewidth=1, **props)

ins_ax.set_xlim(0, omega.max())
ins_ax.set_yscale('linear')
ins_ax.tick_params(direction='in', which='both', labelsize=6)
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

ff_ax.text(0.015, 0.875, '(b)', transform=ff_ax.transAxes, fontsize=10)

# Transfer matrices
Kmax = abs(K[:, :16, :16]).max()
Kmin = -Kmax
if colorscale == 'log':
    linthresh = np.abs(K).mean()/10
    norm = colors.SymLogNorm(linthresh=linthresh, vmin=Kmin, vmax=Kmax, base=10)
elif colorscale == 'linear':
    norm = colors.Normalize(vmin=Kmin, vmax=Kmax)

imshow_kw = {}
imshow_kw.setdefault('origin', 'upper')
imshow_kw.setdefault('interpolation', 'nearest')
imshow_kw.setdefault('cmap', plt.get_cmap('RdBu'))
imshow_kw.setdefault('norm', norm)
basis_labels = np.array([
    ''.join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=2)
])

tm_axes = [fig.add_subplot(gs[2:, 3*i:3*(i+1)]) for i in range(3)]
subfigs = ['(c)', '(d)', '(e)']
for i, (n_oper_identifier, tm_ax) in enumerate(zip(identifiers, tm_axes)):
    if i == 2:
        idx = [np.ravel_multi_index((j, i), (4, 4))
               for i in range(4) for j in range(4)]
    else:
        idx = [np.ravel_multi_index((i, j), (4, 4))
               for i in range(4) for j in range(4)]
    im = tm_ax.imshow(K[i][..., idx][idx, ...], **imshow_kw)
    tm_ax.text(0.01, 1.05, subfigs[i] + ' ' + n_oper_identifier,
               transform=tm_ax.transAxes, fontsize=10)
    tm_ax.set_xticks(np.arange(16))
    tm_ax.set_yticks(np.arange(16))
    tm_ax.set_xticklabels(basis_labels[idx], rotation='vertical', fontsize=8)
    tm_ax.set_yticklabels(basis_labels[idx], fontsize=8)
    tm_ax.spines['left'].set_visible(False)
    tm_ax.spines['right'].set_visible(False)
    tm_ax.spines['top'].set_visible(False)
    tm_ax.spines['bottom'].set_visible(False)

gs.tight_layout(fig, h_pad=0., w_pad=0., pad=0.)

# cb_ax = fig.add_subplot(gs[1:, -1])
# cb_ax = fig.add_axes([1, 0.105, 0.015, 0.375])
cb_ax = fig.add_axes([1, 0.105, 0.015, 0.405])
# divider = make_axes_locatable(tm_axes)
# cb_ax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(im, ax=tm_axes, cax=cb_ax, fraction=0.045, pad=0.04)
cb.set_label(r"Cumulant function $\mathcal{K}_{\epsilon_{ij}}(\tau)$")

a = '-'.join(str(a).split('.'))
fname = f'all_in_one_alpha-{a}_linear_complete_{key}'
# fname = f'all_in_one_alpha-{a}_log_complete_{key}'

for ext in ('pdf', 'eps', 'png'):
    if (not (file := save_path / '.'.join([fname, ext])).exists()
            or _force_overwrite):
        fig.savefig(file)

# %% filter function unitary vs complete
key = 'CNOT'

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                         figsize=(figsize_wide[0], figsize_wide[1]/1.75))
omega = np.geomspace(1/T[key], 4, n)

# Excluding identity element
cache_filter_function(pulses_complete[key], omega,
                      np.diag([0, *[1]*15, *[0]*20]))

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=axes[0], fig=fig,
    cycler=cycle
)
leg.remove()
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
axes[0].grid(False)
axes[0].set_xlabel(axes[0].get_xlabel() + r' (ns$^{-1}$)')
axes[0].set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
axes[0].tick_params(direction='out', which='both', labelsize=8)
axes[0].set_ylim(bottom=1e-2, top=1e2)
# axes[0].set_title(r'Excluding $\mathrm{diag}(1,1,1,1,0,0)$')
axes[0].text(0.90, 0.90, '(a)', transform=axes[0].transAxes, fontsize=10)


# Including identity element
cache_filter_function(pulses_complete[key], omega,
                      np.diag([1, *[1]*15, *[0]*20]))

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=axes[1], fig=fig,
    cycler=cycle
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
axes[1].legend(identifiers[:3], frameon=True, loc='lower left', framealpha=1)
axes[1].grid(False)
axes[1].set_xlabel(axes[1].get_xlabel() + r' (ns$^{-1}$)')
axes[1].set_ylabel('')
# axes[1].set_title(r'Including $\mathrm{diag}(1,1,1,1,0,0)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# axes[1].tick_params(direction='out', which='both', labelsize=8)
# axes[1].set_ylim(bottom=1e-9, top=1e3)
axes[1].text(0.90, 0.90, '(b)', transform=axes[1].transAxes, fontsize=10)
axes[1].legend(handlelength=1.95, frameon=False)

fig.tight_layout()
fname = f'filter_function-{key}_unitary_v_complete'
for ext in ('pdf', 'eps', 'png'):
    if (not (file := save_path / '.'.join([fname, ext])).exists()
            or _force_overwrite):
        fig.savefig(file)

# %% fidelities
# repo = git.Repo('Z:/Code/filter_functions')
repo = git.Repo('~/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'fidelities_commit-{}.txt'.format(sha[:7])

PI_c_liouville_ggm = ff.superoperator.liouville_representation(PI_c_op,
                                                               ff.Basis.ggm(6))

with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('==========================================================\n')
    f.write('calculated using ff.infidelity (subspace)\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_subspace.items()):
            pulse.d = 4
            omega = np.geomspace(1e-2/T[key], 1e2, n)
            S = A[i]/omega**a
            infidelities[i, j] = ff.infidelity(pulse, S, omega)
            f.write(f'{key}\t{a}\t\t{infidelities[i, j, :3].sum():.3e}\t\t\t' +
                    f'{infidelities[i, j, :3].sum()*4/5:.3e}\n')
            pulse.d = 6
    f.write('==========================================================\n')
    f.write('calculated using ff.error_transfer_matrix, separated basis\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = 1 - np.einsum('...ii',
                                  P_complete[key][a][:16, :16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using cumulant function, excl. identity\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = np.einsum('...ii',
                              -K_complete[key][a][:3, 1:16, 1:16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using cumulant function, separated basis\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = np.einsum('...ii',
                              -K_complete[key][a][:3, :16, :16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('calculated using ff.error_transfer_matrix, excl. identity\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            infid = 1 - np.einsum('...ii',
                                  P_complete[key][a][1:16, 1:16]).sum().real/d_c**2
            f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
    f.write('==========================================================\n')
    f.write('MC\n')
    f.write('==========================================================\n')
    f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
    for i, a in enumerate(alpha):
        f.write('----------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            F_e = infid_fast[key][i]
            f.write(f'{key}\t{a}\t\t{F_e:.3e}\t\t\t{F_e*4/5:.3e}\n')
    f.write('==========================================================')

# %% leakage
repo = git.Repo('~/Code/filter_functions')
# repo = git.Repo('Z:/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'leakage_commit-{}.txt'.format(sha[:7])
pulse = pulses_complete['CNOT']

d_c = 4
d_l = 2
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1
PI_c_vec = ff.basis.expand(PI_c_op, pulse.basis)
PI_l_vec = ff.basis.expand(PI_l_op, pulse.basis)


def leakage(E):
    return (PI_l_vec.T @ E @ PI_c_vec/d_c).real


def seepage(E):
    return (1 - PI_l_vec.T @ E @ PI_l_vec/d_l).real


with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('====================================================\n')
    f.write('                   CNOT (eps_23)\n')
    f.write('====================================================\n')
    f.write('----------------------------------------------------\n')
    f.write('\t\tSystematic (L(Q))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Q_liouville)
        L_l = seepage(Q_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')

    f.write('----------------------------------------------------\n')
    f.write('\t\tDecoherence (L(Uerr))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Uerr_liouville)
        L_l = seepage(Uerr_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')

    f.write('----------------------------------------------------\n')
    f.write('\t\tBoth (L(Q @ Uerr))\n')
    f.write('alpha\tLeakage\t\t\tSeepage\n')
    f.write('----------------------------------------------------\n')
    for i, a in enumerate(alpha):
        Q_liouville = pulse.total_propagator_liouville.real
        Uerr_liouville = np.eye((d_c + d_l)**2) + K_complete['CNOT'][a][1]
        L_c = leakage(Q_liouville @ Uerr_liouville)
        L_l = seepage(Q_liouville @ Uerr_liouville)
        f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')
    f.write('====================================================')

# %% Leakage Wood & Gambetta


def apply_channel(dmat, ops):
    if dmat.shape[0] == d:
        res = ops @ dmat @ ops.conj().swapaxes(-1, -2)
        return res.sum(0) if ops.ndim == 3 else res
    else:
        return ops @ dmat


def leakage_op(channel, PI_l, PI_c, d_c):
    return np.trace(PI_l @ apply_channel(PI_c, channel)).real/d_c


def leakage_vec(channel, PI_lvec, PI_cvec, d_c):
    return np.einsum('i,...i', PI_lvec, apply_channel(PI_cvec, channel)).real/d_c


d_c = 4
d_l = 2
d = d_c + d_l
# basis = ff.Basis.ggm(d)

d_c = 4
d_l = 2
PI_c_op = np.zeros((6, 6))
PI_l_op = np.zeros((6, 6))
PI_c_op[range(1, 5), range(1, 5)] = 1
PI_l_op[(0, 5), (0, 5)] = 1

# %%% basis transform
computational_basis_vec = np.eye(d**2)
computational_basis_c = ff.Basis(
    [np.reshape(qt.basis(d**2, i).full(), (d, d), order='C')
     for i in range(d**2)],
)
computational_basis_r = ff.Basis(
    [np.reshape(qt.basis(d**2, i).full(), (d, d), order='F')
     for i in range(d**2)],
)

# Convert from column stacking to complete basis and v.v.
T_sc_complete = np.einsum('ai,aj', computational_basis_vec,
                          ff.basis.expand(computational_basis_c, complete_basis))
# Convert from row stacking to complete basis and v.v.
T_sr_complete = np.einsum('ai,aj', computational_basis_vec,
                          ff.basis.expand(computational_basis_r, complete_basis))

# Convert from column stacking to GGM basis and v.v.
T_sc_ggm = np.einsum('ai,aj', computational_basis_vec,
                     ff.basis.expand(computational_basis_c, ff.Basis.ggm(d)))
# Convert from row stacking to GGM basis and v.v.
T_sr_ggm = np.einsum('ai,aj', computational_basis_vec,
                     ff.basis.expand(computational_basis_r, ff.Basis.ggm(d)))

# Convert from complete basis to GGM basis and v.v.
T_ssp = np.einsum('ai,aj', computational_basis_vec,
                  ff.basis.expand(complete_basis, ff.Basis.ggm(d)))

a = np.random.randn(6, 6) + np.random.randn(6, 6)*1j
a += a.conj().T

A_rvec = np.zeros((36), dtype=complex)
eye = np.eye(6)
for i in range(6):
    for j in range(6):
        A_rvec += a[i, j]*np.kron(eye[i], eye[j]).T

A_cvec = np.zeros((36), dtype=complex)
eye = np.eye(6)
for i in range(6):
    for j in range(6):
        A_cvec += a[i, j]*np.kron(eye[j], eye[i])

A_svec_complete = ff.basis.expand(a, complete_basis)
A_svec_ggm = ff.basis.expand(a, ff.Basis.ggm(d))

PI_cvec_complete = ff.basis.expand(PI_c_op, complete_basis)
PI_lvec_complete = ff.basis.expand(PI_l_op, complete_basis)
PI_cvec_ggm = ff.basis.expand(PI_c_op, ff.Basis.ggm(d))
PI_lvec_ggm = ff.basis.expand(PI_l_op, ff.Basis.ggm(d))

P_col = {key: {a: T_sc_ggm @ P @ T_sc_ggm.conj().T for a, P in p.items()}
         for key, p in K_ggm.items()}
P_row = {key: {a: T_sr_ggm @ P @ T_sr_ggm.conj().T for a, P in p.items()}
         for key, p in K_ggm.items()}

# %%% test correct transform of vectors
print('complete vs row:', np.allclose(T_sr_complete @ A_svec_complete, A_rvec))
print('complete vs col:', np.allclose(T_sc_complete @ A_svec_complete, A_cvec))
print('ggm vs row:', np.allclose(T_sr_ggm @ A_svec_ggm, A_rvec))
print('ggm vs col:', np.allclose(T_sc_ggm @ A_svec_ggm, A_cvec))
print('inverse complete vs row:', np.allclose(T_sr_complete.conj().T @ A_rvec, A_svec_complete))
print('inverse complete vs col:', np.allclose(T_sc_complete.conj().T @ A_cvec, A_svec_complete))
print('inverse ggm vs row:', np.allclose(T_sr_ggm.conj().T @ A_rvec, A_svec_ggm))
print('inverse ggm vs col:', np.allclose(T_sc_ggm.conj().T @ A_cvec, A_svec_ggm))

# %%% test correct transform of superoperators
S_s_complete = -K_complete['CNOT'][0.0].sum(0)
S_s_ggm = -K_ggm['CNOT'][0.0].sum(0)
S_c = -P_col['CNOT'][0.0].sum(0)
S_r = -P_row['CNOT'][0.0].sum(0)

print('superoperator ggm to complete:', np.allclose(T_ssp@S_s_ggm@T_ssp.conj().T, S_s_complete))
print('superoperator complete to ggm:', np.allclose(T_ssp.conj().T@S_s_complete@T_ssp, S_s_ggm))

# %%% leakage
for a in (0.0, 0.7):
    L = leakage_vec(np.eye(d**2) + K_complete['CNOT'][a], PI_lvec_complete,
                    PI_cvec_complete, d_c)
    print(f'leakage complete\t {a}:', L[:3])
    L = leakage_vec(np.eye(d**2) + K_ggm['CNOT'][a], PI_lvec_ggm,
                    PI_cvec_ggm, d_c)
    print(f'leakage ggm\t\t {a}:', L[:3])

# %%% process fidelity
for key in pulses_ggm.keys():
    for a in (0.0, 0.7):
        i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16])/d_c**2
        print(f'fidelity complete \t {key} {a}:', i.real)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_col[key][a][:3])/d_c**2
        print(f'fidelity col\t\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_row[key][a][:3])/d_c**2
        print(f'fidelity row\t\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji',
                      ff.superoperator.liouville_representation(PI_c_op, complete_basis),
                      -K_complete[key][a][:3])/d_c**2
        print(f'fidelity complete proj.\t {key} {a}:', i.real)
        i = np.einsum('ij,...ji',
                      ff.superoperator.liouville_representation(PI_c_op, ff.Basis.ggm(d)),
                      -K_ggm[key][a][:3])/d_c**2
        print(f'fidelity ggm proj.\t {key} {a}:', i.real)

# %%% average gate fidelity without leakage
for key in pulses_complete.keys():
    for a in (0.0, 0.7):
        l = leakage_vec(np.eye(d**2) + K_complete['CNOT'][a][:3],
                        PI_lvec_complete, PI_cvec_complete, d_c)
        i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16])/d_c**2
        f = (d_c*(1 - i.real) + 1 - l.real)/(d_c + 1)
        print(f'avg gate fidelity complete \t {key} {a}:', 1 - f)
        i = np.einsum('ij,...ji', np.kron(PI_c_op, PI_c_op), -P_col[key][a][:3])/d_c**2
        f = (d_c*(1 - i.real) + 1 - l.real)/(d_c + 1)
        print(f'avg gate fidelity col\t\t {key} {a}:', 1 - f)

# %%% save average gate fidelity without leakage
# repo = git.Repo('Z:/Code/filter_functions')
sha = repo.head.object.hexsha

fname = 'fidelity-leakage_commit-{}.txt'.format(sha[:7])
pulse = pulses_complete['CNOT']

with open(save_path / fname, 'w+') as f:
    infidelities = np.empty((2, 3, 6))
    f.write(f'Git commit hash: {sha}\n\n')
    f.write('============================================================\n')
    f.write('Only eps contributions\n')
    f.write('============================================================\n\n')

    f.write('============================================================\n')
    f.write('calculated using ff.error_transfer_matrix separated basis\n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for k, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            l = leakage_vec(np.eye(d**2) + K_complete[key][a][:3],
                            PI_lvec_complete, PI_cvec_complete, d_c).real
            proj = ff.superoperator.liouville_representation
            i = np.einsum('...ii', -K_complete[key][a][:3, :16, :16]).real/d_c**2
            iav = 1 - (d_c*(1 - i) + 1 - l)/(d_c + 1)
            f.write(f'{key}\t{a}\t\t{i.sum():.3e}\t{i.sum()*4/5:.3e}\t{iav.sum():.3e}\n')
    f.write('============================================================\n')
    f.write('calculated using ff.error_transfer_matrix projected \n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for k, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            l = leakage_vec(np.eye(d**2) + K_complete[key][a][:3],
                            PI_lvec_complete, PI_cvec_complete, d_c).real
            infid = np.einsum('ij,...ji', PI_c_liouville_ggm,
                              -K_ggm[key][a][:3]).real/d_c**2
            iav = 1 - (d_c*(1 - infid) + 1 - l)/(d_c + 1)
            f.write(f'{key}\t{a}\t\t{infid.sum():.3e}\t{infid.sum()*4/5:.3e}\t{iav.sum():.3e}\n')
    f.write('============================================================\n')
    f.write('\t\t\tMC\n')
    f.write('============================================================\n')
    f.write('Gate\talpha\tF_ent\t\tF_avg\t\tF_avg sans L\n')
    for i, a in enumerate(alpha):
        f.write('------------------------------------------------------------\n')
        for j, (key, pulse) in enumerate(pulses_complete.items()):
            F_e = infid_fast[key][i]
            f.write(f'{key}\t{a}\t\t{F_e:.3e}\t{F_e*4/5:.3e}\t{F_e*4/5:.3e}\n')
    f.write('============================================================')
