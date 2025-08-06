# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:57:50 2018

@author: Tobias Hangleiter (tobias.hangleiter@rwth-aachen.de)
"""
from itertools import product
from pathlib import Path

import git

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants, io, odr

import filter_functions as ff
from filter_functions import util

from matplotlib import colors, lines
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition)

plt.style.use('thesis')
# Hotfix latex preamble
for key in ('text.latex.preamble', 'pgf.preamble'):
    plt.rcParams.update({key: '\n'.join(plt.rcParams.get(key).split('|'))})

golden_ratio = (np.sqrt(5) - 1.)/2.
figsize_narrow = (3.40457, 3.40457*golden_ratio)
figsize_wide = (5.65071, 5.65071*golden_ratio)
figsize_twocol = (7.05826, 7.05826*golden_ratio)
exts = ('pdf', 'eps', 'pgf')
# %% Load the data and set up the operators
thesis_path = Path('/home/tobias/Physik/Master/Thesis')
# thesis_path = Path('C:/Users/Tobias/Documents/Uni/Physik/Master/Thesis')
# thesis_path = Path('Z:/MA/')
data_path = thesis_path / 'data'
save_path = thesis_path / 'thesis' / 'img'

gates = ['X2ID', 'Y2ID', 'CNOT']
struct = {'X2ID': io.loadmat(str(data_path / 'X2ID.mat')),
          'Y2ID': io.loadmat(str(data_path / 'Y2ID.mat')),
          'CNOT': io.loadmat(str(data_path / 'CNOT.mat'))}
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

Id, Px, Py, Pz = util.P_np
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
    skip_check=True, btype='Pauli'
)
complete_basis = ff.Basis(
    [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)[1:]]
)

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

for key in gates:
    pulses_subspace[key].diagonalize()
    pulses_complete[key].diagonalize()

# for pulse in pulses_complete.values():
#     pulse._Q = ff.pulse_sequence.closest_unitary(pulse._Q, qubit_subspace_inds)
#     pulse.total_Q = ff.pulse_sequence.closest_unitary([pulse.total_Q],
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

# %% fit high frequency behavior


def log_fitfun(beta, x):
    return beta[0]*np.cos(x*beta[1]) + beta[2]


def fjacb(beta, x):
    return np.array([np.cos(x*beta[1]),
                     -beta[0]*x*np.sin(beta[1]*x),
                     np.ones_like(x)])


def fjacd(beta, x):
    return np.array([-beta[0]*beta[1]*np.sin(beta[1]*x)])


# plt.close('all')
key = 'CNOT'
pulse = pulses_subspace[key]
labels = [rf'$\epsilon_{{{i}{i+1}}}$' for i in range(1, 4)]
omega = np.linspace(1e-2, 1e2, 300)
omega = np.geomspace(1/T[key], 1e2, n)
F = pulses_subspace[key].get_filter_function(omega)
beta0 = [2, dt[key][0]/pulses_subspace[key].t[-1], 1]
log_model = odr.Model(log_fitfun, fjacb=fjacb, fjacd=fjacd)
for i in range(3):
    logdata = odr.Data(omega*T[key], np.log10(F[i, i].real*omega**2))
    ODR = odr.ODR(logdata, log_model, beta0, ifixb=[0, 0, 1])
    output = ODR.run()
    print(output.beta)

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot(omega*T[key], F[i, i].real, '.--', label=labels[i])
    ax.plot(omega*T[key], 10**log_fitfun(output.beta, omega*T[key])/omega**2,
            label=r'$C e^{A\cos(b\omega)}/\omega^2$')
    ax.plot(omega*T[key], F[i, i].real / 10**log_fitfun(output.beta, omega*T[key]),
            label=r'FF / fit')
    ax.legend()
    ax.set_xlim(min(omega*T[key]), max(omega*T[key]))
    ax.grid()

# %% inset plot, linear
# omega = np.linspace(0, 1e2, n)
# fig, ax, leg = ff.plot_filter_function(pulses_subspace[key], omega, [0, 1, 2],
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

# # %% Pulse Train
# plt.rcParams['font.size'] = 8
# fig, ax, leg = ff.plot_pulse_train(
#     pulses_subspace[key], c_oper_identifiers=identifiers[:3],
#     figsize=figsize_narrow, plot_kw=dict(linewidth=1)
# )
# ax.set_xlim(0, 50)
# ax.set_xlabel(r'$t$ (ns)')
# ax.set_ylabel(r'$J(\epsilon_{ij})$ (ns$^{-1}$)')
# fig.tight_layout()
# fname = f'pulse_train_{key}'
# fig.savefig(save_path / Path(fname + '.eps'), dpi=600)

# %% inset plot, log
# omega = np.geomspace(1/T[key], 1e2, n)
# fig, ax, leg = ff.plot_filter_function(
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

# # fname = f'filter_function_{key}_log_with_inset'
# # fig.savefig(save_path / Path(fname + '.eps'), dpi=600)

# # %% Pulse train + filter function in one figure
# fig, ax = plt.subplots(2, 1,
#                        figsize=(figsize_narrow[0], figsize_narrow[1]*1.5),
#                        gridspec_kw={'height_ratios': [1, 2]})

# *_, leg = ff.plot_pulse_train(
#     pulses_subspace[key], c_oper_identifiers=identifiers[:3], axes=ax[0],
#     fig=fig
# )

# ax[0].set_xlabel(r'$t$ (ns)')
# ax[0].set_ylabel(r'$J(\epsilon_{ij})$ (ns$^{-1}$)')
# ax[0].grid(False)
# ax[0].text(0.02, 0.8, '(a)', transform=ax[0].transAxes)
# leg.remove()

# omega = np.geomspace(1/T[key], 1e2, n)
# *_, leg = ff.plot_filter_function(
#     pulses_subspace[key], omega, n_oper_identifiers=identifiers[:3],
#     yscale='log', omega_in_units_of_tau=False, axes=ax[1], fig=fig
# )
# # ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# # ax.set_ylim(1e-15, ax.get_ylim()[1])
# ax[1].legend(identifiers[:3], frameon=False, loc='upper right')
# ax[1].grid(False)
# ax[1].set_xlabel(ax[1].get_xlabel() + r' (ns$^{-1}$)')
# ax[1].set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# # ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
# ax[1].tick_params(direction='out', which='both')
# ax[1].set_ylim(bottom=1.0501675711991102e-09, top=1e3)

# omega = np.linspace(0, 1e2/T[key], n)
# F = pulses_subspace[key].get_filter_function(omega)
# ins_ax = inset_axes(ax[1], 1, 1)
# inset_position = InsetPosition(ax[1], [0.1, 0.15, 0.5, 0.5])
# ins_ax.set_axes_locator(inset_position)

# for i in range(3):
#     ins_ax.plot(omega, F[i, i])

# ins_ax.set_xlim(0, omega.max())
# ins_ax.set_yscale('linear')
# ins_ax.tick_params(direction='in', which='both', labelsize='small')
# ins_ax.spines['right'].set_visible(False)
# ins_ax.spines['top'].set_visible(False)
# ins_ax.patch.set_alpha(0)

# ax[1].text(0.02, 0.9, '(b)', transform=ax[1].transAxes)

# fig.tight_layout(h_pad=0)

# fname = f'pulse_train_filter_function_{key}'
# for ext in exts:
#     fig.savefig(save_path / ext / '.'.join((fname, ext)))

# plt.close('all')
# %% calculate transfer matrix
u_kl = {}
P = {}
for key, pulse in pulses_complete.items():
    u_kl[key] = {}
    P[key] = {}
    for i, a in enumerate(alpha):
        omega = np.geomspace(1/T[key], 1e2, n)
        S = A[i]/omega**a
        S, omega = ff.util.symmetrize_spectrum(S, omega)
        u_kl[key][a] = ff.numeric.calculate_error_vector_correlation_functions(
            pulse, S, omega
        )
        P[key][a] = ff.error_transfer_matrix(pulse, S, omega)

basis_labels = ['']
basis_labels.extend([''.join(tup) for tup in
                     product(['I', 'X', 'Y', 'Z'], repeat=2)][1:])
basis_labels.extend(['$C_{{{}}}$'.format(i) for i in range(16, 36)])
# %% transfer matrices all in one, logscale
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = ff.plot_error_transfer_matrix(
#             P=P[key][a][:3], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='log', basis_labels=basis_labels, basis_labelsize=4,
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_log_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'), dpi=600)
#         fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# %% transfer matrices all in one, linear scale
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = ff.plot_error_transfer_matrix(
#             P=P[key][a][:3], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='linear', basis_labels=basis_labels, basis_labelsize=4,
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_linear_{key}'
#         fig.tight_layout()
#         for ext in exts:
#             fig.savefig(save_path / ext / '.'.join((fname, ext)))
#
#
# plt.close('all')
# %% transfer matrices all in one, logscale, subspace only
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = ff.plot_error_transfer_matrix(
#             P=P[key][a][:3, :16, :16], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='log', basis_labels=basis_labels[:16],
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_log_subspace_{key}'
#         fig.tight_layout()
#         fig.savefig(save_path / Path(fname + '.eps'), dpi=600)
#         fig.savefig(save_path / Path(fname + '.pdf'))

# plt.close('all')
# # %% transfer matrices all in one, linear scale, subspace only
# for key in pulses_complete.keys():
#     for i, a in enumerate(alpha):
#         fig, grid = ff.plot_error_transfer_matrix(
#             P=P[key][a][:3, :16, :16], n_oper_identifiers=identifiers[:3],
#             figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#             colorscale='linear', basis_labels=basis_labels[:16],
#             imshow_kw=dict(origin='upper')
#         )

#         a = '-'.join(str(a).split('.'))
#         fname = f'error_transfer_matrix_alpha-{a}_linear_subspace_{key}'
#         fig.tight_layout()
#         for ext in exts:
#             fig.savefig(save_path / ext / '.'.join((fname, ext)))


# plt.close('all')

# %% everything together
key = 'CNOT'
a = 0.7
Q = P[key][a][:3]
colorscale = 'linear'

fig = plt.figure(constrained_layout=False, figsize=figsize_wide)
gs = GridSpec(5, 9, figure=fig)

# Pulse train
pt_ax = fig.add_subplot(gs[:2, :4])
*_, leg = ff.plot_pulse_train(
    pulses_subspace[key], c_oper_identifiers=identifiers[:3],
    axes=pt_ax, fig=fig,
)
pt_ax.set_xlabel(r'$t$ (\si{\nano\second})')
pt_ax.set_ylabel(r'$J(\epsilon_{ij})$ (\si{\per\nano\second})')
pt_ax.grid(False)
pt_ax.text(0.02, 0.85, '(a)', transform=pt_ax.transAxes, fontsize=10)
leg.remove()

# Filter functions
ff_ax = fig.add_subplot(gs[:2, 4:])
omega = np.geomspace(1/T[key], 1e2, n)
*_, leg = ff.plot_filter_function(
    pulses_subspace[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=ff_ax, fig=fig,
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
ff_ax.legend(identifiers[:3], frameon=False, loc='upper right', ncol=3)
ff_ax.grid(False)
ff_ax.set_xlabel(ff_ax.get_xlabel() + r' (\si{\per\nano\second})')
ff_ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
ff_ax.tick_params(direction='out', which='both', labelsize=8)
ff_ax.set_ylim(bottom=1e-9, top=1e5)

omega = np.linspace(0, 1e2/T[key], n)
F = pulses_subspace[key].get_filter_function(omega)
ins_ax = inset_axes(ff_ax, 1, 1)
inset_position = InsetPosition(ff_ax, [0.1, 0.15, 0.5, 0.5])
ins_ax.set_axes_locator(inset_position)

for i in range(3):
    ins_ax.plot(omega, F[i, i].real, linewidth=1)

ins_ax.set_xlim(0, omega.max())
ins_ax.set_yscale('linear')
ins_ax.tick_params(direction='in', which='both', labelsize=6)
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

ff_ax.text(0.02, 0.85, '(b)', transform=ff_ax.transAxes, fontsize=10)

# Transfer matrices
Qmax = Q.max()
Qmin = -Qmax
if colorscale == 'log':
    linthresh = np.abs(Q).mean()/10
    norm = colors.SymLogNorm(linthresh=linthresh, vmin=Qmin, vmax=Qmax)
elif colorscale == 'linear':
    norm = colors.Normalize(vmin=Qmin, vmax=Qmax)

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
    im = tm_ax.imshow(Q[i][..., idx][idx, ...], **imshow_kw)
    tm_ax.text(0.01, 1.05, subfigs[i] + ' ' + n_oper_identifier,
               transform=tm_ax.transAxes, fontsize=10)
    # tm_ax.set_title(subfigs[i] + ' ' + n_oper_identifier)
    tm_ax.set_xticks(np.arange(16))
    tm_ax.set_yticks(np.arange(16))
    tm_ax.set_xticklabels(basis_labels[idx], rotation='vertical', fontsize=8)
    tm_ax.set_yticklabels(basis_labels[idx], fontsize=8)
    tm_ax.spines['left'].set_visible(False)
    tm_ax.spines['right'].set_visible(False)
    tm_ax.spines['top'].set_visible(False)
    tm_ax.spines['bottom'].set_visible(False)

gs.tight_layout(fig, h_pad=0., w_pad=0.4, pad=0.)

# cb_ax = fig.add_subplot(gs[1:, -1])
# cb_ax = fig.add_axes([1, 0.105, 0.015, 0.375])
cb_ax = fig.add_axes([1, 0.113, 0.015, 0.380])
# divider = make_axes_locatable(tm_axes)
# cb_ax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(im, ax=tm_axes, cax=cb_ax, fraction=0.045, pad=0.04)

aa = '-'.join(str(a).split('.'))
fname = f'all_in_one_alpha-{aa}_linear_subspace_{key}'
for ext in exts:
    fig.savefig(save_path / ext / '.'.join((fname, ext)))

# %% pulse train + ff side-by-side, transfer matrices extra
key = 'CNOT'
a = 0.7
Q = P[key][a][:3]
colorscale = 'linear'

fig = plt.figure(constrained_layout=True,
                 figsize=(figsize_wide[0], figsize_wide[1]/2))
gs = GridSpec(1, 5, figure=fig)

# Pulse train
pt_ax = fig.add_subplot(gs[0, :2])
*_, leg = ff.plot_pulse_train(
    pulses_subspace[key], c_oper_identifiers=identifiers[:3],
    axes=pt_ax, fig=fig,
)
pt_ax.set_xlabel(r'$t$ (\si{\nano\second})')
pt_ax.set_ylabel(r'$J(\epsilon_{ij})$ (\si{\per\nano\second})')
pt_ax.grid(False)
pt_ax.text(0.02, 0.9, '(a)', transform=pt_ax.transAxes, fontsize=10)
leg.remove()

# Filter functions
ff_ax = fig.add_subplot(gs[0, 2:])
omega = np.geomspace(1/T[key], 1e2, n)
*_, leg = ff.plot_filter_function(
    pulses_subspace[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False, axes=ff_ax, fig=fig,
)
# ax.plot(omega, 10**log_fitfun(output.beta, omega*T[key]), '--')
# ax.set_ylim(1e-15, ax.get_ylim()[1])
ff_ax.legend(identifiers[:3], frameon=False, loc='upper right')
ff_ax.grid(False)
ff_ax.set_xlabel(ff_ax.get_xlabel() + r' (\si{\per\nano\second})')
ff_ax.set_ylabel(r'$F_{\epsilon_{ij}}(\omega)$')
# ax.set_yticks([1e-8, 1e-5, 1e-2, 1e1])
ff_ax.tick_params(direction='out', which='both')
ff_ax.set_ylim(bottom=1e-9, top=1e3)

omega = np.linspace(0, 1e2/T[key], n)
F = pulses_subspace[key].get_filter_function(omega)
ins_ax = inset_axes(ff_ax, 1, 1)
inset_position = InsetPosition(ff_ax, [0.1, 0.15, 0.5, 0.5])
ins_ax.set_axes_locator(inset_position)

for i in range(3):
    ins_ax.plot(omega, F[i, i].real, linewidth=1)

ins_ax.set_xlim(0, omega.max())
ins_ax.set_yscale('linear')
ins_ax.tick_params(direction='in', which='both')
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

ff_ax.text(0.02, 0.9, '(b)', transform=ff_ax.transAxes, fontsize=10)

aa = '-'.join(str(a).split('.'))
fname = f'{key}-{aa}_pulse_train-filter_function'
for ext in exts:
    fig.savefig(save_path / ext / '.'.join((fname, ext)))

# %% Transfer matrices
# fig, grid = ff.plot_error_transfer_matrix(
#     P=P[key][a][:3, :16, :16], n_oper_identifiers=identifiers[:3],
#     figsize=figsize_wide, grid_kw=dict(axes_pad=0.1, cbar_pad=0.1),
#     colorscale='linear', basis_labels=basis_labels[:16], basis_labelsize=6,
#     imshow_kw=dict(origin='upper')
# )

fig, grid = plt.subplots(1, 3, figsize=figsize_wide, constrained_layout=False)
Qmax = Q.max()
Qmin = -Qmax
if colorscale == 'log':
    linthresh = np.abs(Q).mean()/10
    norm = colors.SymLogNorm(linthresh=linthresh, vmin=Qmin, vmax=Qmax)
elif colorscale == 'linear':
    norm = colors.Normalize(vmin=Qmin, vmax=Qmax)

imshow_kw = {}
imshow_kw.setdefault('origin', 'upper')
imshow_kw.setdefault('interpolation', 'nearest')
imshow_kw.setdefault('cmap', plt.get_cmap('RdBu'))
imshow_kw.setdefault('norm', norm)
basis_labels = np.array([
    ''.join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=2)
])
basis_labelsize = 6

subfigs = ['(c)', '(d)', '(e)']
for i, (n_oper_identifier, subfig) in enumerate(zip(identifiers,
                                                    ('(c)', '(d)', '(e)'))):
    if i == 2:
        idx = [np.ravel_multi_index((j, i), (4, 4))
               for i in range(4) for j in range(4)]
    else:
        idx = [np.ravel_multi_index((i, j), (4, 4))
               for i in range(4) for j in range(4)]

    tm_ax = grid[i]
    im = tm_ax.imshow(Q[i][..., idx][idx, ...], **imshow_kw)
    tm_ax.text(0.01, 1.05, subfig + ' ' + n_oper_identifier,
               transform=tm_ax.transAxes, fontsize=10)
    # tm_ax.set_title(subfigs[i] + ' ' + n_oper_identifier, fontsize='medium')
    tm_ax.set_xticks(np.arange(16))
    tm_ax.set_yticks(np.arange(16))
    tm_ax.set_xticklabels(basis_labels[idx], rotation='vertical',
                          fontsize=basis_labelsize)
    tm_ax.set_yticklabels(basis_labels[idx], fontsize=basis_labelsize)
    tm_ax.spines['left'].set_visible(False)
    tm_ax.spines['right'].set_visible(False)
    tm_ax.spines['top'].set_visible(False)
    tm_ax.spines['bottom'].set_visible(False)

fig.subplots_adjust(wspace=0.3)
gs.tight_layout(fig, h_pad=0.0)

# cb_ax = fig.add_subplot(gs[1:, -1])
# cb_ax = fig.add_axes([1, 0.105, 0.015, 0.375])
# cb_ax = fig.add_axes([1, 0.105, 0.015, 0.85])
# divider = make_axes_locatable(grid.ravel().tolist())
# cb_ax = divider.append_axes('right', size='5%', pad=0.05)
# cb = fig.colorbar(im, ax=grid.ravel().tolist(), cax=cb_ax, fraction=0.045, pad=0.04)
cb = fig.colorbar(im, ax=grid.flat, fraction=0.045, pad=0.02, shrink=0.425)

aa = '-'.join(str(a).split('.'))
fname = f'{key}-{aa}_error_transfer_matrix-subspace'
for ext in exts:
    fig.savefig(save_path / ext / '.'.join((fname, ext)))

# %% fidelities
# repo = git.Repo('Z:/Code/filter_functions')
# sha = repo.head.object.hexsha

# fname = 'fidelities_commit-{}.txt'.format(sha[:7])

# with open(save_path / fname, 'w+') as f:
#     infidelities = np.empty((2, 3, 6))
#     f.write(f'Git commit hash: {sha}\n\n')
#     f.write('calculated using ff.infidelity\n')
#     f.write('====================================================\n')
#     f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
#     for i, a in enumerate(alpha):
#         f.write('----------------------------------------------------\n')
#         for j, (key, pulse) in enumerate(pulses_complete.items()):
#             omega = np.geomspace(1/T[key], 1e2, n)
#             S = A[i]/omega**a
#             S, omega = ff.util.symmetrize_spectrum(S, omega)
#             infidelities[i, j] = ff.infidelity(pulse, S, omega)
#             f.write(f'{key}\t{a}\t\t{infidelities[i, j, :3].sum():.3e}\t\t\t' +
#                     f'{infidelities[i, j, :3].sum()*4/5:.3e}\n')
#     f.write('====================================================\n')
#     f.write('calculated using ff.error_transfer_matrix\n')
#     f.write('====================================================\n')
#     f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
#     for i, a in enumerate(alpha):
#         f.write('----------------------------------------------------\n')
#         for j, (key, pulse) in enumerate(pulses_complete.items()):
#             omega = np.geomspace(1/T[key], 1e2, n)
#             S = A[i]/omega**a
#             S, omega = ff.util.symmetrize_spectrum(S, omega)
#             infid = np.einsum('...ii', P[key][a][:3, :16, :16]).sum()/4**2
#             f.write(f'{key}\t{a}\t\t{infid:.3e}\t\t\t{infid*4/5:.3e}\n')
#     f.write('====================================================\n')
#     f.write('MC\n')
#     f.write('====================================================\n')
#     f.write('Gate\talpha\tF_ent (eps only)\tF_avg (eps only)\n')
#     for i, a in enumerate(alpha):
#         f.write('----------------------------------------------------\n')
#         for j, (key, pulse) in enumerate(pulses_complete.items()):
#             F_e = infid_fast[key][i]
#             f.write(f'{key}\t{a}\t\t{F_e:.3e}\t\t\t{F_e*4/5:.3e}\n')
#     f.write('====================================================')
#
# # %% leakage
# repo = git.Repo('Z:/Code/filter_functions')
# sha = repo.head.object.hexsha
#
# fname = 'leakage_commit-{}.txt'.format(sha[:7])
# pulse = pulses_complete['CNOT']
#
# d_c = 4
# d_l = 2
# PI_c_op = np.zeros((6, 6))
# PI_l_op = np.zeros((6, 6))
# PI_c_op[range(1, 5), range(1, 5)] = 1
# PI_l_op[(0, 5), (0, 5)] = 1
# PI_c_vec = ff.basis.expand(PI_c_op, pulse.basis)
# PI_l_vec = ff.basis.expand(PI_l_op, pulse.basis)
#
#
# def leakage(E):
#     return (PI_l_vec.T @ E @ PI_c_vec/d_c).real
#
#
# def seepage(E):
#     return (1 - PI_l_vec.T @ E @ PI_l_vec/d_l).real
#
#
# with open(save_path / fname, 'w+') as f:
#     infidelities = np.empty((2, 3, 6))
#     f.write(f'Git commit hash: {sha}\n\n')
#     f.write('====================================================\n')
#     f.write('                   CNOT (eps_23)\n')
#     f.write('====================================================\n')
#     f.write('----------------------------------------------------\n')
#     f.write('\t\t\tSystematic (L(Q))\n')
#     f.write('alpha\tLeakage\t\t\tSeepage\n')
#     f.write('----------------------------------------------------\n')
#     for i, a in enumerate(alpha):
#         Q_liouville = ff.numeric.liouville_representation(pulse.total_Q,
#                                                           pulse.basis)
#         Uerr_liouville = np.eye((d_c + d_l)**2) - P['CNOT'][a][1]
#         L_c = leakage(Q_liouville)
#         L_l = seepage(Q_liouville)
#         f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')
#
#     f.write('----------------------------------------------------\n')
#     f.write('\t\t\tDecoherence (L(Uerr))\n')
#     f.write('alpha\tLeakage\t\t\tSeepage\n')
#     f.write('----------------------------------------------------\n')
#     for i, a in enumerate(alpha):
#         Q_liouville = ff.numeric.liouville_representation(pulse.total_Q,
#                                                           pulse.basis)
#         Uerr_liouville = np.eye((d_c + d_l)**2) - P['CNOT'][a][1]
#         L_c = leakage(Uerr_liouville)
#         L_l = seepage(Uerr_liouville)
#         f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')
#
#     f.write('----------------------------------------------------\n')
#     f.write('\t\t\tBoth (L(Q @ Uerr))\n')
#     f.write('alpha\tLeakage\t\t\tSeepage\n')
#     f.write('----------------------------------------------------\n')
#     for i, a in enumerate(alpha):
#         Q_liouville = ff.numeric.liouville_representation(pulse.total_Q,
#                                                           pulse.basis)
#         Uerr_liouville = np.eye((d_c + d_l)**2) - P['CNOT'][a][1]
#         L_c = leakage(Q_liouville @ Uerr_liouville)
#         L_l = seepage(Q_liouville @ Uerr_liouville)
#         f.write(f'{a}\t\t{L_c:.3e}\t\t{L_l:.3e}\n')
#     f.write('====================================================')
