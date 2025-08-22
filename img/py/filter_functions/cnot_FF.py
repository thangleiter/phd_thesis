# %% Imports
import pathlib
import sys
from itertools import product
from unittest import mock

import filter_functions as ff
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter_functions import plotting, util
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil import misc
from qutil.plotting.colors import RWTH_COLORS, make_diverging_colormap
from scipy import constants, io

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import MAINSTYLE, MARGINSTYLE, TOTALWIDTH, MARGINWIDTH, PATH, init

with np.errstate(divide='ignore', invalid='ignore'):
    DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'), endpoint='white')

LINE_COLORS = list(RWTH_COLORS.values())[1:]
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/filter_functions'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/filter_functions'
SAVE_PATH.mkdir(exist_ok=True)

init(MAINSTYLE, backend := 'pgf')

pernanosecond = r' (\unit{\per\nano\second})' if backend == 'pgf' else r' (ns$^{-1}$)'
# %% set up the operators
gates = ['X2ID', 'Y2ID', 'CNOT']
struct = {'X2ID': io.loadmat(str(DATA_PATH / 'X2ID.mat')),
          'Y2ID': io.loadmat(str(DATA_PATH / 'Y2ID.mat')),
          'CNOT': io.loadmat(str(DATA_PATH / 'CNOT.mat'))}
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

# %% constants
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
omega = {key: np.geomspace(1e-2/val, 1e2, n) for key, val in T.items()}
S = {key: {alpha[i]: A[i]/val**alpha[i] for i in range(len(alpha))} for key, val in omega.items()}

basis_labels = ['']
basis_labels.extend([''.join(tup) for tup in
                     product(['I', 'X', 'Y', 'Z'], repeat=2)][1:])
basis_labels.extend(['$C_{{{}}}$'.format(i) for i in range(16, 36)])
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


# %% Big plot (main text)
key = 'CNOT'
a = 0.7
K = ff.numeric.calculate_cumulant_function(pulses_complete[key], S[key][a], omega[key],
                                           n_oper_identifiers=identifiers[:3],
                                           show_progressbar=True)

fig = plt.figure(figsize=(TOTALWIDTH, 4.25))
gs = GridSpec(5, 9, figure=fig)

pt_ax = fig.add_subplot(gs[:2, :4])
ff_ax = fig.add_subplot(gs[:2, 4:9])

ins_ax = ff_ax.inset_axes([0.075, 0.15, 0.55, 0.5])
grid = ImageGrid(fig, gs[2:, 0:9], (1, 3), axes_pad=0.1, cbar_mode=None, label_mode='L')

# Pulse train
*_, leg = plotting.plot_pulse_train(
    pulses_complete[key], c_oper_identifiers=identifiers[:3],
    axes=pt_ax, fig=fig,
)
leg.remove()
pt_ax.set_xlabel(r'$t$ (ns)')
pt_ax.set_ylabel(r'$J(\epsilon_{ij}(t))$' + pernanosecond)
pt_ax.grid(False)
pt_ax.text(0.015, 0.875, '(a)', transform=pt_ax.transAxes)

pt_ax.legend(identifiers[:3], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
             ncols=3, mode="expand", borderaxespad=0., frameon=False)

# Filter functions
omega = np.geomspace(1/T[key], 1e2, n)

cache_idx = np.diag([0, *[1]*15, *[0]*20])
cache_filter_function(pulses_complete[key], omega, cache_idx)

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    yscale='log', omega_in_units_of_tau=False,
    axes=ff_ax, fig=fig,
)
leg.remove()
# ff_ax.legend(identifiers[:3], frameon=True, loc='upper right')
ff_ax.grid(False)
ff_ax.yaxis.tick_right()
ff_ax.yaxis.set_label_position('right')
ff_ax.set_xlabel(ff_ax.get_xlabel() + pernanosecond)
ff_ax.set_ylabel(r'$\mathcal{F}_{\epsilon_{ij}}(\omega)$')
ff_ax.tick_params(direction='out', which='both')
ff_ax.set_ylim(bottom=1e-9, top=1e3)

# Inset
omega = np.linspace(0, 1e2/T[key], n)
cache_filter_function(pulses_complete[key], omega, cache_idx)

*_, leg = plotting.plot_filter_function(
    pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
    xscale='linear', yscale='linear', omega_in_units_of_tau=False,
    axes=ins_ax, fig=fig
)
leg.remove()

ins_ax.set_xlabel(None)
ins_ax.set_ylabel(None)
ins_ax.grid(False)
ins_ax.tick_params(direction='in', which='both', labelsize='x-small')
ins_ax.spines['right'].set_visible(False)
ins_ax.spines['top'].set_visible(False)
ins_ax.patch.set_alpha(0)

ff_ax.text(0.015, 0.875, '(b)', transform=ff_ax.transAxes)

# Transfer matrices
norm = mpl.colors.CenteredNorm(0)
basis_labels = np.array([
    ''.join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=2)
])
subfigs = ['(c)', '(d)', '(e)']

for i, (n_oper_identifier, tm_ax) in enumerate(zip(identifiers, grid)):
    idx = np.arange(16).reshape(4, 4).ravel(order='C' if i != 2 else 'F')
    im = tm_ax.imshow(K[i, *np.ix_(idx, idx)], interpolation=None, norm=norm, cmap=DIVERGING_CMAP)
    tm_ax.text(0.01, 1.05, subfigs[i] + ' ' + n_oper_identifier,
               transform=tm_ax.transAxes)
    tm_ax.set_xticks(np.arange(16), basis_labels[idx], rotation='vertical', fontsize='small')
    tm_ax.set_yticks(np.arange(16), basis_labels[idx], fontsize='small')

with misc.filter_warnings(action='ignore', category=UserWarning):
    gs.tight_layout(fig, h_pad=0., w_pad=0.5)

# Only add after applying the tigh layout!
cb_ax = fig.add_axes([0.9, 0.0958, 0.015, 0.3818])
cb = fig.colorbar(im, cax=cb_ax, label=r"Cumulant function $\mathcal{K}_{\epsilon_{ij}}(\tau)$")

fig.savefig(SAVE_PATH / 'CNOT_FF.pdf')
# %% Plot unitary vs complete (appendix)
key = 'CNOT'

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, layout='constrained',
                             figsize=(MARGINWIDTH, 2))
    omega = np.geomspace(1/T[key], 4, n)

    for i, (lab, ax) in enumerate(zip(['(a)', '(b)'], axes)):

        # Excluding identity element
        cache_filter_function(pulses_complete[key], omega,
                              np.diag([i, *[1]*15, *[0]*20]))

        *_, leg = plotting.plot_filter_function(
            pulses_complete[key], omega, n_oper_identifiers=identifiers[:3],
            yscale='log', omega_in_units_of_tau=False, axes=ax, fig=fig,
        )
        leg.remove()
        ax.grid(True)
        ax.set_xlabel(ax.get_xlabel() + pernanosecond)
        ax.set_ylabel(r'$\mathcal{F}_{\epsilon_{ij}}(\omega)$')
        ax.label_outer()
        ax.tick_params(direction='out', which='both')
        ax.text(0.04, 0.1, lab, transform=ax.transAxes)

fig.savefig(SAVE_PATH / 'CNOT_FF_unitary_v_complete.pdf')

# %% Fidelities table
infidelities_ff_etm = np.empty((2, 3, 3))
infidelities_ff = np.empty((2, 3, 3))
infidelities_mc = np.empty((2, 3))

for i, a in enumerate(alpha):
    for j, (key, pulse) in enumerate(pulses_subspace.items()):
        omega = np.geomspace(1e-2/T[key], 1e2, n)
        S = A[i]/omega**a
        with mock.patch.object(pulse, 'd', 4):
            infidelities_ff[i, j] = ff.infidelity(pulse, S, omega, identifiers[:3])

        # Just for comparison
        for k in range(3):
            infidelities_ff_etm[i, j, k] = 1 - ff.error_transfer_matrix(
                pulse, S, omega, identifiers[k]
            ).trace()/d_c**2

for i, a in enumerate(alpha):
    for j, (key, pulse) in enumerate(pulses_complete.items()):
        F_e = infid_fast[key][i]
        infidelities_mc[i, j] = infid_fast[key][i]

data = {
    (r"\textsc{This work}", r"\sisetup{round-precision=1} 0"):
        infidelities_ff[alpha.tolist().index(0.0)].sum(-1),
    (r"\textsc{This work}", r"\sisetup{round-precision=1} 0.7"):
        infidelities_ff[alpha.tolist().index(0.7)].sum(-1),
    (r"\textsc{\citer{Cerfontaine2020b}}", r"\sisetup{round-precision=1} 0"):
        infidelities_mc[alpha.tolist().index(0.0)],
    (r"\textsc{\citer{Cerfontaine2020b}}", r"\sisetup{round-precision=1} 0.7"):
        infidelities_mc[alpha.tolist().index(0.7)],
}

index = [r"\XID2", r"\YID2", r"\CNOT"]
# entanglement -> average gate fidelity: *4/5
rows = [[label] + [col[i] * 4/5 for col in data.values()] for i, label in enumerate(index)]
cols = pd.MultiIndex.from_tuples([("", "$a$")] + list(data.keys()))
df = pd.DataFrame(rows, columns=cols)

with (DATA_PATH / 'gate_fidelities.tex').open('w') as file:
    latex = df.to_latex(
        column_format=(
            'l *{4}{S[table-number-alignment=center,table-text-alignment=left,'
            'table-format=+1.1e+3,round-mode=figures,round-precision=2]}'
        ),
        escape=False, index=False, multicolumn=True, multicolumn_format="c", float_format="%.6e"
    ).replace(
        '$a$', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}' + '\n$a$'
    ).replace(
        r'\begin{tabular}', r'\begin{tabularx}{\textwidth}'
    ).replace(
        r'\end{tabular}', r'\end{tabularx}'
    )

    file.write(f'% This table is automatically generated by {FILE_PATH} \n '.replace('\\', '/'))
    file.write(latex)
