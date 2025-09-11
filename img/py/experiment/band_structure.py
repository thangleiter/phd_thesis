import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from poisson_schroedinger import PoissonSchroedingerSolver

from qutil import ui
# %% Functions


def V_TG(V_DM, V_CM):
    return 0.5*(V_CM + V_DM)


def V_BG(V_DM, V_CM):
    return 0.5*(V_CM - V_DM)


def setup_structure(doping=6.5e17, cap=10, qw=20, qr_ix=(1, 2, 3, 4, 5, 6, 7), qw_ix=(2, 6)):
    structure = PoissonSchroedingerSolver()

    band_offset = 0.24
    structure.add_layer(cap)
    structure.add_layer(40, band_offset_eV=band_offset, doping_concentration_per_cm=doping)
    structure.add_layer(40, band_offset_eV=band_offset)
    structure.add_layer(10, band_offset_eV=band_offset)
    structure.add_layer(qw)
    structure.add_layer(10, band_offset_eV=band_offset)
    structure.add_layer(40, band_offset_eV=band_offset)
    structure.add_layer(40, band_offset_eV=band_offset, doping_concentration_per_cm=doping)
    structure.add_layer(cap)

    structure.setup(quantum_regions=qr_ix, doping_trap_energy=-(0.33 - 0.22)*0.65,
                    temperature_K=10e-3, z_stepsize_nm=0.5, effective_mass_m0=0.067,
                    number_eigenvectors=2)

    z_qw = structure.z_interface_position_nm[slice(*qw_ix)]
    z_qw_ix = np.concat(np.where(structure.z_nm == z_qw[0]) + np.where(structure.z_nm == z_qw[-1]))

    return structure, z_qw, z_qw_ix


def cooldown_structure(structure, V_FP=0.76):
    structure.set_bias_voltages(V_FP, V_FP)
    structure.set_temperature(300)
    structure.solve(True, False, verbose=False, convergence_tolerance_eV=1e-4)

    structure.set_temperature(100)
    structure.solve(True, False, verbose=False)


def apply_voltage(structure, V_T, V_B, V_FP=0.76):
    structure.set_bias_voltages(V_FP - V_T, V_FP - V_B)
    structure.set_temperature(np.sqrt(100*10e-3))
    structure.solve(False, True, convergence_tolerance_eV=1e-2, verbose=False)
    structure.set_temperature(10e-3)
    structure.solve(False, True, verbose=False)


def extract_params(structure, z_qw_ix):
    E = structure.eigen_energies_eV
    V = structure.electric_potential_eV + structure.band_potential_eV
    psi = structure.psi
    n_tot = np.trapezoid(structure.carrier_density_per_nm*1e21,
                         dx=structure.z_stepsize_nm*1e-7)
    n_2DEG = np.trapezoid(structure.carrier_density_per_nm[slice(*z_qw_ix)]*1e21,
                          dx=structure.z_stepsize_nm*1e-7)
    return E, V, psi, n_tot, n_2DEG


# %%
if __name__ == '__main__':
    # don't run this script in batched mode. there's nothing to plot
    sys.exit(0)

# %% Simulate single-gate bias
V_FP = 0.76
structure, z_qw, z_qw_ix = setup_structure(doping=1.8e18, cap=10, qw=20+5.65e-1, qw_ix=(2, 6))
cooldown_structure(structure, V_FP=V_FP)

npts = 51
E = np.empty((npts, structure.number_eigenvectors))
V = np.empty((npts, structure.total_points))
psi = np.empty((npts, structure.quantum_region_width_nm, structure.number_eigenvectors))
n_2DEG = np.empty((npts,))
n_tot = np.empty((npts,))

V_CM = -0.5
for i, v in enumerate(ui.progressbar(voltage := np.linspace(0, -1.5, npts))):
    apply_voltage(structure, V_TG(v, V_CM), V_BG(v, V_CM))
    E[i], V[i], psi[i], n_tot[i], n_2DEG[i] = extract_params(structure, z_qw_ix)

    # if i % 10:
    #     structure.print_results()
    #     structure.plot_density()
# %%%%
fig, ax = plt.subplots(2, sharex=True, layout='constrained')
ax[0].plot(voltage, n_tot)
ax2 = ax[0].twinx()
ax2.plot(voltage, n_2DEG, color='tab:orange')
ax[1].plot(voltage, E*1e3)

ax[0].set_ylabel(r'$n_{\mathrm{tot}}$ (/cm²)')
ax2.set_ylabel(r'$n_{\mathrm{2DEG}}$ (/cm²)')
ax[1].set_ylabel('$E$ (meV)')
ax[-1].set_xlabel('$V$ (V)')

# %%%%
fig, ax = plt.subplots(layout='constrained')
img = ax.pcolormesh(voltage, structure.z_nm, V.T, norm=mpl.colors.TwoSlopeNorm(0), cmap='RdBu')
cb = fig.colorbar(img, label='$V$ (eV)', ticks=[-0.06, -0.04, -0.02, 0, 0.5, 1.0, 1.5])
ax.set_ylabel('$z$ (nm)')
ax.set_xlabel('$V$ (V)')

# %%%%
fig, axs = plt.subplots(structure.number_eigenvectors, sharex=True, sharey=True,
                        layout='constrained')
for i, ax in enumerate(axs):
    ax.pcolormesh(z := structure.z_nm[slice(*z_qw_ix)], V, abs(psi[:, slice(*z_qw_ix - 10), i]))
    ax.plot(z.mean() + np.trapezoid((z-z.mean()) * abs(psi[:, slice(*z_qw_ix - 10), i]) ** 2, z), V, 'k')

# %%%
npts = 101
E = np.empty((npts, npts, structure.number_eigenvectors))
psi = np.empty((npts, npts, structure.quantum_region_width_nm, structure.number_eigenvectors))
n_2DEG = np.empty((npts, npts))
n_tot = np.empty((npts, npts))

for i, vt in enumerate(ui.progressbar(V_T := np.linspace(-1, 1, npts))):
    for j, vb in enumerate(ui.progressbar(V_B := np.linspace(-1, 1, npts))):
        structure.set_bias_voltages(V_FP - vt, V_FP - vb)
        structure.set_temperature(np.sqrt(100*10e-3))
        structure.solve(False, True, convergence_tolerance_eV=1e-2, verbose=False)

        structure.set_temperature(10e-3)
        structure.solve(False, True, verbose=False)

        E[i, j] = structure.eigen_energies_eV
        psi[i, j] = structure.psi
        n_tot[i, j] = np.trapezoid(structure.carrier_density_per_nm*1e21,
                                   dx=structure.z_stepsize_nm*1e-7)
        n_2DEG[i, j] = np.trapezoid(structure.carrier_density_per_nm[slice(*z_qw_ix)]*1e21,
                                    dx=structure.z_stepsize_nm*1e-7)
