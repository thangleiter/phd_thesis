# %% Imports
import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from mjolnir.helpers import save_to_hdf5
from mjolnir.plotting import plot_nd  # noqa
from qcodes.dataset import initialise_or_create_database_at, load_by_run_spec
from qutil import const
from qutil.plotting.colors import (RWTH_COLORS, RWTH_COLORS_50, RWTH_COLORS_75,
                                   make_sequential_colormap)
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, TOTALWIDTH, PATH, init,  # noqa
                    secondary_axis, apply_sketch_style, E_AlGaAs, effective_mass, sliceprops)
from experiment.plotting import browse_db  # noqa

EXTRACT_DATA = os.environ.get('EXTRACT_DATA', False)
FILE_PATH = pathlib.Path(__file__).relative_to(pathlib.Path(__file__).parents[3])
DATA_PATH = PATH.parent / 'data/pl'
DATA_PATH.mkdir(exist_ok=True)
ORIG_DATA_PATH = pathlib.Path(r"\\janeway\User AG Bluhm\Common\GaAs\PL Lab\Data\Triton\db")
SAVE_PATH = PATH / 'pdf/experiment'
SAVE_PATH.mkdir(exist_ok=True)
with np.errstate(divide='ignore', invalid='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('magenta', endpoint='blackwhite').reversed()

LINE_COLORS = [color for name, color in RWTH_COLORS.items() if name not in ('magenta',)]

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def extract_data(ds, V_DM, E0):
    da = ds.ccd_ccd_data_bg_corrected_per_second.sel(
        doped_M1_05_49_2_trap_2_central_difference_mode=V_DM,
        method='nearest'
    )
    da_pl = da.sel(excitation_path_wavelength_constant_power=795, method='nearest')
    da_pl = da_pl[~da_pl.isnull()]

    da_ple = da[
        da.excitation_path_wavelength_constant_power < const.eV2lambda(E0)*1e9,
        da.ccd_horizontal_axis < E0
    ].integrate('ccd_horizontal_axis')
    da_ple = da_ple[~da_ple.isnull()]

    return da_pl, da_ple


def analyze_ple(ds):
    da = ds.ccd_ccd_data_bg_corrected_per_second.where(
        ds.ccd_horizontal_axis + 1e-3
        < const.h*const.c/(1e-9*ds.excitation_path_wavelength_constant_power*const.e)
    )
    integral = da[..., 1:].values + da[..., :-1].values
    integral *= np.diff(da.ccd_horizontal_axis)
    integral = np.nansum(integral, axis=-1) / 2
    integral[integral == 0] = np.nan

    return xr.DataArray(
        integral,
        coords={'excitation_path_wavelength_constant_power':
                ds.excitation_path_wavelength_constant_power,
                'doped_M1_05_49_2_trap_2_central_difference_mode':
                ds.doped_M1_05_49_2_trap_2_central_difference_mode},
        name='ple_power',
        attrs={'units': 'eV/s'}
    )


def annotate_shift(ax, E_1, E_1p, hs=(325, 450), xytext=None):
    arrowprops = dict(arrowstyle='<->', mutation_scale=7.5, color=RWTH_COLORS_75['black'],
                      linewidth=0.75, shrinkA=0, shrinkB=0, alpha=0.66)

    match backend:
        case 'pgf':
            s = r'$\Delta E_{\mathrm{S}} = '
            s += rf'\qty{{{(E_1p[0] - E_1[0]) * 1e3:.1f}}}{{\milli\electronvolt}}$'
        case _:
            s = r'$\Delta E_{{\mathrm{{S}}}} = $' + f'{(E_1p[0] - E_1[0]) * 1e3:.1f} meV'

    for Es, ls, h in zip(zip(E_1, E_1p), [(0, (2, 2)), ':'], hs):
        ax.axvline(Es[0], ls=ls, color=RWTH_COLORS_75['black'], alpha=0.66)
        ax.axvline(Es[1], ls=ls, color=RWTH_COLORS_75['black'], alpha=0.66)
        ax.annotate('', (Es[0], h), (Es[1], h), arrowprops=arrowprops)

    ax.annotate(s, (np.sum([E_1, E_1p]) / (2*len(E_1)), np.average(hs)), xytext=xytext,
                verticalalignment='center', horizontalalignment='center', fontsize='small')


def fit_esser(da, p0):
    def exciton(E, I_X, E_X, Γ_X):
        return I_X / np.cosh((E - E_X)/Γ_X)

    def trion(E, I_T, E_T, T, c_1=15, eps_1=1.1e-3*const.e):
        M_X = m[0, 0] + m[1, 0]
        eps = E_T - E
        MQ = c_1*np.exp(eps/eps_1)
        return I_T*MQ*np.exp(-eps*m[0, 0]/(const.k*T*M_X))*E[E >= eps]

    return np.convolve(exciton(), trion(), mode='same')


# %% Load data
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH / 'membrane_doped_M1_05_49-2.db')

    save_to_hdf5(60, DATA_PATH / 'doped_M1_05_49-2_ple.h5',
                 parameters='ccd_ccd_data_bg_corrected_per_second')

ds = load_by_run_spec(captured_run_id=60).to_xarray_dataset('ccd_ccd_data_bg_corrected_per_second')
# %% Plot
m = effective_mass()
# %%% Line traces
# browse_db(60, max=500, vertical_target='wavelength')
# browse_db(60, vmax=350, horizontal_target='path_wavelength', vertical_target='difference_mode')

da_ple_full = analyze_ple(ds)
V_DM = [2.65, 1.678, 0.6, -1.57, -2.65]

fig, axs_pl = plt.subplots(nrows=len(V_DM), sharex=True, sharey=True, figsize=(TEXTWIDTH, 3.75))
axs_ple = []
ax2, unit = secondary_axis(axs_pl[0])

for i, (v, color, ax_pl) in enumerate(zip(V_DM, RWTH_COLORS, axs_pl)):
    da_pl = ds.ccd_ccd_data_bg_corrected_per_second.sel(
        excitation_path_wavelength_constant_power=795,
        doped_M1_05_49_2_trap_2_central_difference_mode=v,
        method='nearest'
    )
    da_ple = da_ple_full.sel(doped_M1_05_49_2_trap_2_central_difference_mode=v, method='nearest')

    ax_pl.plot(da_pl.ccd_horizontal_axis, da_pl, color=RWTH_COLORS[color])
    axs_ple.append(ax_ple := ax_pl.twinx())
    ax_ple.plot(const.lambda2eV(da_ple.excitation_path_wavelength_constant_power*1e-9), da_ple,
                color=RWTH_COLORS_75[color], ls='--')

    ax_pl.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax_pl.set_ylim(0)
    ax_pl.set_yticks([0, 150, 300])
    ax_ple.set_ylim(0)

    V = da_pl.doped_M1_05_49_2_trap_2_central_difference_mode.item()
    match backend:
        case 'pgf':
            s = rf'$V_{{\mathrm{{DM}}}} = \qty{{{V:.2f}}}{{\volt}}$'
        case _:
            s = rf'$V_{{\mathrm{{DM}}}} = {V:.2f}$ V'
    if i < len(V_DM) - 1:
        ax_pl.tick_params(which='both', bottom=False, labelbottom=False)
    else:
        s = s.replace('=', '=$\n$\\quad')
    if i > 0:
        axs_ple[i].sharey(axs_ple[i-1])

    ax_pl.text(0.015, 0.875, s, transform=ax_pl.transAxes,
               verticalalignment='top', horizontalalignment='left', fontsize='small')

E_1 = 1.5112
E_1p = 1.5383
annotate_shift(axs_pl[0], [E_1], [E_1p], (350, 475))

E_1 = 1.5192
E_1p = 1.5479
annotate_shift(axs_pl[1], [E_1], [E_1p])

E_1 = 1.5180
E_1p = 1.5457
annotate_shift(axs_pl[2], [E_1], [E_1p])

E_1 = 1.511
E_1p = 1.5391
annotate_shift(axs_pl[3], [E_1], [E_1p])

E_1 = 1.4863
E_2 = 1.4900
E_1p = 1.5269
E_2p = 1.5306
annotate_shift(axs_pl[4], [E_1, E_2], [E_1p, E_2p], (300, 425))

ax_pl.set_xlim(1.465, const.lambda2eV(da_ple.excitation_path_wavelength_constant_power[0]*1e-9))
ax_pl.set_xlabel('$E$ (eV)')
ax2.set_xlabel(rf'$\lambda$ ({unit})')
fig.supylabel('PL count rate (cps)', fontsize='medium')
fig.text(0.98, 0.5, 'PLE power (eV/s)', horizontalalignment='right', verticalalignment='center',
         rotation=90)

fig.tight_layout(h_pad=0)
fig.subplots_adjust(hspace=0, left=0.15, right=0.86)

fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple_cuts.pdf')

# %%% Full map
da_pl = ds.ccd_ccd_data_bg_corrected_per_second.sel(
    excitation_path_wavelength_constant_power=795)

fig, axs = plt.subplots(ncols=4, layout='constrained', figsize=(TEXTWIDTH, 2),
                        gridspec_kw=dict(width_ratios=[15, 1, 15, 1]))

ax = axs[0]
img = ax.pcolormesh(da_pl.ccd_horizontal_axis,
                    da_pl.doped_M1_05_49_2_trap_2_central_difference_mode,
                    da_pl * 1e-3,
                    cmap=SEQUENTIAL_CMAP, vmin=0, rasterized=True)

cb = fig.colorbar(img, cax=axs[1], label='PL count rate (kcps)')
ax.set_xlim(const.lambda2eV(np.array([840, 809])*1e-9))
ax.set_xlabel(r'$E_{\mathrm{det}}$ (eV)')
ax.set_ylabel(r'$V_{\mathrm{DM}}$ (V)')
ax2, unit = secondary_axis(ax, 'eV')
ax2.set_xlabel(rf'$\lambda_{{\mathrm{{det}}}}$ ({unit})')

ax = axs[2]
img = ax.pcolormesh(const.lambda2eV(da_ple_full.excitation_path_wavelength_constant_power*1e-9),
                    da_ple_full.doped_M1_05_49_2_trap_2_central_difference_mode,
                    da_ple_full.T,
                    cmap=SEQUENTIAL_CMAP, vmin=0, rasterized=True)

cb = fig.colorbar(img, cax=axs[3], label='PLE power (eV/s)')
ax.set_xlabel(r'$E_{\mathrm{exc}}$ (eV)')
ax2, unit = secondary_axis(ax, 'eV')
ax2.set_xlabel(rf'$\lambda_{{\mathrm{{exc}}}}$ ({unit})')

for v in V_DM:
    axs[0].axhline(v, **sliceprops(RWTH_COLORS_50['black'], linewidth=0.5, alpha=0.5))
    axs[2].axhline(v, **sliceprops(RWTH_COLORS_50['black'], linewidth=0.5, alpha=0.5))

fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple_wide.pdf')

# %%%% margin size

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.7))

    img = ax.pcolormesh(
        const.lambda2eV(da_ple_full.excitation_path_wavelength_constant_power*1e-9),
        da_ple_full.doped_M1_05_49_2_trap_2_central_difference_mode,
        da_ple_full.T,
        cmap=SEQUENTIAL_CMAP, vmin=0, rasterized=True
    )

    cb = fig.colorbar(img, label='PLE power (eV/s)')
    ax.set_xticks([1.525, 1.550])
    ax.set_xlabel(r'$E_{\mathrm{exc}}$ (eV)')
    ax.set_ylabel(r'$V_{\mathrm{DM}}$ (V)')
    ax2, unit = secondary_axis(ax, 'eV')
    ax2.set_xlabel(rf'$\lambda_{{\mathrm{{exc}}}}$ ({unit})')

    for v in V_DM:
        ax.axhline(v, **sliceprops(RWTH_COLORS_50['black'], linewidth=0.5, alpha=0.5))

    fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple.pdf')

# %%%% single wavelength
da = ds.ccd_ccd_data_bg_corrected_per_second.sel(ccd_horizontal_axis=[1.51342, 1.51897, 1.52275],
                                                 method='nearest')

fig = plt.figure(figsize=(TEXTWIDTH, 1.6))
grid = ImageGrid(fig, 111, (1, da.shape[-1]), cbar_mode='single', aspect=False,
                 axes_pad=0.04, cbar_pad=0.05)

norm = mpl.colors.Normalize(vmin=0, vmax=da.max().item())
for i in range(da.shape[-1]):
    img = grid[i].pcolormesh(const.lambda2eV(da.excitation_path_wavelength_constant_power*1e-9),
                             da.doped_M1_05_49_2_trap_2_central_difference_mode,
                             da[..., i].T,
                             norm=norm, cmap=SEQUENTIAL_CMAP, rasterized=True)

    E = da.ccd_horizontal_axis[i].item()
    match backend:
        case 'pgf':
            s = rf'$E_{{\mathrm{{det}}}} = \qty{{{E:.3f}}}{{\electronvolt}}$'
        case _:
            s = rf'$E_{{\mathrm{{det}}}} = {E:.3f}$ eV'
    grid[i].annotate(s, (1.557, -2.4), horizontalalignment='right', fontsize='small')
    ax2, unit = secondary_axis(grid[i], 'eV')
    if i == da.shape[-1] // 2:
        ax2.set_xlabel(rf'$\lambda_{{\mathrm{{exc}}}}$ ({unit})')
        grid[i].set_xlabel(r'$E_{\mathrm{exc}}$ (eV)')

cb = grid.cbar_axes[0].colorbar(img, label='PLE count rate (cps)')
grid[0].set_ylabel(r'$V_{\mathrm{DM}}$ (V)')

fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple_delta.pdf')

# %%% Only at large shift (unused)
da_pl, da_ple = extract_data(ds, -2.65, 1.5)

fig, ax_pl = plt.subplots(layout='constrained', figsize=(TEXTWIDTH, 2))
ax_pl.plot(da_pl.ccd_horizontal_axis, da_pl, color=RWTH_COLORS['blue'])
ax_pl.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax_pl.set_ylim(0)

ax2, unit = secondary_axis(ax_pl)

ax_ple = ax_pl.twinx()
ax_ple.plot(const.lambda2eV(da_ple.excitation_path_wavelength_constant_power*1e-9), da_ple,
            color=RWTH_COLORS_75['blue'], ls='--')
ax_ple.set_ylim(0)

E_1 = 1.4863
E_2 = 1.4900
E_1p = 1.5269
E_2p = 1.5306
arrowprops = dict(arrowstyle='<->', mutation_scale=7.5, color=RWTH_COLORS_50['black'],
                  linewidth=0.75, shrinkA=0, shrinkB=0)
annotate_kwargs = dict(color=RWTH_COLORS_75['black'], arrowprops=arrowprops)

ax_pl.axvline(E_1, ls=':', color=RWTH_COLORS['magenta'], alpha=0.66)
ax_pl.axvline(E_2, ls=':', color=RWTH_COLORS['green'], alpha=0.66)
ax_pl.axvline(E_1p, ls=':', color=RWTH_COLORS['magenta'], alpha=0.66)
ax_pl.axvline(E_2p, ls=':', color=RWTH_COLORS['green'], alpha=0.66)

ax_pl.annotate('', (E_1, 275), (E_1p, 275),
               arrowprops=arrowprops | dict(color=RWTH_COLORS['magenta']))
ax_pl.annotate('', (E_2, 200), (E_2p, 200),
               arrowprops=arrowprops | dict(color=RWTH_COLORS['green']))

match backend:
    case 'pgf':
        ax_pl.annotate(rf'$\Delta E = \qty{{{(E_2p-E_2)*1e3:.1f}}}{{\milli\electronvolt}}$',
                       ((E_1 + E_2 + E_1p + E_2p)/4, (200 + 275)/2), verticalalignment='center',
                       horizontalalignment='center')
    case _:
        ax_pl.annotate(r'$\Delta E = $' + f'{(E_2p-E_2)*1e3:.1f} meV',
                       ((E_1 + E_2 + E_1p + E_2p)/4, (200 + 275)/2), verticalalignment='center',
                       horizontalalignment='center')

ax2.set_xlabel(rf'$\lambda$ ({unit})')
ax_pl.set_xlabel('$E$ (eV)')
ax_pl.set_ylabel('PL count rate (cps)')
ax_ple.set_ylabel('PLE power (eV/s)')

fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple_single.pdf')
