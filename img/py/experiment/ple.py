# %% Imports
import os
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mjolnir.helpers import save_to_hdf5
from mjolnir.plotting import plot_nd  # noqa
from qcodes.dataset import initialise_or_create_database_at, load_by_run_spec
from qutil import const, itertools
from qutil.plotting.colors import (RWTH_COLORS, RWTH_COLORS_50, RWTH_COLORS_75,
                                   make_sequential_colormap)

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
PAD = 2

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def extract_data(V_DM, E0):
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


# %%
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH / 'membrane_doped_M1_05_49-2.db')

    save_to_hdf5(60, DATA_PATH / 'doped_M1_05_49-2_ple.h5',
                 parameters='ccd_ccd_data_bg_corrected_per_second')

ds = load_by_run_spec(captured_run_id=60).to_xarray_dataset('ccd_ccd_data_bg_corrected_per_second')

# %% PLE
# browse_db(60, max=500, vertical_target='wavelength')

V_DM = [2.65, 1.57, 0, -1.57, -2.65]
E0 = [1.516, 1.5266, 1.528, 1.5165, 1.5]

fig, axs_pl = plt.subplots(nrows=len(V_DM), sharex=True, sharey=True, figsize=(TEXTWIDTH, 3.5))
axs_ple = []
ax2, unit = secondary_axis(axs_pl[0])

for i, ((da_pl, da_ple), color, ax_pl) in enumerate(zip(
        itertools.starmap(extract_data, zip(V_DM, E0)),
        RWTH_COLORS,
        axs_pl
)):

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
    if i < len(E0) - 1:
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

E_1 = 1.5209
E_1p = 1.5479
annotate_shift(axs_pl[1], [E_1], [E_1p])

E_1 = 1.5194
E_1p = 1.5461
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

fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_ple.pdf')

# %% Only at large shift (unused)
da_pl, da_ple = extract_data(-2.65, 1.5)

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
