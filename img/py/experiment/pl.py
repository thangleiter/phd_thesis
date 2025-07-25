# %% Imports
import os
import json
import pathlib
import sys
from typing import Literal
from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import xarray as xr
from mjolnir.helpers import save_to_hdf5
from mjolnir.plotting import plot_nd  # noqa
from mpl_toolkits.axes_grid1 import ImageGrid
from qcodes.dataset import initialise_or_create_database_at
from qutil import const, itertools
from qutil.plotting import reformat_axis
from qutil.plotting.colors import (make_sequential_colormap,
                                   RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_50, RWTH_COLORS_25)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, TOTALWIDTH, PATH, init,  # noqa
                    secondary_axis, apply_sketch_style, E_AlGaAs, effective_mass, sliceprops)
from experiment.plotting import browse_db, print_params  # noqa

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


def generate_mosaic(fig, nrows_ncols: tuple[int, int], slc: bool = False,
                    cb: Literal[False, 'each', 'single', 'row'] = 'each',
                    slc_height_ratio: float = 1/4, cb_width_ratio: float = 1/20,
                    sharex: Literal['col'] | bool = False, sharey: Literal['row'] | bool = False,
                    **kwargs):
    nr, nc = nrows_ncols
    mosaic = np.empty(((1 + slc)*nr, nc), dtype='<U6')
    for i in range(nr):
        for j in range(nc):
            if slc:
                mosaic[2*i, j] = f'slc_{i}{j}'
                mosaic[2*i + 1, j] = f'img_{i}{j}'
            else:
                mosaic[i, j] = f'img_{i}{j}'

    match cb:
        case 'each':
            width_ratios = [1 - cb_width_ratio, cb_width_ratio]*nc
            for j in range(nc):
                mosaic = np.insert(mosaic, 2*j+1, '.', axis=1)
                for i in range(nr):
                    mosaic[2*i+1 if slc else i, 2*j+1] = f'cb_{i}{j}'
        case 'row':
            width_ratios = [1 - cb_width_ratio]*nc + [cb_width_ratio]
            mosaic = np.insert(mosaic, nc, '.', axis=1)
            for i in range(nr):
                mosaic[2*i+1 if slc else i, -1] = f'cb_{i}{nc-1}'
        case 'single':
            width_ratios = [1 - cb_width_ratio]*nc + [cb_width_ratio]
            mosaic = np.insert(mosaic, nc, f'cb_0{nc-1}', axis=1)
        case _:
            width_ratios = [1]*nc

    if slc:
        height_ratios = [slc_height_ratio, 1 - slc_height_ratio]*nr
    else:
        height_ratios = [1]*nr

    axs_dct = fig.subplot_mosaic(mosaic, height_ratios=height_ratios, width_ratios=width_ratios,
                                 **kwargs)
    axs = np.empty((nr, nc), dtype=object)
    for i in range(nr):
        for j in range(nc):
            axs[i, j] = {itm: axs_dct.get(f'{itm}_{i}{j}', None)
                         for itm in {'img', 'cb' if cb else 'img', 'slc' if slc else 'img'}}

    if sharex in (True, 'col'):
        axall = []
        for j in range(nc):
            for i, _ in enumerate(ax := list(itertools.chain(*(
                    [a for k, a in ax.items() if not k.startswith('cb') and k != '.']
                    for ax in axs[:, j]
            )))):
                if i > 0:
                    ax[i].sharex(ax[i-1])
            axall.append(ax)
            if sharex is True:
                axall[j][0].sharex(axall[j-1][-1])
    if sharey in (True, 'row'):
        for typ in ('img',) + (('slc',) if slc else ()):
            axall = []
            for i in range(nr):
                for j, _ in enumerate(ax := list(itertools.chain(*(
                        [a for k, a in ax.items() if k.startswith(typ)]
                        for ax in axs[i, :]
                )))):
                    if j > 0:
                        ax[j].sharey(ax[j-1])
                axall.append(ax)
                if sharey is True:
                    axall[i][0].sharey(axall[i-1][-1])

    return axs


def plot_pl(das, ylabel='', scaley=1, norm=None, figsize=None, aspect=False,
            cbar_location='right', cbar_mode='single', cbar_pad=0.05, **grid_kw):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, aspect=aspect, cbar_location=cbar_location, cbar_mode=cbar_mode,
                     cbar_pad=cbar_pad, **grid_kw)

    if norm is None and cbar_mode == 'single':
        norm = mpl.colors.Normalize(vmin=0, vmax=max(da.max() for da in das).item())

    for da, ax in zip(das, grid):
        x = da.coords['ccd_horizontal_axis']
        y = da.coords[itertools.first_true(da.coords, pred=lambda c: c != 'ccd_horizontal_axis')]
        img = ax.pcolormesh(
            x, y*scaley, da, cmap=SEQUENTIAL_CMAP,
            norm=mpl.colors.Normalize(vmin=0) if norm is None and cbar_mode == 'each' else norm,
            rasterized=True
        )
        cb = ax.cax.colorbar(img, extend='neither')
        prefix = reformat_axis(cb, da, 'cps', 'c')
        ax.set_xlabel('$E$ (eV)')
        ax.set_ylabel(ylabel)
        ax.label_outer()

        ax2, unit = secondary_axis(ax)
        ax2.set_xlabel(rf'$\lambda$ ({unit})')

    cb.set_label(f'Count rate ({prefix}cps)')
    return fig, grid


def plot_pl_slice(fig, axs, da, sel: dict[str, ...], slice_vals: Sequence[...],
                  vertical_target=None, ylabel='', scaley=1, slice_scales: Sequence[...] = None,
                  norm=None, tick_pad=3.5, **line_kwargs):

    if norm is None:
        norm = mpl.colors.Normalize(vmin=0)

    if slice_scales is None:
        slice_scales = [1] * len(slice_vals)

    for i, scale in enumerate(slice_scales):
        if scale is None:
            slice_scales[i] = 1

    x = da.coords['ccd_horizontal_axis']
    y = da.coords[vertical_target := vertical_target if vertical_target is not None else
                  itertools.first_true(da.coords, pred=lambda c: c != 'ccd_horizontal_axis')]

    img = axs['img'].pcolormesh(x, scaley*y, da.sel(sel, method='nearest'),
                                cmap=SEQUENTIAL_CMAP, norm=norm, rasterized=True)
    for slice_val, color in zip(slice_vals, LINE_COLORS):
        axs['img'].axhline(scaley*y.sel({vertical_target: slice_val}, method='nearest'),
                           **(sliceprops(color) | line_kwargs))
    axs['img'].set_xlabel('$E$ (eV)', labelpad=tick_pad)
    axs['img'].set_ylabel(ylabel, labelpad=tick_pad)
    axs['img'].tick_params(pad=tick_pad)

    if axs.get('cb', None) is not None:
        cb = fig.colorbar(img, cax=axs['cb'], extend='neither')
        prefix = reformat_axis(cb, da.sel(sel, method='nearest'), 'cps', 'c')
        cb.set_label(f'Count rate ({prefix}cps)', labelpad=tick_pad)
        cb.ax.tick_params(pad=tick_pad)

    for slice_val, slice_scale, color in zip(slice_vals, slice_scales, LINE_COLORS):
        ln, = axs['slc'].plot(
            x, slice_scale*da.sel(sel | {vertical_target: slice_val}, method='nearest'),
            color=color, **line_kwargs
        )
    axs['slc'].grid(axis='y')
    axs['slc'].tick_params(pad=tick_pad)
    axs['slc'].xaxis.set_tick_params(which="both", labelbottom=False)
    prefix = reformat_axis(axs['slc'], da.sel(sel, method='nearest'), 'cps', 'y')
    ax2, unit = secondary_axis(axs['slc'])
    ax2.set_xlabel(rf'$\lambda$ ({unit})', labelpad=tick_pad)
    ax2.tick_params(pad=tick_pad)

    return img, prefix, ax2


def parabola(x, a, b, c):
    return a*(x - b)**2 + c


# %% 2DEG PL sketch
ΔE_g = E_AlGaAs(0.33) - E_AlGaAs(0)
Q_e = 0.57
ΔE_c = Q_e * ΔE_g
ΔE_v = (1 - Q_e) * ΔE_g
k_F = 0.7
E_F = k_F**2
mu = ΔE_c + E_F
r = np.divide(*effective_mass()).item()
s_dE = (r'$E_\mathrm{g} + '
        r'E_\mathrm{F}\left(1 + \frac{m_{\mathrm{c}}^\ast}{m_{\mathrm{hh}}^\ast}\right)$')

txtfontsize = 'medium'
arrowprops = dict(arrowstyle='<->', mutation_scale=7.5, color=RWTH_COLORS_50['black'],
                  linewidth=0.75, shrinkA=0, shrinkB=0)
annotate_kwargs = dict(color=RWTH_COLORS_75['black'], arrowprops=arrowprops)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.5))

    k = np.linspace(-1, 1, 1001)
    ax.plot(k, k**2 + ΔE_c, color='k')
    ax.plot(k, -r*k**2 - ΔE_v, color='k')
    ax.fill_between(k[abs(k) <= k_F], k[abs(k) <= k_F]**2 + ΔE_c, k_F**2 + ΔE_c,
                    color=RWTH_COLORS_50['blue'], linewidth=0,
                    hatch='', edgecolor=RWTH_COLORS_50['blue'], hatch_linewidth=5)
    ax.plot(np.array([-1.2, 1.2])*k_F, np.array([1, 1])*mu, ls='--')

    ax.annotate('', (0, ΔE_c), (0, -ΔE_v), **annotate_kwargs)
    ax.text(0.05, 0, r'$E_\mathrm{g}$', verticalalignment='center', fontsize=txtfontsize)

    ax.annotate('', (-k_F, mu), (-k_F, -r*k_F**2 - ΔE_v), **annotate_kwargs)
    ax.text(-k_F - 0.075, mu + 0.06, s_dE, verticalalignment='bottom', fontsize=txtfontsize)

    ax.text(0.9, mu + 0.1, r'$E_\mathrm{c}$', verticalalignment='bottom', fontsize=txtfontsize)
    ax.text(0.9, -ΔE_v - 0.1, r'$E_\mathrm{hh}$', fontsize=txtfontsize)

    ax.set_xticks([-k_F, 0, k_F], [r'$-k_\mathrm{F}$', '$0$', r'$k_\mathrm{F}$'],
                  verticalalignment='bottom')
    ax.set_yticks([mu], [r'$\mu$'])
    ax.tick_params('x', pad=12.5)

    ax.margins(x=0.05)

    ax.set_xlabel(r'$k_\parallel$', verticalalignment='center')
    ax.set_ylabel('$E$', rotation='horizontal', horizontalalignment='center')
    ax.xaxis.set_label_coords(1.05, -0.01)
    ax.yaxis.set_label_coords(0, 1.05)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    fig.savefig(SAVE_PATH / '2deg_sketch.pdf')

# %% Honey H13
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH / 'honey_H13.db')

    # Plenty of currents measured that are not interesting, so reduce file size in repo
    to_skip = (
        'honey_H13_trap_1_tunnel_north_bottom',
        'honey_H13_trap_1_tunnel_south_top_current',
        'honey_H13_trap_1_tunnel_south_bottom_current',
        'honey_H13_trap_1_central_top_current',
        'honey_H13_trap_1_central_bottom_current',
        'honey_H13_trap_1_guard_north_top_current',
        'honey_H13_trap_1_guard_north_bottom_current',
        'honey_H13_trap_1_guard_south_top_current',
        'honey_H13_trap_1_guard_south_bottom_current'
    )
    save_to_hdf5(21, DATA_PATH / 'honey_H13_bot_gate_stark_shift.h5', *to_skip)
    save_to_hdf5(22, DATA_PATH / 'honey_H13_top_gate_stark_shift.h5', *to_skip)
# %%% Top vs bottom gate
ds_top = xr.load_dataset(DATA_PATH / 'honey_H13_top_gate_stark_shift.h5')
ds_bot = xr.load_dataset(DATA_PATH / 'honey_H13_bot_gate_stark_shift.h5')
da_top = ds_top['ccd_ccd_data_rate_bg_corrected']
da_bot = ds_bot['ccd_ccd_data_rate_bg_corrected']

fig, grid = plot_pl((da_bot, da_top), ylabel=r'$V_{\mathrm{gate}}$ (V)', figsize=(TEXTWIDTH, 1.1),
                    nrows_ncols=(1, 2), axes_pad=0.35, cbar_mode='each')
grid[0].set_ylim(top=0.55)
fig.savefig(SAVE_PATH / 'honey_H13_stark_shift_vs_gate.pdf')

# %% Doped M1_05_49-2
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH / 'membrane_doped_M1_05_49-2.db')

    save_to_hdf5(41, DATA_PATH / 'doped_M1_05_49-2_difference_mode.h5')
    save_to_hdf5(69, DATA_PATH / 'doped_M1_05_49-2_power.h5')
    save_to_hdf5(140, DATA_PATH / 'doped_M1_05_49-2_multiplets.h5')
    save_to_hdf5(255, DATA_PATH / 'doped_M1_05_49-2_2deg.h5')

# %%% Unbiased PL
da = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_2deg.h5')[
    'ccd_ccd_data_rate_bg_corrected'
]

txtfontsize = 'small'
arrowprops = dict(arrowstyle='<->', mutation_scale=7.5, color=RWTH_COLORS_75['black'],
                  linewidth=0.75, shrinkA=0, shrinkB=0)
annotate_kwargs = dict(color=RWTH_COLORS_75['black'], fontsize=txtfontsize, arrowprops=arrowprops)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.5))
    ax.plot(x := da.ccd_horizontal_axis, y := da.sel(anc_y_axis_steps=-75))
    spl = sc.interpolate.make_smoothing_spline(x[y != 0], np.log10(abs(y)[y != 0]), lam=5e-11)
    ax.plot(x, 10**spl(x))
    ax.axline((1.49554, 23.7781), (1.5008, 90.1246), alpha=0.66, ls='--',
              color=RWTH_COLORS_75['black'])
    ax.axline((1.52719, 917.206), (1.52819, 17.8526), alpha=0.66, ls='--',
              color=RWTH_COLORS_75['black'])
    ax.axvline(x1 := 1.506, ls=':', color=RWTH_COLORS_75['black'])
    ax.axvline(x2 := 1.5277, ls=':', color=RWTH_COLORS_75['black'])

    ax.set_yscale('log')
    ax.set_xlim(1.49, 1.535)
    ax.set_ylim(8)
    ax.set_xlabel('$E$ (eV)')
    ax.set_ylabel('Count rate (cps)')
    ax2, unit = secondary_axis(ax)
    ax2.set_xlabel(rf'$\lambda$ ({unit})')

    ax.annotate('', (x1, 15), (x2, 15), **annotate_kwargs)
    ax.annotate(r'$E_\mathrm{F}\left(1 + \frac{m_{\mathrm{c}}^\ast}{m_{\mathrm{hh}}^\ast}\right)$',
                ((x2-x1)/2 + x1, 25), ha='center', va='bottom', fontsize=txtfontsize,
                color=RWTH_COLORS['black'])

    fig.savefig(SAVE_PATH / '2deg_pl.pdf')

m = effective_mass()
mu = 1/(1/m).sum()
E_F = (x2 - x1) / (1 + np.divide(*m)).item()
n = m[0, 0]*const.e*E_F/(const.pi*const.hbar**2)
k_F = np.sqrt(2*np.pi*n)
S = const.hbar**2*k_F**2/(2*mu)/const.e
print(f'n = {n*1e-4:.2g} /cm²')
print(f'k_F = {k_F:.2g} /m')
print(f'E_F = {E_F:.2g} meV')
# Delalande 87
print(f'Stokes shift = {S:.2g} meV')
# %%% Difference mode
da = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_difference_mode.h5')[
    'ccd_ccd_data_bg_corrected_per_second'
][0]

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig = plt.figure(layout='constrained', figsize=(MARGINWIDTH, 1.8))
    axs = generate_mosaic(fig, (1, 1), slc=True, slc_height_ratio=1/3, sharex=True)
    img, prefix, ax2 = plot_pl_slice(
        fig, axs[0, 0], da, ylabel=r'$V_{\mathrm{DM}}$ (V)',
        vertical_target='doped_M1_05_49_2_trap_2_central_difference_mode',
        sel=dict(), slice_vals=[V0 := 0.75, -1.1, -2.3]
    )

    E0 = 1.5375
    axs[0, 0]['img'].plot(
        parabola(
            da.doped_M1_05_49_2_trap_2_central_difference_mode, a := -3.5e-3, V0, E0
        ),
        da.doped_M1_05_49_2_trap_2_central_difference_mode,
        ls='--', lw=0.75, color=RWTH_COLORS_75['black'], alpha=0.66
    )
    axs[0, 0]['img'].set_xlim(1.48, 1.54)

    fig.get_layout_engine().set(w_pad=2/72, h_pad=1/72)
    fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_difference_mode.pdf')

# %%% Power dependence
da = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_power.h5')[
    'ccd_ccd_data_bg_corrected_per_second'
]
annotate_kwargs = dict(
    color=RWTH_COLORS_50['black'],
    horizontalalignment='center',
    verticalalignment='top',
    arrowprops=dict(arrowstyle='->', mutation_scale=7.5, color=RWTH_COLORS_50['black'])
)

with mpl.style.context(MAINSTYLE, after_reset=True):
    fig, grid = plot_pl((da,), scaley=1e9, ylabel='$P$ (nW)', nrows_ncols=(1, 1), cbar_size='4%',
                        cbar_pad=0.1, norm=mpl.colors.AsinhNorm(linear_width=10))
    ax = grid[0]

    ax.set_yscale('log')
    ax.set_xlim(1.472, 1.497)
    ax.yaxis.minorticks_on()
    cbar = ax.get_children()[0].colorbar
    cbar.set_ticks(np.delete(cbar.get_ticks(), 0))

    # Guides to the eye
    # ax.plot([1.4805, 1.4855, 1.487], [1, 7, 1000], ls='--', color=RWTH_COLORS_50['black'], alpha=0.66)
    ax.plot([1.4805, 1.4865], [1, 10], ls='--', color=RWTH_COLORS_50['black'], alpha=0.66)
    ax.plot([1.4847, 1.4852], [3e0, 1e3], ls='--', color=RWTH_COLORS_50['black'], alpha=0.66)

    ax.annotate('1', (1.480, 2e1), (1.480, .66e1), **annotate_kwargs)
    ax.annotate('2', (1.48175, 2e1), (1.48175, .66e1), **annotate_kwargs)
    ax.annotate('3', (1.492, 3e2), (1.492, 1e2), **annotate_kwargs)

    fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_power.pdf')

# %%% Multiplets
# // Interactive exploration. Use browse_db alternatively //
# Gate voltage dependence
# fig, ax, sliders = plot_nd(140, slider_scale={'excitation_path_power_at_sample': 'log'})
# Power dependence
# fig, ax, sliders = plot_nd(140, vertical_target='power', yscale='log')

ds = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_multiplets.h5')
print_params(ds)

fig = plt.figure(layout='constrained', figsize=(TOTALWIDTH, 5))
axs = generate_mosaic(fig, (3, 4), slc=True, cb='row', cb_width_ratio=1/15, slc_height_ratio=1/3,
                      sharex=True, sharey='row')

Vs = [-1.92, -1.98, -2.02, -2.08]
Ps = [7e-9, 16e-9, 35e-9]
wavs = [790, 805, 815]

for i, (ax, wav) in enumerate(zip(axs, wavs)):
    da = ds['ccd_ccd_data_rate_bg_corrected'].sel(
        excitation_path_wavelength=wav,
        doped_M1_05_49_2_trap_2_central_top=Vs,
        method='nearest'
    )
    slc_max = da.sel(excitation_path_power_at_sample=Ps, method='nearest').max()
    norm = mpl.colors.Normalize(vmin=0, vmax=da.max())

    for j, (a, V) in enumerate(zip(ax, Vs)):
        img, prefix, ax2 = plot_pl_slice(
            fig, a,
            ds['ccd_ccd_data_rate_bg_corrected'],
            scaley=1e9,
            vertical_target='excitation_path_power_at_sample',
            sel=dict(excitation_path_wavelength=wav, doped_M1_05_49_2_trap_2_central_top=V),
            slice_vals=Ps,
            slice_scales=[5, 2.2, 1],
            norm=norm,
        )
        a['img'].set_xlim(1.477, 1.497)
        a['img'].set_yscale('log')
        a['img'].yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
        a['img'].yaxis.set_major_formatter(mpl.ticker.LogFormatter())
        a['img'].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        a['img'].set_xlabel(None)
        a['img'].label_outer()

        a['slc'].label_outer()
        a['slc'].text(0.025, 0.265, r'$\times 5$', color='tab:blue', verticalalignment='center',
                      horizontalalignment='left', transform=a['slc'].transAxes, fontsize='small')
        a['slc'].text(0.025, 0.515, r'$\times 2.2$', color='tab:green', verticalalignment='center',
                      horizontalalignment='left', transform=a['slc'].transAxes, fontsize='small')

        ax2.set_xlabel(None)
        ax2.set_xticks([825, 830, 835, 840])

        if j == 0:
            match backend:
                case 'pgf':
                    a['img'].set_ylabel(rf'$\lambda_{{\mathrm{{exc}}}} = \qty{{{wav}}}{{nm}}$')
                case _:
                    a['img'].set_ylabel(rf'$\lambda_{{\mathrm{{exc}}}} = {wav}$ nm')
        if i == 0:
            match backend:
                case 'pgf':
                    a['slc'].set_title(rf'$V_{{\mathrm{{T}}}} = \qty{{{V:.2f}}}{{\volt}}$',
                                       fontsize='medium')
                case _:
                    a['slc'].set_title(rf'$V_{{\mathrm{{T}}}} = {V:.2f}$ V',
                                       fontsize='medium')
        else:
            ax2.xaxis.set_tick_params(which="both", labeltop=False)

    a['slc'].set_ylim(0)

fig.text(0.5, -0.03, r'$E$ (eV)', fontsize='medium')
fig.supxlabel(r'$\lambda$ (nm)', y=1.05, va='top', fontsize='medium')
fig.supylabel(r'$P$ (nW)', x=-0.04, fontsize='medium')
fig.get_layout_engine().set(w_pad=2/72, h_pad=0/72, hspace=0, wspace=0)
fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_multiplets.pdf')
# %% Fig F10
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH / 'fig_F10.db')

    save_to_hdf5(4, DATA_PATH / 'fig_F10_positioning.h5')
# %%% Positioning
ds = xr.load_dataset(DATA_PATH / 'fig_F10_positioning.h5')
# The first ten steps are hysteresis-afflicted, so drop them to extract x(steps)
fit = ds.anc_y_axis_position[ds.anc_y_axis_steps >= 10].polyfit('anc_y_axis_steps', 1)

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig = plt.figure(layout='constrained', figsize=(MARGINWIDTH, 1.7))
    axs = generate_mosaic(fig, 1, slc=True, cb=False, slc_height_ratio=1/3.5)

    img, prefix = plot_pl_slice(fig, axs[0], ds['ccd_ccd_data_rate_bg_corrected'],
                                ylabel='Positioner steps', sel=dict(), slice_vals=[26, 45],
                                slice_scales=[20, 1], tick_pad=PAD, linewidth=.75)

    axs[0]['slc'].annotate(r'$\times 20$', (1.475, 300), color=LINE_COLORS[0])
    axs[0]['img'].set_ylim(top=53)
    ax2y = axs[0]['img'].secondary_yaxis(
        -0.275,
        functions=(lambda x: x*fit.polyfit_coefficients[0].item()*1e3,
                   lambda x: x/fit.polyfit_coefficients[0].item()/1e3)
    )
    ax2y.tick_params(pad=PAD)
    ax2y.set_ylabel(r'$\Delta x$ (μm)', labelpad=PAD)

    cax = axs[0]['img'].inset_axes([0.05, 0.075, 0.05, 0.85])
    cb = fig.colorbar(img, cax=cax)
    prefix = reformat_axis(cb, ds['ccd_ccd_data_rate_bg_corrected'], 'cps', 'c')
    cb.set_label(f'Rate ({prefix}cps)', labelpad=PAD)
    cb.ax.tick_params(pad=PAD)

    fig.get_layout_engine().set(w_pad=2/72, h_pad=0/72, hspace=0, wspace=0)
    fig.savefig(SAVE_PATH / 'fig_F10_positioning.pdf')
