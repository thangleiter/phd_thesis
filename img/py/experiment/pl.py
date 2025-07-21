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
import xarray as xr
from mjolnir.helpers import save_to_hdf5
from mjolnir.plotting import plot_nd  # noqa
from mpl_toolkits.axes_grid1 import ImageGrid
from qcodes.dataset import initialise_or_create_database_at
from qutil import itertools
from qutil.plotting import reformat_axis
from qutil.plotting.colors import (make_sequential_colormap,
                                   RWTH_COLORS, RWTH_COLORS_75, RWTH_COLORS_50, RWTH_COLORS_25)

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # noqa

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, TEXTWIDTH, PATH, init,  # noqa
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
LINE_COLORS = [RWTH_COLORS_75['magenta'], RWTH_COLORS_50['magenta']]

PAD = 2

init(MAINSTYLE, backend := 'pgf')
# %% Functions


def generate_mosaic(fig, n: int, slc: bool = False, cb: Literal[False] | str = 'each',
                    slc_height_ratio: float = 1/4, cb_width_ratio: float = 1/20, **kwargs):
    if slc:
        mosaic = [list(itertools.chain.from_iterable(zip([f'slc_{i}' for i in range(n)],
                                                         ['.']*2)))]
        height_ratios = [slc_height_ratio, 1 - slc_height_ratio]
    else:
        mosaic = []
        height_ratios = [1]

    match cb:
        case 'each':
            if slc:
                mosaic = [list(itertools.chain.from_iterable(zip([f'slc_{i}' for i in range(n)],
                                                                 ['.']*2)))]
            else:
                mosaic = []
            mosaic.append(list(itertools.chain.from_iterable(zip([f'img_{i}' for i in range(n)],
                                                                 [f'cb_{i}' for i in range(n)]))))
            width_ratios = [1 - cb_width_ratio, cb_width_ratio]*n
        case 'single':
            if slc:
                mosaic = [[f'slc_{i}' for i in range(n)] + ['.']]
            else:
                mosaic = []
            mosaic.append([f'img_{i}' for i in range(n)] + ['cb_0'])
            width_ratios = [1 - cb_width_ratio]*n + [cb_width_ratio]
        case False:
            if slc:
                mosaic = [[f'slc_{i}' for i in range(n)]]
            else:
                mosaic = []
            mosaic.append([f'img_{i}' for i in range(n)])
            width_ratios = [1]*n

    axs_dct = fig.subplot_mosaic(mosaic, height_ratios=height_ratios, width_ratios=width_ratios,
                                 **kwargs)
    axs = []
    for i in range(n):
        axs.append({itm: axs_dct.get(f'{itm}_{i}', None)
                    for itm in {'img', 'cb' if cb else 'img', 'slc' if slc else 'img'}})

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
    axs['slc'].sharex(axs['img'])
    axs['slc'].xaxis.set_tick_params(which="both", labelbottom=False)
    prefix = reformat_axis(axs['slc'], da.sel(sel, method='nearest'), 'cps', 'y')
    ax2, unit = secondary_axis(axs['slc'])
    ax2.set_xlabel(rf'$\lambda$ ({unit})', labelpad=tick_pad)
    ax2.tick_params(pad=tick_pad)

    return img, prefix


def print_params(ds, voltages=True, wavelength=True, power=True, tex=False):
    print('Measurement:', ds.ds_name)
    s = json.loads(ds.attrs['snapshot'])
    try:
        ep = s['station']['instruments']['excitation_path']['parameters']
        sample = s['station']['instruments'][ds.sample_name]
    except KeyError:
        print('No snapshot.')
        return

    active_trap = sample['parameters']['active_trap']['value'].split(',')[0].lstrip('Trap(name=')
    gates = sample['submodules']['traps']['channels'][active_trap]['parameters']

    if voltages:
        for typ in ('guard', 'central'):
            for mode in ('difference_mode', 'common_mode'):
                if f'{active_trap}_{typ}_{mode}' in gates:
                    if tex:
                        print(r'\thevoltage{', end='')
                    else:
                        print(f'Trap {active_trap} {typ} {mode.replace("_", " ")}: ', end='')
                    print(f"{gates[f'{active_trap}_{typ}_{mode}']['value']:.2f}",
                          end='' if tex else '\n')
                    if tex:
                        print('}{' + typ[0] + ''.join(m[0] for m in mode.split('_')) + '}')
    if wavelength:
        if tex:
            print(r'\thewavelength{', end='')
        else:
            print('Excitation wavelength: ', end='')
        print(f"{ep['wavelength']['value']:.1f}", end='' if tex else '\n')
        if tex:
            print('}')
    if power:
        if tex:
            print(r'\thepower{', end='')
        else:
            print('Excitation power at sample: ', end='')
        print(f"{ep['power_at_sample']['value']*1e6:.2g}", end='' if tex else '\n')
        if tex:
            print(r'}{\micro}')


def fit_peak():
    pass


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
    ax.set_yticks([0, mu], ['', r'$\mu$'])
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
# %%% Difference mode
da = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_difference_mode.h5')[
    'ccd_ccd_data_bg_corrected_per_second'
]

with mpl.style.context(MARGINSTYLE, after_reset=True):
    fig, grid = plot_pl((da,), ylabel=r'$V_{\mathrm{DM}}$ (V)', nrows_ncols=(1, 1))
    grid[0].axhline(0.90, **sliceprops(color=RWTH_COLORS_50['black'], alpha=0.66))
    fig.savefig(SAVE_PATH / 'doped_M1_05_49-2_difference_mode.pdf')

# %%% Power dependence 2
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
# // Use browse_db alternatively //
# Gate voltage dependence
# fig, ax, sliders = plot_nd(140, slider_scale={'excitation_path_power_at_sample': 'log'})
# Power dependence
# fig, ax, sliders = plot_nd(140, vertical_target='power', yscale='log')

ds = xr.load_dataset(DATA_PATH / 'doped_M1_05_49-2_multiplets.h5')
print_params(ds)

fig = plt.figure(layout='constrained', figsize=(TEXTWIDTH, 2.2))
axs = generate_mosaic(fig, 2, slc=True, cb='each', cb_width_ratio=1/15)

img, prefix = plot_pl_slice(
    fig, axs[0],
    ds['ccd_ccd_data_rate_bg_corrected'],
    ylabel=r'$V_{\mathrm{T}}$ (V)',
    vertical_target='doped_M1_05_49_2_trap_2_central_top',
    sel=dict(excitation_path_wavelength=800, excitation_path_power_at_sample=17.7e-9),
    slice_vals=[-2.02],
    tick_pad=PAD,
)
img.colorbar.set_label('')
axs[0]['img'].set_xlim(1.475, 1.50)

img, prefix = plot_pl_slice(
    fig, axs[1],
    ds['ccd_ccd_data_rate_bg_corrected'],
    ylabel=r'$P_{\mathrm{exc}}$ (nW)',
    scaley=1e9,
    vertical_target='excitation_path_power_at_sample',
    sel=dict(excitation_path_wavelength=790, doped_M1_05_49_2_trap_2_central_top=-1.92),
    slice_vals=[35.4e-9],
    tick_pad=PAD,
)

axs[1]['img'].set_xlim(1.475, 1.50)
axs[1]['img'].set_yscale('log')
axs[1]['img'].yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
axs[1]['img'].yaxis.set_major_formatter(mpl.ticker.LogFormatter())

fig.get_layout_engine().set(w_pad=1/72, h_pad=0/72, hspace=0, wspace=0)
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
