# %% Imports
import pathlib
import sys
import json
from unittest import mock

import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from qutil.plotting.colors import (RWTH_COLORS_25, RWTH_COLORS,
                                   make_diverging_colormap, make_sequential_colormap)
from qutil.plotting import changed_plotting_backend
from qutil import const
from qutil.itertools import absmax
from qutil.ui.gate_layout import GateLayout

import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import PATH, TOTALWIDTH, TEXTWIDTH, MARGINWIDTH, MAINSTYLE, MARGINSTYLE, MARKERSTYLE # noqa

ORIG_DATA_PATH = pathlib.Path(r'\\janeway\User AG Bluhm\Common\GaAs\PL Lab\Data\Triton\2022-07-13')
DATA_PATH = PATH.parent / 'data/transport'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)
DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'))
with np.errstate(divide='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('red', endpoint='blackwhite')

backend = 'pgf'
mpl.style.use(MAINSTYLE)

if (ipy := IPython.get_ipython()) is not None:
    ipy.run_line_magic('matplotlib', backend)

# %% Definitions


def load_and_sanitize(file, resave=False):
    with np.load(ORIG_DATA_PATH / file, allow_pickle=True) as loaded:
        coords = {}
        for d in ('y', 'x')[2-loaded['data'].ndim:]:
            if len(loaded[f'{d}_params']) > 1:
                # Virtual gate. We define V = A + x B with a mixing factor x
                # I.e., the A "leverarm" is 1, and that of B is x.
                x = np.divide(*np.diff(loaded[f'{d}_rngs']).squeeze()[::-1])
                X = np.array([1, x]) / np.linalg.norm([1, x])
                long_name = ' + '.join([
                    f"{x:.2f} {param}" for x, param in zip(X, loaded[f'{d}_params'])
                ])
            else:
                X = np.array([1])
                long_name = loaded[f'{d}_params'].item()

            coords['_'.join(loaded[f'{d}_params'])] = xr.DataArray(
                (np.linspace(*loaded[f'{d}_rngs'].T, loaded[f'{d}_pts']) * X).sum(axis=-1),
                coords={'_'.join(loaded[f'{d}_params']): np.arange(loaded[f'{d}_pts'])},
                name='_'.join(loaded[f'{d}_params']),
                attrs={
                    'rngs': loaded[f'{d}_rngs'],
                    'ramptime': loaded[f'{d}_ramptime'],
                    'ramprates': loaded[f'{d}_ramprates'],
                    'waittime': loaded[f'{d}_waittime'],
                    'long_name': long_name,
                    'units': 'V',
                }
            )

        attrs = {
            'voltages': loaded['voltages'].item(),
            'timestamp': loaded['timestamp'].item(),
            'ohmics': loaded['ohmics'].item(),
            'gain': loaded['gain'].item(),
            'fcut': loaded['fcut'].item(),
            'bias_at': loaded['bias_at'].item(),
            'long_name': 'DMM voltage',
            'units': 'V'
        }
        if 'bias' in loaded:
            attrs['bias'] = loaded['bias'].item()

        data = xr.DataArray(
            loaded['data'],
            coords=coords,
            name='dmm_voltage',
            attrs=attrs
        )

    if resave:
        with mock.patch.dict(data.attrs, {
                'voltages': json.dumps(data.attrs['voltages']),
                'timestamp': str(data.attrs['timestamp'])
        }):
            data.to_netcdf((DATA_PATH / file).with_suffix('.h5'), engine='h5netcdf')

    return data


def dfermi(x, Γ, V_0, T, offset):
    prefactor = 1/(4*const.k*T)*const.e**2/2/const.h
    return prefactor*Γ/np.cosh(const.e*(alpha*x - V_0)/(2*const.k*T))**2 + offset


# %% Diamonds
file = 'coulomb_diamonds_2022-07-13_17-56-52.npz'
# dmm_voltage_diamonds = load_and_sanitize(file, resave=True)
dmm_voltage_diamonds = xr.load_dataarray((DATA_PATH / file).with_suffix('.h5'))
tia_current_diamonds = dmm_voltage_diamonds / dmm_voltage_diamonds.attrs['gain'] * 1e12
tia_current_diamonds.attrs.update(units='pA', long_name='TIA Current')

# %%% Leverarm
c = tia_current_diamonds.differentiate('NBC_TBC')
x = tia_current_diamonds.NBC_TBC

c1 = c.isel(Bias=0).where((x >= -0.515) & (x <= -0.513), drop=True)
c2 = c.isel(Bias=-1).where((x >= -0.499) & (x <= -0.497), drop=True)
c3 = c.isel(Bias=-1).where((x >= -0.497) & (x <= -0.495), drop=True)
c4 = c.isel(Bias=0).where((x >= -0.494) & (x <= -0.493), drop=True)

xs = np.array([[c1.idxmin(), c2.idxmin()],
               [c3.idxmax(), c4.idxmax()]])
ys = np.array([[c1.idxmin().Bias, c2.idxmin().Bias],
               [c3.idxmax().Bias, c4.idxmax().Bias]])

a = (np.diff(xs)/np.diff(ys)).squeeze()
b = xs[:, 0] - a * ys[:, 0]

ΔV = np.diff(b).item()
E_c = abs(np.diff(b) / np.diff(a)).item()
alpha = E_c / ΔV

# %%% GL
with mpl.style.context([MARGINSTYLE, {'patch.linewidth': 0.25}], after_reset=True):
    gl = GateLayout(DATA_PATH / 'gl_005d.dxf',
                    foreground_color=RWTH_COLORS_25['black'],
                    cmap=SEQUENTIAL_CMAP,
                    # cmap=mpl.colormaps.get_cmap('afmhot'),
                    v_min=-1)
    gl.ax.set_aspect('equal')
    gl.ax.set_axis_off()
    # gl.ax.grid()
    for txt in gl.ax.texts:
        if not any(
                txt.get_text().startswith(dim) for dim in dmm_voltage_diamonds.dims[-1].split("_")
        ):
            txt.set_visible(False)
        else:
            txt.set_visible(False)

    gl.ax.text(1.29, +1.15, 'NBC', horizontalalignment='center', verticalalignment='bottom')
    gl.ax.text(1.29, -0.15, 'TBC', horizontalalignment='center', verticalalignment='top')

    cb = gl.fig.colorbar(
        mpl.cm.ScalarMappable(gl.norm, gl.cmap), ax=gl.ax, label='$V$ (V)',
        shrink=.33, panchor=(1.0, 0.0), pad=-0.05, aspect=7.5
    )

    gl.update(json.loads(dmm_voltage_diamonds.attrs['voltages']))

    gl.fig.tight_layout()
    gl.fig.savefig(SAVE_PATH / 'diamonds_gl.pdf', backend='pdf' if backend == 'qt' else backend)

# %%% Plot
# TODO: double-check units!
fig = plt.figure(figsize=(TOTALWIDTH, TOTALWIDTH / const.golden / 1.5))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), share_all=True, aspect=False, cbar_mode='each',
                 cbar_location="top", axes_pad=(.2, .05))

image_data = [tia_current_diamonds - np.median(tia_current_diamonds),
              1e-3 * tia_current_diamonds.differentiate('NBC_TBC')]
match backend:
    case 'pgf':
        image_labels = ["$I$ (pA)", r"$\pdv*{I}{V}$ ($\flatfrac{\mathrm{pA}}{\mathrm{mV}}$)"]
    case 'qt':
        image_labels = ["$I$ (pA)", r"$\partial I/\partial V$ (pA/mV)"]

widths = [0.045e3, 40]

x = tia_current_diamonds['NBC_TBC']
y = tia_current_diamonds['Bias']

for ax, cax, img, label, width in zip(grid, grid.cbar_axes, image_data, image_labels, widths):
    norm = mpl.colors.AsinhNorm(width, vmin=-absmax(img.data.flat), vmax=absmax(img.data.flat))
    im = ax.pcolormesh(x * 1e3, y * 1e3, img, cmap=DIVERGING_CMAP, norm=norm, rasterized=True)
    cb = cax.colorbar(im, label=label)
    cb.set_ticks(np.delete(ticks := cb.get_ticks(), len(ticks) // 2))
    ax.set_xlabel(f"{x.attrs['long_name']} (mV)")
    ax.set_ylabel(r"$V_\mathrm{bias}$ (mV)")

ax = grid[0]
txt_kwargs = dict(horizontalalignment='center', verticalalignment='center', transform=ax.transData)
ax.text(-512.5, 0, '$N-1$', **txt_kwargs)
ax.text(-501.0, 0, '$N$', **txt_kwargs)
ax.text(-490.0, 0, '$N+1$', **txt_kwargs)
ax.text(-478.5, 0, '$N+2$', **txt_kwargs)

ax.plot(xs[0] * 1e3, ys[0] * 1e3, '--', color=RWTH_COLORS['black'], alpha=0.3)
ax.plot(xs[1] * 1e3, ys[1] * 1e3, '--', color=RWTH_COLORS['black'], alpha=0.3)

fig.tight_layout()
fig.savefig(SAVE_PATH / 'diamonds.pdf', backend='pdf' if backend == 'qt' else backend)

# %% Plunger sweep
file = 'plunger_1d_2022-07-13_15-40-33.npz'
# dmm_voltage_plunger = load_and_sanitize(file, resave=True)
dmm_voltage_plunger = xr.load_dataarray((DATA_PATH / file).with_suffix('.h5'))
conductance_plunger = (dmm_voltage_plunger
                       / dmm_voltage_plunger.attrs['gain']
                       / dmm_voltage_plunger.attrs['bias']
                       / const.physical_constants['conductance quantum'][0])
conductance_plunger.attrs.update(dmm_voltage_plunger.attrs)
conductance_plunger.attrs.update(units='$G_0$', long_name='$G$')
# %%% Fit
x = conductance_plunger['NBC_TBC']
y = conductance_plunger

# Linear response, equal baths
ix = slice(0, None)

V_0 = alpha*x[ix][y[ix].argmax()]
T = 100e-3
Γ = np.ptp(y[ix].values) / (1/(4*const.k*T)*const.e**2/(2*const.h))
off = np.median(y[ix])

params = ['Γ', 'V_0', 'T', 'off']
bounds = dict(zip(params, zip(
    [Γ/2, alpha*x[ix].min(), 1e-3, y[ix].min()],
    [Γ*2, alpha*x[ix].max(), 1e+1, y[ix].max()],
)))
p0 = dict(zip(params, [Γ, V_0, T, off]))

fit = conductance_plunger[ix].curvefit('NBC_TBC', dfermi, p0=p0, bounds=bounds)

# %%% Plot
with (
        mpl.style.context([MARGINSTYLE, {'axes.formatter.limits': (-3, 6)}], after_reset=True),
        changed_plotting_backend('pgf')
):
    fig, ax = plt.subplots(layout='constrained',
                           figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 1.1))

    ax.plot(x[ix] * 1e3, y[ix] - fit.curvefit_coefficients[-1], ls='',
            color=RWTH_COLORS['blue'],
            marker='o',
            markersize=3,
            markeredgewidth=0.5,
            markeredgecolor=RWTH_COLORS['blue'],
            markerfacecolor=mpl.colors.to_rgb(RWTH_COLORS['blue']) + (0.5,))

    ax.plot(x[ix] * 1e3, dfermi(x[ix], *fit.curvefit_coefficients) - fit.curvefit_coefficients[-1],
            color=RWTH_COLORS['magenta'], marker='')

    ax.set_xlabel(f"{x.attrs['long_name']} (m{x.attrs['units']})")
    ax.set_ylabel(f"{y.attrs['long_name']} ({y.attrs['units']})")

    fig.savefig(SAVE_PATH / 'coulomb_resonance.pdf')

# %%%
conductance_diamonds = dmm_voltage_diamonds.sel(Bias=0, method='nearest')
conductance_diamonds = (conductance_diamonds
                        / conductance_diamonds.Bias
                        / dmm_voltage_plunger.attrs['gain']
                        / const.physical_constants['conductance quantum'][0])
conductance_diamonds.attrs.update(conductance_diamonds.attrs)
conductance_diamonds.attrs.update(units='$G_0$', long_name='$G$')

with (
        mpl.style.context([MARGINSTYLE, {'axes.formatter.limits': (-3, 6)}], after_reset=True),
        # changed_plotting_backend('pgf')
):
    fig, ax = plt.subplots(layout='constrained',
                           figsize=(TEXTWIDTH, TEXTWIDTH / const.golden * 1.0))

    ax.plot(conductance_diamonds['NBC_TBC'] * 1e3, conductance_diamonds)
    ax.set_xlabel(f"{x.attrs['long_name']} (m{x.attrs['units']})")
    ax.set_ylabel(f"{y.attrs['long_name']} ({y.attrs['units']})")

    xlim = conductance_plunger['NBC_TBC'][[0, -1]] * 1e3
    axins = ax.inset_axes([0.075, 0.5, 0.425, 0.425], xlim=xlim)

    axins.plot(x[ix] * 1e3, y[ix] - fit.curvefit_coefficients[-1], ls='',
               color=RWTH_COLORS['blue'],
               marker='o',
               markersize=3,
               markeredgewidth=0.5,
               markeredgecolor=RWTH_COLORS['blue'],
               markerfacecolor=mpl.colors.to_rgb(RWTH_COLORS['blue']) + (0.5,))

    axins.plot(x[ix] * 1e3,
               dfermi(x[ix], *fit.curvefit_coefficients) - fit.curvefit_coefficients[-1],
               color=RWTH_COLORS['magenta'], marker='')
