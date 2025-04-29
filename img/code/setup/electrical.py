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
from scipy import optimize

from qutil.plotting.colors import RWTH_COLORS_25, make_diverging_colormap, make_sequential_colormap
from qutil import const
from qutil.itertools import absmax
from qutil.ui.gate_layout import GateLayout

import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import PATH, TOTALWIDTH, MAINSTYLE, MARGINSTYLE # noqa

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
        coords = {
            '_'.join(loaded[f'{d}_params']): xr.DataArray(
                np.linspace(0, 1, loaded[f'{d}_pts']) if len(loaded[f'{d}_params']) > 1 else
                np.linspace(*loaded[f'{d}_rngs'][0], loaded[f'{d}_pts']),
                coords={'_'.join(loaded[f'{d}_params']): np.arange(loaded[f'{d}_pts'])},
                name='_'.join(loaded[f'{d}_params']),
                attrs={
                    'rngs': loaded[f'{d}_rngs'],
                    'ramptime': loaded[f'{d}_ramptime'],
                    'ramprates': loaded[f'{d}_ramprates'],
                    'waittime': loaded[f'{d}_waittime'],
                    'long_name': ' | '.join(loaded[f'{d}_params']),
                    'units': 'a.u.' if len(loaded[f'{d}_params']) > 1 else 'V',
                }
            )
            for d in ('y', 'x')[2-loaded['data'].ndim:]
        }

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


def dfermi(x, A, V_0, T, offset):
    return A/np.cosh(const.e*(alpha*x - V_0)/(2*const.k*T))**2/(4*const.k*T)*const.e**2/2/const.h + offset


# %% Diamonds
file = 'coulomb_diamonds_2022-07-13_17-56-52.npz'
# dmm_voltage = load_and_sanitize(file)
dmm_voltage = xr.load_dataarray((DATA_PATH / file).with_suffix('.h5'))
tia_current = dmm_voltage / dmm_voltage.attrs['gain'] * 1e12
tia_current.attrs.update(units='pA', long_name='TIA Current')

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
        if not any(txt.get_text().startswith(dim) for dim in dmm_voltage.dims[1].split("_")):
            txt.set_visible(False)
        else:
            txt.set_visible(False)

    gl.ax.text(1.29, +1.15, 'NBC', horizontalalignment='center', verticalalignment='bottom')
    gl.ax.text(1.29, -0.15, 'TBC', horizontalalignment='center', verticalalignment='top')

    cb = gl.fig.colorbar(
        mpl.cm.ScalarMappable(gl.norm, gl.cmap), ax=gl.ax, label='$V$ (V)',
        shrink=.33, panchor=(1.0, 0.0), pad=-0.05, aspect=7.5
    )

    gl.update(json.loads(dmm_voltage.attrs['voltages']))

    gl.fig.tight_layout()
    gl.fig.savefig(SAVE_PATH / 'diamonds_gl.pdf', backend='pdf' if backend == 'qt' else backend)

# %%% Plot
fig = plt.figure(figsize=(TOTALWIDTH, TOTALWIDTH / const.golden / 1.5))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), share_all=True, aspect=False, cbar_mode='each',
                 cbar_location="top", axes_pad=(.2, .05))

image_data = [tia_current - np.median(tia_current), 1e-3 * tia_current.differentiate('NBC_TBC')]
match backend:
    case 'pgf':
        image_labels = [
            f"$I$ ({tia_current.attrs['units']})",
            r"$\pdv*{I}{V}$ ($\flatfrac{\mathrm{nA}}{\mathrm{V}}$)"
        ]
    case 'qt':
        image_labels = [f"$I$ ({tia_current.attrs['units']})",
                        r"$\partial I/\partial V$ (nA/V)"]
widths = [0.05e3, 5]

x = tia_current['NBC_TBC']
y = tia_current['Bias']

for ax, cax, img, label, width in zip(grid, grid.cbar_axes, image_data, image_labels, widths):
    norm = mpl.colors.AsinhNorm(width, vmin=-absmax(img.data.flat), vmax=absmax(img.data.flat))
    im = ax.pcolormesh(x, y * 1e3, img, cmap=DIVERGING_CMAP, norm=norm, rasterized=True)
    cb = cax.colorbar(im, label=label)
    cb.set_ticks(np.delete(ticks := cb.get_ticks(), len(ticks) // 2))
    ax.set_xlabel(f"{x.attrs['long_name']} ({x.attrs['units']})")
    ax.set_ylabel(f"{y.attrs['long_name']} (mV)")

fig.tight_layout()
fig.savefig(SAVE_PATH / 'diamonds.pdf', backend='pdf' if backend == 'qt' else backend)

# %% Plunger sweep
file = 'plunger_1d_2022-07-13_15-40-33.npz'
# dmm_voltage = load_and_sanitize(file, resave=True)
dmm_voltage = xr.load_dataarray((DATA_PATH / file).with_suffix('.h5'))
tia_current = dmm_voltage / dmm_voltage.attrs['gain'] * 1e12
tia_current.attrs.update(dmm_voltage.attrs)
tia_current.attrs.update(units='pA', long_name='TIA Current')
conductance = (tia_current / tia_current.attrs['bias']
               * 1e-12 / const.physical_constants['conductance quantum'][0] * 1e3)
conductance.attrs.update(tia_current.attrs)
conductance.attrs.update(units='m$G_0$', long_name='$G$')
# %%% Fit
alpha = 0.07974104  # coulomb_diamonds_2022-07-13_12-33-31.npz
bias = conductance.attrs['bias']
gain = conductance.attrs['gain']

x = conductance['NBC_TBC']
y = conductance

# Linear response, equal baths
ix = slice(0, None)

V_0 = alpha*x[ix][y[ix].argmax()]
T = 100e-3
A = np.ptp(y[ix]) / (1/(4*const.k*T)*const.e**2/(2*const.h))
off = np.median(y[ix])

bounds = (
    [A/2, x[ix].min(), 1e-3, y[ix].min()],
    [A*2, x[ix].max(), 1e+1, y[ix].max()],
)
p0 = [A, V_0, T, off]

popt, pcov = optimize.curve_fit(dfermi, x[ix], y[ix], p0=p0, bounds=bounds)
