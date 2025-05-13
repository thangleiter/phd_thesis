# %% Imports
import json
import pathlib
import sys
from unittest import mock

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import interpolate, optimize
import numpy as np
import xarray as xr
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from qutil import const
from qutil.functools import partial
from qutil.itertools import absmax, minmax
from qutil.plotting import changed_plotting_backend
from qutil.plotting.colors import (RWTH_COLORS_25, RWTH_COLORS,
                                   make_diverging_colormap, make_sequential_colormap)
from qutil.ui.gate_layout import GateLayout

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (  # noqa
    init, markerprops, PATH, TOTALWIDTH, TEXTWIDTH, MARGINWIDTH, MAINSTYLE, MARGINSTYLE
)

DATA_PATH = PATH.parent / 'data/cooling'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)
DIVERGING_CMAP = make_diverging_colormap(('magenta', 'green'))
with np.errstate(divide='ignore'):
    SEQUENTIAL_CMAP = make_sequential_colormap('red', endpoint='blackwhite')

init(MARGINSTYLE, backend := 'inline')

# %% Functions


def Qdot(T, a, b):
    # Otten Phd Eq. (4.8)
    return a * T**2 + b


# %% ANC readout heating
temp = pd.read_table(DATA_PATH / 'anc_readout_heating.txt', skiprows=1, sep='\t+', engine='python',
                     index_col=0)['MXC Temperature (mK)'].to_xarray()
volt = temp.coords['Voltage (mV)']
fit = temp.polyfit('Voltage (mV)', 2).polyfit_coefficients
x = volt.interp({'Voltage (mV)': np.linspace(0, 300*1.05, 1001)},
                kwargs={"fill_value": "extrapolate"})

fig, ax = plt.subplots(layout='constrained')
ax.margins(x=0.05, y=0.1)
temp.plot(ax=ax, **markerprops(RWTH_COLORS['blue']))
ax.set_xlim(ax.get_xlim())  # freezes them
ax.plot(x, xr.polyval(x, fit))
ax.grid()
ax.set_xlabel(r'$V_{\mathrm{AC}}$ (mV)')
ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (mK)')

fig.savefig(SAVE_PATH / 'anc_readout_heating.pdf')

# %% Window heating
temp = pd.read_table(DATA_PATH / 'window_heating.txt', skiprows=1, sep='\t+', engine='python',
                     index_col=0)['MXC Temperature (mK)']
x = [0, 1, 2]

cycle = cycler(color=mpl.color_sequences['rwth'][:3], marker=['o', 'D', 'v'], markersize=[5, 5, 6])


fig, ax = plt.subplots(layout='constrained')
ax.margins(x=0.05, y=0.1)
ax.grid(axis='y')

for i, (xx, sty) in enumerate(zip(x, cycle)):
    ax.plot(xx, temp.to_numpy()[::-1][i], **markerprops(**sty))

ax.set_xticks(x, labels=['PT1+PT2+Still', 'Cold', 'None'])
ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (mK)')

fig.savefig(SAVE_PATH / 'window_heating.pdf')

# %% Laser heating
heater = pd.read_table(DATA_PATH / 'mxc_heater_temperature.txt', skiprows=3, sep='\t+',
                       engine='python', index_col=0)['Temperature (K)'].to_xarray()
laser = xr.load_dataarray(DATA_PATH / 'laser_absorption.h5')

fig, ax = plt.subplots(layout='constrained',
                       figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 1.25))

ix = 2
T = heater.data * 1e3
P = heater["Power (uW)"].data
ax.plot(P[ix:], T[ix:], **markerprops(RWTH_COLORS['magenta']))
ax.plot(P[:ix], T[:ix],
        **markerprops(RWTH_COLORS['magenta'], markeredgealpha=0.5, markerfacealpha=0.1))

popt_heater, _ = optimize.curve_fit(Qdot, T[ix:], P[ix:])
ax.plot(Qdot(x := np.geomspace(3, 40, 1001), *popt_heater), x, color=RWTH_COLORS['magenta'])

ix = 0
T = laser.data * 1e3
P = laser['excitation_path_power_at_sample'].data * 1e6
ax.plot(P[ix:], T[ix:], **markerprops(RWTH_COLORS['green'], marker='D', markersize=4))
ax.plot(P[:ix], T[:ix], **markerprops(
    RWTH_COLORS['green'], marker='D', markersize=4, markeredgealpha=0.5, markerfacealpha=0.1
))

popt_laser, _ = optimize.curve_fit(partial(Qdot, b=popt_heater[1]), T[ix:], P[ix:])
ax.plot(Qdot(x := np.geomspace(3, 40, 1001), *popt_laser, popt_heater[1]), x,
        color=RWTH_COLORS['green'])

spline = interpolate.make_splrep(P, T, s=5e-5)
pop_spline, _ = optimize.curve_fit(lambda x, R: spline(x / R),
                                   heater["Power (uW)"].data, heater.data * 1e3)

spline = interpolate.make_splrep(P, T * 1e-3, s=5e-8)
pop_spline, _ = optimize.curve_fit(lambda x, R: spline(x / R),
                                   heater["Power (uW)"].data, heater.data)

ax.plot(x := np.geomspace(3e-1, P[-1], 1001), spline(x) * 1e3, color=RWTH_COLORS['green'], ls='--')
ax.plot(x, spline(x / pop_spline) * 1e3, color=RWTH_COLORS['magenta'], ls='--')

ax.grid()
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim(3, 40)
# ax.set_ylim(5, 40)
ax.set_xlabel('Power (μW)')
ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (mK)')

fig.savefig(SAVE_PATH / 'laser_heating.pdf')
