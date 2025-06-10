# %% Imports
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
import pandas as pd
import xarray as xr
from cycler import cycler
from qutil import const, functools
from qutil.plotting.colors import RWTH_COLORS
from scipy import interpolate, optimize
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import (MAINSTYLE, MARGINSTYLE, MARGINWIDTH, PATH, TEXTWIDTH, TOTALWIDTH, init,  # noqa
                    markerprops)

DATA_PATH = PATH.parent / 'data/cooling'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')

# %% Functions


def Qdot(T, a, b):
    # Otten Phd Eq. (4.8)
    return a * T**2 + b


def P_AC(V, R, P_0):
    return V**2/R + P_0


@functools.cache
def planck_mpm(E, kT):
    return E**3 / mpm.expm1(E/kT)


# %% Estimate BBR fraction
T = np.linspace(1e-3, 300, 101)
E_c = const.lambda2eV(mpm.mpf(4.5e-6))

P_0, P_inf = [], []
for t in tqdm(T):
    kT = const.k * mpm.mpf(t) / const.e
    P_0.append(mpm.quad(functools.partial(planck_mpm, kT=kT), [E_c, mpm.inf]))
    P_inf.append(mpm.quad(functools.partial(planck_mpm, kT=kT), [0, mpm.inf]))

F = np.array(P_0) / np.array(P_inf)

fig, ax = plt.subplots(layout='constrained')
ax.semilogy(T, F)
ax.grid()
ax.set_xlim(0)
ax.set_ylim(1e-20, 1e1)
ax.set_yticks([1e-20, 1e-10, 1e-0])
ax.set_xlabel(r'$T$ (K)')
ax.set_ylabel('Rel. radiance')

fig.savefig(SAVE_PATH / 'black_body_radiance.pdf')

# %% Laser heating
heater = pd.read_table(DATA_PATH / 'mxc_heater_temperature.txt', skiprows=3, sep='\t+',
                       engine='python', index_col=0)['Temperature (K)'].to_xarray()
laser = xr.load_dataarray(DATA_PATH / 'laser_absorption.h5')

fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.25))
ylim = (5, 40)

# Heater data
ix = 2
T = heater.data * 1e3  # mK
P = heater["Power (uW)"].data  # μW
ax.plot(P[ix:], T[ix:], **markerprops(RWTH_COLORS['magenta']))
ax.plot(P[:ix], T[:ix],
        **markerprops(RWTH_COLORS['magenta'], markeredgealpha=0.5, markerfacealpha=0.1))

popt_heater, _ = optimize.curve_fit(Qdot, T[ix:], P[ix:])
ax.plot(Qdot(x := np.geomspace(*ylim, 1001), *popt_heater), x, color=RWTH_COLORS['magenta'])

# Laser data
ix = 5
T = laser.data * 1e3  # mK
P = laser['excitation_path_power_at_sample'].data * 1e6  # μW
ax.plot(P[ix:], T[ix:], **markerprops(RWTH_COLORS['green'], marker='D', markersize=4))
ax.plot(P[:ix], T[:ix], **markerprops(
    RWTH_COLORS['green'], marker='D', markersize=4, markeredgealpha=0.5, markerfacealpha=0.1
))

popt_laser, _ = optimize.curve_fit(Qdot, T[ix:], P[ix:])
ax.plot(Qdot(x, *popt_laser), x, color=RWTH_COLORS['green'])

# Fit quadratic smoothing spline to laser data and scale to fit heater data
spline = interpolate.make_splrep(T, P, k=2, s=1)
popt_spline, pcov_spline = optimize.curve_fit(lambda x, A: A * spline(x),
                                              heater.data * 1e3, heater["Power (uW)"].data)

ax.plot(spline(x), x, color=RWTH_COLORS['green'], ls='--')
ax.plot(spline(x) * popt_spline, x, color=RWTH_COLORS['magenta'], ls='--')

ax.grid()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(3e-1)
ax.set_ylim(5, 40)

ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
for loc, txt in zip(ax.yaxis.get_ticklocs(minor=True), ax.yaxis.get_ticklabels(minor=True)):
    if loc not in (5, 20, 40):
        txt.set_visible(False)

match backend:
    case 'pgf':
        ax.set_xlabel(r'Power (\unit{\micro\watt})')
        ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (\unit{\milli\kelvin})')
    case 'qt':
        ax.set_xlabel('Power (μW)')
        ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (mK)')

fig.savefig(SAVE_PATH / 'laser_heating.pdf')

# %% ANC readout heating
temp = pd.read_table(DATA_PATH / 'anc_readout_heating.txt', skiprows=1, sep='\t+', engine='python',
                     index_col=0)['MXC Temperature (mK)'].to_xarray()
volt = temp.coords['Voltage (mV)']
fit = temp.curvefit('Voltage (mV)', P_AC).curvefit_coefficients
x = volt.interp({'Voltage (mV)': np.linspace(0, 300*1.05, 1001)},
                kwargs={"fill_value": "extrapolate"})

fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.1))
ax.margins(x=0.05, y=0.1)
temp.plot(ax=ax, **markerprops(RWTH_COLORS['blue']))
ax.set_xlim(ax.get_xlim())  # freezes them
ax.plot(x, P_AC(x, *fit))
ax.grid()
ax.set_xticks([0, 100, 200, 300])

# Use splines to convert temperature to power from the heater measurements
T = heater.data * 1e3  # mK
P = heater["Power (uW)"].data  # μW

ax2 = ax.secondary_yaxis(
    'right',
    (interpolate.make_splrep(T, P, k=2, s=1),
     interpolate.make_splrep(P, T, k=2, s=1))
)

match backend:
    case 'pgf':
        ax.set_xlabel(r'$V_{\mathrm{AC}}$ (\unit{\milli\volt})')
        ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (\unit{\milli\kelvin})')
        ax2.set_ylabel(r'$P$ (\unit{\micro\watt})')
    case 'qt':
        ax.set_xlabel(r'$V_{\mathrm{AC}}$ (mV)')
        ax.set_ylabel(r'$T_{\mathrm{MXC}}$ (mK)')
        ax2.set_ylabel(r'$P$ (μW)')

fig.savefig(SAVE_PATH / 'anc_readout_heating.pdf')
