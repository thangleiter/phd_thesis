# %% Imports
import os
import pathlib
import sys

import matplotlib as mpl  # noqa
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xarray as xr
from mjolnir.helpers import save_to_hdf5
from mjolnir.plotting import plot_nd  # noqa
from qcodes.dataset import initialise_or_create_database_at
from qutil import const, plotting
from qutil.plotting.colors import RWTH_COLORS
from uncertainties import ufloat

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import MARGINSTYLE, MARGINWIDTH, PATH, init, markerprops, secondary_axis  # noqa

EXTRACT_DATA = os.environ.get('EXTRACT_DATA', False)
ORIG_DATA_PATH = pathlib.Path(
    r"\\janeway\User AG Bluhm\Common\GaAs\Hangleiter\InGaAs_dots_M1_12_47_18.db"
)
DATA_PATH = PATH.parent / 'data/ingaas'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

init(MARGINSTYLE, backend := 'pgf')
# %% Functions


def J(tau, alpha, sigma):
    # analytical convolution of exp(-α|τ|) and a Gaussian
    return (
        0.5*np.exp(alpha**2*sigma**2/2 - alpha*tau)
        * sp.special.erfc((alpha*sigma**2 - tau)/(np.sqrt(2)*sigma))
        + 0.5*np.exp(alpha**2*sigma**2/2 + alpha*tau)
        * sp.special.erfc((alpha*sigma**2 + tau)/(np.sqrt(2)*sigma))
    )


def g2_model_convolved(tau, gamma, sigma=350*np.sqrt(2)):
    # analytical convolution of g2_model() and a Gaussian
    return 1 - 2*J(np.abs(tau), gamma/2, sigma) + J(np.abs(tau), gamma, sigma)


def g2_model(tau, gamma):
    return np.square(1 - np.exp(-0.5*gamma*np.abs(tau)))


def lorentz(E, E_0, gamma, A):
    return A * gamma / (gamma**2 + (E - E_0)**2)


# %% Extract runs from database
if EXTRACT_DATA:
    initialise_or_create_database_at(ORIG_DATA_PATH)
    save_to_hdf5(12, DATA_PATH / 'power_dependence.h5')
    save_to_hdf5(20, DATA_PATH / 'g2.h5')

# %% A simple PL spectrum
power_dependence = xr.load_dataset(DATA_PATH / 'power_dependence.h5', engine='h5netcdf')
# Plot with interactive sliders
# fig, ax, sliders = plot_nd(power_dependence, norm=mpl.colors.AsinhNorm(vmin=0))

bounds = [(1.4182, 1.4187), (1.4194, 1.4198), (1.4210, 1.4214)]
fits = []
masks = []
powers = []
for bound in bounds:
    mask = ((power_dependence.ccd_horizontal_axis > bound[0])
            & (power_dependence.ccd_horizontal_axis < bound[1]))
    fit = power_dependence.sel(ccd_horizontal_axis=mask).curvefit(
        'ccd_horizontal_axis',
        lorentz,
        p0={'E_0': np.mean(bound), 'gamma': np.diff(bound), 'A': 1},
        bounds={'E_0': bound, 'gamma': (0, 1e-3), 'A': (0, np.inf)}
    )
    # power = lorentz(
    #     power_dependence.sel(ccd_horizontal_axis=mask),
    #     *fit.ccd_ccd_data_rate_bg_corrected_curvefit_coefficients.T
    # ).integrate('ccd_horizontal_axis')
    power = power_dependence.sel(ccd_horizontal_axis=mask).integrate('ccd_horizontal_axis')

    fits.append(fit)
    masks.append(mask)
    powers.append(power)

# %%% Plot
power_data = power_dependence.ccd_ccd_data_rate_bg_corrected.sel(
    excitation_path_power_at_sample=1e-6
)

fig, ax = plt.subplots(layout='constrained', figsize=(MARGINWIDTH, 1.5))
ax.plot(power_data.ccd_horizontal_axis, power_data * 1e-3)
ax.set_xlabel('$E$ (eV)')
ax.set_ylabel('CCD data rate (kcps)')
ax.set_ylim(top=1)
ax2, unit = secondary_axis(ax, 'eV')
ax2.set_xlabel(rf'$\lambda$ ({unit})')
ax2.set_xticks([868, 871, 874, 877])

fig.savefig(SAVE_PATH / 'ingaas_pl.pdf')

# %%% Plot fits, not used
with plotting.changed_plotting_backend('qtagg'):
    fig, axs = plt.subplots(2, layout='constrained', sharex=True, figsize=(MARGINWIDTH, 2.25))
    axs[0].semilogx(
        power_dependence.excitation_path_power_at_sample * 1e9,
        np.transpose([fit.ccd_ccd_data_rate_bg_corrected_curvefit_coefficients[:, 1] * 1e6
                      for fit in fits])
    )
    axs[1].loglog(
        power_dependence.excitation_path_power_at_sample * 1e9,
        np.transpose([power.ccd_ccd_data_rate_bg_corrected for power in powers]) * const.e * 1e24
    )
    axs[0].grid()
    axs[1].grid()

    axs[1].set_yticks([1, 10, 100, 1_000, 10_000])

    axs[1].set_xlabel('$P$ (nW)')
    axs[1].set_ylabel('$P$ (yW)')
    axs[0].set_ylabel(r'$\gamma$ (μeV)')
# %% G2 measurement
g2 = xr.load_dataset(DATA_PATH / 'g2.h5', engine='h5netcdf')

g2_data = g2.tagger_correlation_1_data_normalized
g2_data_log = g2.tagger_histogram_log_bins_1_g2

mask = np.abs(x := g2_data.tagger_correlation_1_time_bins) < 11e3
popt, pcov = sp.optimize.curve_fit(g2_model_convolved, x[mask], g2_data.data.squeeze()[mask],
                                   p0=[1e-3], bounds=np.transpose([(0, np.inf)]*1))

τmax = g2_data_log.tagger_histogram_log_bins_1_time_bins[g2_data_log.argmax()].item()
print(f'γ = {ufloat(popt[0], np.sqrt(np.diag(pcov))[0])*1e3} GHz')
print(f'Bump is at τ = {τmax} ps = {τmax*1e-12*const.c} m')
# %%% Plot
DOWNSAMPLING = 10
x_down = x[DOWNSAMPLING//2::DOWNSAMPLING]
y_down = np.reshape(g2_data.data, (-1, DOWNSAMPLING)).mean(-1)

fig, axs = plt.subplots(2, layout='constrained', sharey=False, figsize=(MARGINWIDTH, 2.25))
axs[0].plot(x_down * 1e-3, y_down,
            **markerprops(RWTH_COLORS['blue'], marker='.', markeredgealpha=0.75,
                          markerfacealpha=0.25))
axs[0].plot(x[mask]*1e-3, g2_model_convolved(x[mask], *popt))
axs[1].semilogx(g2_data_log.tagger_histogram_log_bins_1_time_bins * 1e-3, g2_data_log.T,
                **markerprops(RWTH_COLORS['blue'], marker='.', markeredgealpha=.75,
                              markerfacealpha=0.25))
axs[1].plot(x[mask]*1e-3, g2_model_convolved(x[mask], *popt))

axs[0].grid()
axs[1].grid()
axs[1].set_yticks([0, 1, 2])
axs[0].set_xlim(-11, 11)
axs[0].set_ylim(0, 1.3)
axs[1].set_ylim(top=2)
axs[1].set_xlabel(r'$\tau$ (ns)')

fig.supylabel(r'$g^{(2)}(\tau)$', fontsize=mpl.rcParams['font.size'])
fig.get_layout_engine().set(hspace=0, wspace=0)
fig.savefig(SAVE_PATH / 'ingaas_g2.pdf')
