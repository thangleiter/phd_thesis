# %% Imports
import json
import pathlib
import sys

import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, odr
import tifffile
import uncertainties.unumpy as unp
from uncertainties import ufloat
import xarray as xr
from qcodes.utils.json_utils import NumpyJSONEncoder

from qutil.plotting import changed_plotting_backend
from qutil.plotting.colors import get_rwth_color_cycle, RWTH_COLORS, RWTH_COLORS_50
from qutil import const, functools, io
from qutil.misc import filter_warnings
from qutil.signal_processing import fourier_space, real_space
from python_spectrometer import Spectrometer

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from common import PATH, TEXTWIDTH, MARGINWIDTH, MAINSTYLE, MARGINSTYLE, MARKERSTYLE  # noqa

ORIG_DATA_PATH = pathlib.Path(
    r'\\janeway\User AG Bluhm\Common\GaAs\Hangleiter\characterization\vibrations'
)
DATA_PATH = PATH.parent / 'data/vibrations'
DATA_PATH.mkdir(exist_ok=True)
SAVE_PATH = PATH / 'pdf/setup'
SAVE_PATH.mkdir(exist_ok=True)

backend = 'qt'
mpl.style.use(MAINSTYLE)

if (ipy := IPython.get_ipython()) is not None:
    ipy.run_line_magic('matplotlib', backend)
# %% Functions


def to_relative_paths(spect, file, *keys):
    with io.changed_directory(DATA_PATH):
        spect.savepath = '.'
        spect.relative_paths = True
        for key in keys:
            spect[key]['comment'] = spect[key]['comment'].replace('x Valhalla puck ', '')
            spect[key]['comment'] = spect[key]['comment'].replace('Optical gate x ', '')
            spect[key]['comment'] = spect[key]['comment'].replace('cold head resting ', '')
            spect[key]['comment'] = spect[key]['comment'].replace(' suspension', ', susp.')
            spect[key]['filepath'] = pathlib.Path(spect[key]['filepath']).name
            if 'station_snapshot' in spect[key]['settings']:
                spect[key]['settings']['station_snapshot'] = json.dumps(
                    spect[key]['settings']['station_snapshot'],
                    cls=NumpyJSONEncoder
                )
            spect.reprocess_data(key, save='overwrite')
        for key in set(spect.keys()).difference(spect._parse_keys(*keys)):
            spect.drop(key)
        spect.serialize_to_disk(file, verbose=True)


# %%% Accelerometer


def sensitivity_deviation(temp_in_kelvin):
    return 1 - 0.05 / 325 * (const.convert_temperature(temp_in_kelvin, 'K', 'F') - 75)


def sensitivity(x, f, temperatures: dict[str: float] | None = None, sensor=None, **kwargs):
    temperatures = temperatures or {}
    match sensor:
        case 'Wilcoxon Research 731-207':
            return x / (9.9 / const.g), f
        case 'PCB 351B42':
            if (T_MC := temperatures.get('MC Plate Cernox', 293)) < 50:
                T_MC = temperatures.get('MC RuO2', 293)
            with filter_warnings('ignore', RuntimeWarning):
                return x / (_spline(np.log10(f) * sensitivity_deviation(T_MC))), f
        case None:
            raise RuntimeError


def comp_gain(x, gain=1, **_):
    return x / gain


# %%% Optical


def sort_files(s):
    return sorted(s, key=lambda s: int(pathlib.Path(s).stem.split("_")[-1]))


def pos_vs_vdc(vdc, a, b):
    return (a*vdc + b)*1e6


def vdc_vs_pos(pos, a, b):
    return (pos*1e-6 - b)/a


def pos_vs_cps(cps, a, b):
    return (cps*1e-6 - b)/a


def cps_calib(x, fs, pos_vs_cps_calibration, **_):
    return pos_vs_cps(x*fs, *pos_vs_cps_calibration)


# %% Calibrations
# %%% Accelerometer
# This is for the PCB 351B42
f_calib = np.array([10, 15, 30, 50, 100, 300, 500, 1000, 2000])  # Hz
s_calib = np.array([0.995, 0.997, 0.998, 0.999, 1, 1.001, 1.002, 1.005, 1.012])*9.99e-3  # V/(m/s²)
_spline = interpolate.interp1d(np.log10(f_calib), s_calib, 'quadratic', fill_value='extrapolate')

f_calib_cond = np.array([1, 10, 100, 1000, 10000])
v_calib_cond = np.array([1.3, 0.1, 0.08, 0.07, 0.07])*1e-6  # V/sqrt(Hz)
x_calib_cond = abs(functools.chain(sensitivity, fourier_space.derivative, n_args=2)(
    v_calib_cond, f_calib_cond, order=-2, sensor='PCB 351B42'
)[0])

f_noise_floor = np.array([1, 10, 100, 1000])
S_noise_floor = abs(fourier_space.derivative(np.array([980, 216, 58.9, 15.7])*1e-6,
                                             f_noise_floor, order=-2)[0])

# %%% Optical
# %%%% Camera calibration
vdc = np.arange(11)
seq = tifffile.FileSequence(tifffile.imread, str(DATA_PATH / 'vdc_mapping/*.tif'),
                            sort=sort_files).asarray()[..., 0] / 255
row = 470
col = np.arange(480, 540)

I_min = 94 / 255
I_max = 121 / 255
seqnan = np.copy(seq)
seqnan[(I_min > seq) | (I_max < seq)] = np.nan

popt = np.empty((len(vdc), vdc.size, 2))
pcov = np.empty((len(vdc), vdc.size, 2, 2))
for j, r in enumerate(np.arange(-5, 6) + row):
    for i, line in enumerate(seqnan[:, r, col]):
        x = col[~np.isnan(line)]
        y = line[~np.isnan(line)]
        popt[i, j], pcov[i, j] = np.polyfit(x, y, 1, cov=True)

pavg, sow = np.average(popt, axis=1, weights=1/pcov[..., range(2), range(2)], returned=True)
perr = 1 / np.sqrt(sow)
a, b = unp.uarray(pavg, perr).T

gate_width_camera = ufloat(116, 3)  # px
gate_width_actual = 14e-6  # m
magnification = gate_width_camera / gate_width_actual  # px/m
I_set = ufloat((I_max + I_min) / 2, 1/255/np.sqrt(12))  # normalized intensity
position = (I_set - b) / a / magnification  # m
position -= position.mean()  # we don't know anything about absolute positions

popt_posvdc, pcov_posvdc = np.polyfit(vdc, unp.nominal_values(position), 1, cov=True,
                                      w=1/unp.std_devs(position))

# %%%%% Plot
erroralpha = 0.5
errorcolor = RWTH_COLORS['blue']
ix = (np.where(~np.isnan(seqnan[0, row, col]))[0][[0, -1]] + (col[0] - 400))

with mpl.style.context(MARGINSTYLE, after_reset=True), changed_plotting_backend('pgf'):
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(MARGINWIDTH, 2.35),
                            gridspec_kw={'height_ratios': [2*const.golden, 1, 1]})
    ax = axs[0]
    ax.imshow(seq[0, row-75:row+75, 400:600], cmap='binary', aspect='equal')
    ax.plot([0, 200], [75-5]*2, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([0, 200], [75+5]*2, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[0]]*2, [0, 150], '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, [0, 150], '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.axis('off')

    ax = axs[1]
    ax.plot(seq[0, row, 400:600], color='k')
    ax.fill_betweenx(lim := ax.get_ylim(), *ix, color=RWTH_COLORS_50['black'], alpha=0.3,
                     linewidth=0.0)
    ax.plot([ix[0]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[2]
    ax.plot(np.gradient(seq[0, row, 400:600], 1), color='k')
    ax.fill_betweenx(lim := ax.get_ylim(), *ix, color=RWTH_COLORS_50['black'], alpha=0.3,
                     linewidth=0.0)
    ax.plot([ix[0]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.plot([ix[1]]*2, lim, '--', color=RWTH_COLORS['black'], alpha=0.3, linewidth=0.5)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(SAVE_PATH / 'knife_edge.pdf')

with mpl.style.context([MARGINSTYLE, {'axes.xmargin': 0.05}]), changed_plotting_backend('pgf'):
    fig, ax = plt.subplots(layout='constrained')
    ax.errorbar(vdc, unp.nominal_values(position)*1e6, unp.std_devs(position)*1e6,
                ecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,),
                marker='o',
                ls='',
                markersize=5,
                markeredgecolor=errorcolor,
                markerfacecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,))
    ax.plot(vdc, np.polyval(popt_posvdc, vdc)*1e6, zorder=5)
    ax.grid()
    ax.set_xlabel(r'$V_\mathrm{DC}$ (V)')
    ax.set_ylabel(r'$x - \langle x\rangle$ (μm)')

    fig.savefig(SAVE_PATH / 'knife_edge_fits.pdf')

# %%%% Count rate calibration
ds = xr.load_dataset(DATA_PATH / 'vdc_calibration.h5')
count_rate = ds.counter_countrate
x = count_rate['positioners_y_axis_voltage']
y1 = count_rate * 1e-6

v_min = 0.5
v_max = 7.0
mask = (v_min <= x) & (x <= v_max)

vdc = x[mask]
pos = pos_vs_vdc(vdc, *popt_posvdc)
poserr = pos_vs_vdc(vdc, *(popt_posvdc + np.sqrt(np.diag(pcov_posvdc)))) - pos
cps = y1[mask].mean('counter_time_axis')
cpserr = y1[mask].std('counter_time_axis') / np.sqrt(count_rate.sizes['counter_time_axis'])

data = odr.Data(pos, cps, wd=poserr, we=cpserr)
model = odr.Model(lambda beta, x: beta[0]*x + beta[1])
fit = odr.ODR(data, model, beta0=[5, 0])
output = fit.run()

# %%%%% Plot
erroralpha = 0.5
errorcolor = RWTH_COLORS['blue']
xx = pos_vs_vdc(x, *popt_posvdc)
xxerr = [xx - pos_vs_vdc(x, *(popt_posvdc - np.sqrt(np.diag(pcov_posvdc)))),
         pos_vs_vdc(x, *(popt_posvdc + np.sqrt(np.diag(pcov_posvdc)))) - xx]

with mpl.style.context([MARGINSTYLE]), changed_plotting_backend('pgf'):
    fig, ax = plt.subplots(layout='constrained',
                           figsize=(MARGINWIDTH, MARGINWIDTH / const.golden * 1.35))
    ax.errorbar(xx, y1.mean('counter_time_axis'),
                y1.std('counter_time_axis') / np.sqrt(count_rate.sizes['counter_time_axis']),
                xerr=xxerr, label='Data',
                ecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,),
                marker='.',
                ls='',
                markersize=5,
                markeredgecolor=errorcolor,
                markerfacecolor=mpl.colors.to_rgb(errorcolor) + (erroralpha,))
    ax.plot(pos, model.fcn(output.beta, pos), zorder=5, label='Fit')

    ax2 = ax.secondary_xaxis(
        'top',
        functions=(functools.partial(vdc_vs_pos, a=popt_posvdc[0], b=popt_posvdc[1]),
                   functools.partial(pos_vs_vdc, a=popt_posvdc[0], b=popt_posvdc[1]))
    )
    ax2.set_xlabel(r'$V_\mathrm{DC}$ (V)')
    ax.set_ylabel('Count rate (MHz)')
    ax.set_xlabel(r'$x - \langle x\rangle$ (μm)')

    ax.set_xlim(*xx[[0, -1]])
    ax.set_ylim(2, 4)

    fig.savefig(SAVE_PATH / 'knife_edge_slope.pdf')

# %% Load spects
with io.changed_directory(DATA_PATH):
    spect_accel = Spectrometer.recall_from_disk('spectrometer_odin_puck', savepath='.')
    spect_optic = Spectrometer.recall_from_disk('spectrometer_photon_counting_23-09-06',
                                                savepath='.')

spects = [spect_accel, spect_optic]

# %%% Apply settings
spect_accel.procfn = functools.chain(comp_gain, functools.scaled(1e6))
spect_accel.psd_estimator = functools.partial(
    real_space.welch, fourier_procfn=(sensitivity, fourier_space.derivative)
)
spect_accel.processed_unit = 'μm'

spect_optic.procfn = cps_calib
spect_optic.reprocess_data(*spect_optic.keys(), pos_vs_cps_calibration=output.beta)

figure_kw = dict(figsize=(TEXTWIDTH, TEXTWIDTH / const.golden * 1.5))
legend_kw = dict(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                 ncols=2, mode="expand", borderaxespad=0., frameon=False)
settings = dict(
    plot_style=MAINSTYLE,
    prop_cycle=get_rwth_color_cycle(100),
    plot_timetrace=False,
    plot_cumulative=True
)


def apply_settings(spect, settings, figure_kw, legend_kw):
    pm = spect._plot_manager

    spect.hide('all')
    plt.close(spect.fig)

    pm.legend_kw.update(legend_kw)
    pm.figure_kw.update(figure_kw)
    for key, val in settings.items():
        setattr(spect, key, val)

    spect.show('all')
    pm._leg = spect.ax[0].legend(labels=[com for _, com in spect.keys()], **pm.legend_kw)


for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
    apply_settings(spect, settings, figure_kw, legend_kw)
    spect.fig.savefig(SAVE_PATH / f'{typ}.pdf', backend='pdf' if backend == 'qt' else backend)

# %%% Resave
# to_relative_paths(spect_accel, 'spectrometer_odin_puck', 2, 3, 4, 5)
# to_relative_paths(spect_optic, 'spectrometer_photon_counting_23-09-06', *spect_optic.keys())

# %%% Vibration criterion
with mpl.style.context([MAINSTYLE, MARKERSTYLE]), changed_plotting_backend('pgf'):
    for typ, spect in zip(['spect_accel', 'spect_optic'], spects):
        fig, ax = plt.subplots(layout='constrained')

        for key, sty in zip(spect.keys(), get_rwth_color_cycle(100)):
            S = np.sqrt(spect[key]['S_processed'].mean(0))
            f = spect[key]['f_processed']

            vc, vc_f = fourier_space.octave_band_rms(*fourier_space.derivative(S, f, order=1),
                                                     fraction=3)

            ax.loglog(vc_f, vc, label=key[1], ls='--', zorder=5,
                      color=sty['color'],
                      markeredgecolor=sty['color'],
                      markerfacecolor=mpl.colors.to_rgb(sty['color']) + (0.5,))

        lim = ax.get_xlim()
        ax.plot([8, lim[1]], [25, 25], marker='', ls='-', color=RWTH_COLORS_50['black'])
        ax.plot([lim[0], 8], [200/lim[0], 25], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-')
        ax.plot([8, lim[1]], [50, 50], marker='', ls='--', color=RWTH_COLORS_50['black'])
        ax.plot([lim[0], 8], [400/lim[0], 50], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='--')
        ax.plot([8, lim[1]], [100, 100], marker='', ls='-.', color=RWTH_COLORS_50['black'])
        ax.plot([lim[0], 8], [800/lim[0], 100], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls='-.')
        ax.plot([8, lim[1]], [200, 200], marker='', ls=':', color=RWTH_COLORS_50['black'])
        ax.plot([lim[0], 8], [1600/lim[0], 200], color=RWTH_COLORS_50['black'], marker='',
                zorder=0, ls=':')

        ax.grid()
        ax.set_xlim(lim)
        ax.set_xlabel(r'$f_\mathrm{center}$ (Hz)')
        ax.set_ylabel(r'$1/3$ octave band $\mathrm{RMS}$ (μm/s)')
        ax.legend(**legend_kw)

        fig.savefig(SAVE_PATH / f'{typ}_vc.pdf')
